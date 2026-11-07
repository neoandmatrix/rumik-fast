"""RumikInt4 with the fused decode step (fused.py). Same weights, prefill, sampling and generate loop; only _step
changes. One CUDA graph covers every sequence length (attention reads `pos` on the GPU), so there are no KV buckets.
"""
import torch

import fused
from rumik_int4 import BF16, RumikInt4, gemv_int4
from cuda_gemv import gemv_int4_cuda, gemv_int8_cuda, quantize_int8_rows   # ../kernels (on the path via rumik_int4)
import mm_int4 as _mm
from mm_int4 import mm_int4
from int4_gemv import quantize
from sample import FusedSampler
from rumik_int4 import cohere_ln, rope
import torch.nn.functional as F

MLP_BLOCK = 1024
NSPLIT, BT = 16, 16          # flash-decoding: chunks per KV head, slots per inner block (attn_sweep: best)


class RumikFused(RumikInt4):
    def __init__(self, max_seq=2048, split_attn=True, cuda_gemv=True, head="int8", fused_sample=True, **kw):
        kw.pop("kv_buckets", None)
        self.split_attn = split_attn
        self.gemv = gemv_int4_cuda if cuda_gemv else gemv_int4     # hand-written CUDA (step 7) or Triton v1
        super().__init__(max_seq=max_seq, kv_buckets=(max_seq,), **kw)
        # audio head (step 10): "int8" (per-row scale, CUDA GEMV; default), "int4" (too lossy for the head: KL 0.1-0.15
        # vs bf16, check_sample.py) or the exact "bf16". Rows padded to a multiple of 16; padded logits are never read.
        assert head in ("int8", "int4", "bf16")
        self.head = head
        nv = self.head_w.shape[0]
        pad = (-nv) % 16
        if head == "int8":
            self.head_q = quantize_int8_rows(self.head_w)
        elif head == "int4":
            w = torch.cat([self.head_w, self.head_w.new_zeros(pad, self.H)]) if pad else self.head_w
            self.head_q = quantize(w)
        self.logits_b = torch.empty(nv + (pad if head != "bf16" else 0), device=self.dev, dtype=BF16)
        self.sampler = FusedSampler(self, self._logits) if fused_sample else None

    def _alloc(self):
        super()._alloc()
        d = self.dev
        self.x = torch.zeros(self.H, device=d, dtype=BF16)            # residual stream
        self.h = torch.zeros(self.H, device=d, dtype=BF16)            # LayerNorm(x), input of the big GEMV
        self.qb = torch.zeros(self.nh * self.hd, device=d, dtype=BF16)
        self.ob = torch.zeros(self.nh * self.hd, device=d, dtype=BF16)
        self.act = torch.zeros(self.inter, device=d, dtype=BF16)
        f32 = torch.float32
        self.po = torch.zeros(self.nh, NSPLIT, self.hd, device=d, dtype=f32)   # flash-decoding partials
        self.pm = torch.zeros(self.nh, NSPLIT, device=d, dtype=f32)
        self.pl = torch.zeros(self.nh, NSPLIT, device=d, dtype=f32)
        self.n_post = self.nh + 2 * self.nkv + triton_cdiv(self.inter, MLP_BLOCK)

    # ---- prefill with int4 matmuls (step 8): the weights stay int4; no temporary bf16 copies
    @torch.inference_mode()
    def prefill(self, ids):
        T = len(ids)
        ids_t = torch.tensor(ids, device=self.dev)
        x = self.embed(ids_t)                                                   # [T, H]
        cos, sin = self.cos[:T], self.sin[:T]
        causal = torch.full((T, T), float("-inf"), device=self.dev).triu(1)
        nq, nk = self.nh * self.hd, self.nkv * self.hd
        for i in range(self.L):
            h = cohere_ln(x, self.ln[i], self.eps)
            y = mm_int4(h, *self.w_in[i])                                        # [T, q|k|v|gate|up]
            q = y[:, :nq].view(T, self.nh, self.hd).transpose(0, 1)
            k = y[:, nq:nq + nk].view(T, self.nkv, self.hd).transpose(0, 1)
            v = y[:, nq + nk:nq + 2 * nk].view(T, self.nkv, self.hd).transpose(0, 1)
            g, u = y[:, nq + 2 * nk:].split(self.inter, -1)
            if self.sliding[i]:
                q, k = rope(q, cos, sin), rope(k, cos, sin)
            self.kc[i, :, :T], self.vc[i, :, :T] = k, v
            o = self._attend(q, k, v, causal)
            x = x + mm_int4(o, *self.w_o[i]) + mm_int4(F.silu(g) * u, *self.w_down[i])
        self.hn.copy_(cohere_ln(x[-1], self.norm_w, self.eps))
        self.pos.fill_(T)
        self.step_i.zero_()
        return self._sample_token_eager()

    def capture(self):
        _mm.warmup([tuple(w[0].shape[:1]) + (w[0].shape[1] * 2,) for w in (self.w_in[0], self.w_o[0], self.w_down[0])])
        super().capture()

    def _logits(self):
        if self.head == "int8":
            return gemv_int8_cuda(self.hn, *self.head_q, out=self.logits_b)
        if self.head == "int4":
            return gemv_int4_cuda(self.hn, *self.head_q, out=self.logits_b)
        return torch.mv(self.head_w, self.hn, out=self.logits_b)

    def _sample_token_eager(self):
        if self.sampler is None:
            return super()._sample_token_eager()
        self.sampler()                                        # writes out[step_i], tok; step_i += 1
        return self.tok[0]

    def _step(self, Lb=None):
        S, eps = self.max_seq, self.eps
        fused.embed_ln[(1,)](self.tok, self.emb_q, self.emb_s, self.ln[0], self.x, self.h, eps, H=self.H,
                             num_warps=8)
        for i in range(self.L):
            self.gemv(self.h, *self.w_in[i], out=self.buf_in)
            fused.post[(self.n_post,)](self.buf_in, self.pos, self.cos, self.sin, self.kc[i], self.vc[i], self.qb,
                                       self.act, S, self.inter, NH=self.nh, NKV=self.nkv, HD=self.hd,
                                       ROPE=self.sliding[i], BLOCK=MLP_BLOCK, num_warps=4)
            if self.split_attn:
                fused.attn_split[(self.nkv, NSPLIT)](self.qb, self.kc[i], self.vc[i], self.pos, self.po, self.pm,
                                                     self.pl, S, self.hd ** -0.5, REP=self.nh // self.nkv,
                                                     HD=self.hd, NSPLIT=NSPLIT, BT=BT, num_warps=2, PV_BF16=True)
                fused.attn_combine[(self.nh,)](self.po, self.pm, self.pl, self.pos, self.ob, HD=self.hd,
                                               NSPLIT=NSPLIT, BT=BT, num_warps=4)
            else:
                fused.attn[(self.nh,)](self.qb, self.kc[i], self.vc[i], self.pos, self.ob, S, self.hd ** -0.5,
                                       REP=self.nh // self.nkv, HD=self.hd, BT=64, num_warps=4)
            self.gemv(self.ob, *self.w_o[i], out=self.buf_o)
            self.gemv(self.act, *self.w_down[i], out=self.buf_d)
            last = i == self.L - 1
            fused.add_ln[(1,)](self.x, self.buf_o, self.buf_d, self.norm_w if last else self.ln[i + 1],
                               self.hn if last else self.h, eps, H=self.H, num_warps=8)
        self.pos += 1
        self._sample_token_eager()


def triton_cdiv(a, b):
    return (a + b - 1) // b
