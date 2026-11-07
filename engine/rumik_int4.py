"""Our own Rumik-oss-1 inference engine for a small GPU: int4 weights, static KV cache, one CUDA graph per decode step.

Written from the model's published code (Cohere2 + an audio vocabulary + a stop predictor) and the int4 layout of
../modal_export_int4.py. Nothing here is copied from other engines; the techniques (static cache, CUDA graphs,
audio-only head) are the standard ones.

THE MODEL IN ONE PARAGRAPH
Rumik is a 36-layer decoder-only transformer (hidden 2048, 16 query heads / 4 key-value heads of 128, MLP 11008).
Each layer is a *parallel block*: h = LayerNorm(x); x = x + Attention(h) + MLP(h) (attention and MLP read the same h,
so their input projections can be one matrix). 3 of every 4 layers are "sliding" layers that use rotary position
embeddings; every 4th layer uses no position embedding at all. The prompt is text; the model then emits audio tokens,
8 per 80 ms frame (one per Mimi codebook, in order q0..q7). A tiny "stop predictor" MLP decides when speech is over.

WHAT ONE DECODE STEP DOES (the hot loop, ~once per 10 ms)
  token -> embedding row -> 36 x [LayerNorm, qkv+gate+up GEMV, rope, write K/V to cache, attention over the cache,
  o GEMV, silu(gate)*up, down GEMV, add] -> final LayerNorm -> 16,385-row audio head -> sample next token.
Every GEMV reads its whole weight matrix from VRAM for a single token: that is why decode speed = bytes / bandwidth.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))
from int4_gemv import G, gemv_int4  # noqa: E402

WEIGHTS = Path(os.environ.get("RUMIK_WEIGHTS", ROOT / "weights"))     # weights/int4 + weights/hf (download_weights.py)
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------------------------- building blocks
def cohere_ln(x, w, eps=1e-5):
    """Cohere's LayerNorm: subtract the mean, divide by the std, scale by w. No bias. Computed in fp32."""
    xf = x.float()
    xf = xf - xf.mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def rope(x, cos, sin):
    """Rotary embedding, 'interleaved' flavour: pairs (x0,x1), (x2,x3), ... are rotated by position-dependent angles.
    x [heads, T, 128]; cos/sin [T, 128] (each angle repeated twice)."""
    xf = x.float()
    x1, x2 = xf[..., ::2], xf[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (xf * cos.float() + rot * sin.float()).to(x.dtype)


def dequant(q, s, m, out=None, rows=4096):
    """int4 matrix -> bf16 (for the prefill, where a real matrix-matrix product is cheaper than many GEMVs).
    Chunked over rows so the fp32 temporaries stay small."""
    n, k = q.shape[0], q.shape[1] * 2
    out = out if out is not None else torch.empty(n, k, device=q.device, dtype=BF16)
    for r in range(0, n, rows):
        qq = q[r:r + rows]
        code = torch.stack((qq & 0xF, qq >> 4), -1).view(qq.shape[0], -1, G).float()
        out[r:r + rows] = (code * s[r:r + rows, :, None].float() + m[r:r + rows, :, None].float()).view(-1, k)
    return out


# ---------------------------------------------------------------------------------------------- the engine
class RumikInt4:
    def __init__(self, max_seq=2048, kv_buckets=(256, 512, 1024, 2048), temperature=0.8, top_k=30,
                 min_new_tokens=8, max_new_tokens=1800, device="cuda"):
        cfg = json.loads((WEIGHTS / "hf/config.json").read_text())
        self.cfg, self.dev = cfg, device
        self.L, self.H = cfg["num_hidden_layers"], cfg["hidden_size"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.inter, self.eps = cfg["intermediate_size"], cfg["layer_norm_eps"]
        self.sliding = [t == "sliding_attention" for t in cfg["layer_types"]]
        self.Q, self.end_id = cfg["num_quantizers"], cfg["audio_end_token_id"]
        self.first, self.last = cfg["first_unit_id"], cfg["last_unit_id"]
        self.max_seq, self.kv_buckets = max_seq, tuple(b for b in kv_buckets if b <= max_seq)
        self.temperature, self.top_k = temperature, top_k
        self.min_new, self.max_new = min_new_tokens, max_new_tokens
        self._load()
        self._alloc()
        self.graphs = {}

    # ---- weights: the int4 file -> GPU. q/k/v/gate/up are concatenated by rows into ONE matrix per layer, because
    # in a parallel block they all multiply the same vector h: one bigger GEMV instead of five small ones.
    def _load(self):
        t = time.time()
        f = safe_open(str(WEIGHTS / "int4/rumik-oss-1-int4.safetensors"), "pt", device=self.dev)
        get = f.get_tensor
        self.w_in, self.w_o, self.w_down, self.ln = [], [], [], []
        for i in range(self.L):
            p = f"model.layers.{i}."
            parts = [p + "self_attn." + n for n in ("q_proj", "k_proj", "v_proj")] + \
                    [p + "mlp." + n for n in ("gate_proj", "up_proj")]
            self.w_in.append(tuple(torch.cat([get(n + suf) for n in parts]).contiguous()
                                   for suf in (".qweight", ".scale", ".min")))
            self.w_o.append(tuple(get(p + "self_attn.o_proj" + suf) for suf in (".qweight", ".scale", ".min")))
            self.w_down.append(tuple(get(p + "mlp.down_proj" + suf) for suf in (".qweight", ".scale", ".min")))
            self.ln.append(get(p + "input_layernorm.weight"))
        self.norm_w = get("model.norm.weight")
        self.emb_q, self.emb_s = get("model.embed_tokens.qweight"), get("model.embed_tokens.scale")
        # audio-only head: the model ties its output layer to the embedding table (277k rows), but inside <audio>
        # only 16,385 tokens are legal (8 codebooks x 2048 codes + </audio>), so we keep just those rows, exact bf16
        self.head_w, self.audio_ids = get("head.audio_weight"), get("head.audio_ids")
        self.stop = {k: get(f"stop_predictor.{k}") for k in ("0.weight", "0.bias", "1.weight", "1.bias",
                                                              "3.weight", "3.bias")}
        # rotary tables, as the reference computes them (fp32 angles, stored in bf16)
        inv = 1.0 / (self.cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, device=self.dev).float() / self.hd))
        ang = torch.outer(torch.arange(self.max_seq, device=self.dev).float(), inv).repeat_interleave(2, -1)
        self.cos, self.sin = ang.cos().to(BF16), ang.sin().to(BF16)
        torch.cuda.synchronize()
        print(f"weights loaded in {time.time() - t:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    # ---- static buffers: a CUDA graph replays fixed memory addresses, so every tensor the step touches is
    # allocated once here and updated in place (token, position, step counter, KV cache, outputs).
    def _alloc(self):
        d = self.dev
        self.kc = torch.zeros(self.L, self.nkv, self.max_seq, self.hd, device=d, dtype=BF16)
        self.vc = torch.zeros_like(self.kc)
        self.tok = torch.zeros(1, dtype=torch.long, device=d)        # current input token (an id in the full vocab)
        self.pos = torch.zeros(1, dtype=torch.long, device=d)        # its position
        self.step_i = torch.zeros(1, dtype=torch.long, device=d)     # how many tokens generated so far
        self.hn = torch.zeros(self.H, device=d, dtype=BF16)          # final normed hidden of the last forward
        self.out = torch.zeros(self.max_new + 8, dtype=torch.long, device=d)
        self.ar = torch.arange(self.max_seq, device=d)
        self.is_end = self.audio_ids == self.end_id                    # the last of the 16,385 head rows
        n_in = self.w_in[0][0].shape[0]
        self.buf_in = torch.empty(n_in, device=d, dtype=BF16)
        self.buf_o = torch.empty(self.H, device=d, dtype=BF16)
        self.buf_d = torch.empty(self.H, device=d, dtype=BF16)

    def embed(self, ids):
        return (self.emb_q[ids].float() * self.emb_s[ids, None].float()).to(BF16)

    # ---- prefill: the whole prompt at once (T tokens), with bf16 matmuls on dequantized weights
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
            y = h @ dequant(*self.w_in[i]).t()                                  # [T, 2048+512+512+2*11008]
            q = y[:, :nq].view(T, self.nh, self.hd).transpose(0, 1)             # [16, T, 128]
            k = y[:, nq:nq + nk].view(T, self.nkv, self.hd).transpose(0, 1)     # [4, T, 128]
            v = y[:, nq + nk:nq + 2 * nk].view(T, self.nkv, self.hd).transpose(0, 1)
            g, u = y[:, nq + 2 * nk:].split(self.inter, -1)
            if self.sliding[i]:
                q, k = rope(q, cos, sin), rope(k, cos, sin)
            self.kc[i, :, :T], self.vc[i, :, :T] = k, v
            o = self._attend(q, k, v, causal)                                    # [T, 2048]
            x = x + o @ dequant(*self.w_o[i]).t() + (F.silu(g) * u) @ dequant(*self.w_down[i]).t()
        self.hn.copy_(cohere_ln(x[-1], self.norm_w, self.eps))
        self.pos.fill_(T)
        self.step_i.zero_()
        return self._sample_token_eager()

    def _attend(self, q, k, v, bias):
        """Grouped-query attention: 16 query heads share 4 K/V heads (4 queries per K/V head). fp32 scores."""
        nh, T = q.shape[0], q.shape[1]
        rep = nh // k.shape[0]
        qf = q.float().contiguous().view(k.shape[0], rep * T, self.hd)
        s = (qf @ k.float().transpose(1, 2)).view(k.shape[0], rep, T, -1) * self.hd ** -0.5 + bias
        p = torch.softmax(s, -1).view(k.shape[0], rep * T, -1)
        o = (p @ v.float()).view(nh, T, self.hd)
        return o.transpose(0, 1).reshape(T, nh * self.hd).to(BF16)

    # ---- the next token from self.hn: audio head + stop predictor + top-k sampling (all on the GPU)
    def _logits_and_stop(self):
        logits = (self.head_w @ self.hn).float()                                # [16385]
        st = self.stop
        z = F.layer_norm(self.hn.float(), (self.H,), st["0.weight"].float(), st["0.bias"].float(), 1e-5)
        z = F.gelu(F.linear(z, st["1.weight"].float(), st["1.bias"].float()))
        stop = torch.sigmoid(F.linear(z, st["3.weight"].float(), st["3.bias"].float())) > 0.5
        return logits, stop

    def _pick(self, logits, stop):
        early = self.step_i < self.min_new
        logits = torch.where(early & self.is_end, float("-inf"), logits)         # no </audio> in the first 8
        top_v, top_i = logits.topk(self.top_k)
        # sampling from softmax(top_v / T) == argmax(top_v / T + Gumbel noise): no host sync, graph-friendly
        gumbel = -torch.log(-torch.log(torch.rand(self.top_k, device=self.dev).clamp_(1e-10, 1.0)))
        # (index with index_select/gather, never t[gpu_scalar]: that hides a GPU->CPU sync, which graphs forbid)
        choice = top_i.gather(0, torch.argmax(top_v / self.temperature + gumbel).view(1))
        nxt = self.audio_ids.index_select(0, choice)[0]
        return torch.where(stop.view(()) & ~early.view(()), torch.full_like(nxt, self.end_id), nxt)

    def _sample_token_eager(self):
        nxt = self._pick(*self._logits_and_stop())
        self.out[self.step_i] = nxt
        self.tok.copy_(nxt.view(1))
        self.step_i += 1
        return nxt

    # ---- one decode step for a cache bucket of Lb slots. Only reads/writes the static buffers above.
    def _step(self, Lb):
        x = self.embed(self.tok)[0]                                             # [H]
        cos, sin = self.cos[self.pos], self.sin[self.pos]                       # [1, 128]
        bias = torch.where(self.ar[:Lb] <= self.pos, 0.0, float("-inf"))        # attend to positions 0..pos
        nq, nk = self.nh * self.hd, self.nkv * self.hd
        for i in range(self.L):
            h = cohere_ln(x, self.ln[i], self.eps)
            y = gemv_int4(h, *self.w_in[i], out=self.buf_in)
            q = y[:nq].view(self.nh, 1, self.hd)
            k = y[nq:nq + nk].view(self.nkv, 1, self.hd)
            v = y[nq + nk:nq + 2 * nk].view(self.nkv, 1, self.hd)
            g, u = y[nq + 2 * nk:].split(self.inter)
            if self.sliding[i]:
                q, k = rope(q, cos, sin), rope(k, cos, sin)
            self.kc[i].index_copy_(1, self.pos, k)                              # write this token's K/V
            self.vc[i].index_copy_(1, self.pos, v)
            o = self._attend(q, self.kc[i, :, :Lb], self.vc[i, :, :Lb], bias)[0]
            a = gemv_int4(o, *self.w_o[i], out=self.buf_o)
            mm = gemv_int4((F.silu(g) * u).contiguous(), *self.w_down[i], out=self.buf_d)
            x = x + a + mm
        self.hn.copy_(cohere_ln(x, self.norm_w, self.eps))
        self.pos += 1
        self._sample_token_eager()

    def capture(self):
        """Record one CUDA graph per KV bucket. Replaying a graph launches its ~1,000 small kernels with one CPU call,
        which removes the Python/launch overhead (otherwise ~2-5 us x 1,000 per token)."""
        t = time.time()
        save = (self.kc.clone(), self.vc.clone())
        self.pos.fill_(1); self.step_i.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for Lb in self.kv_buckets:                                         # warm-up: Triton autotune + JIT
                self.pos.fill_(1)
                self._step(Lb)
        torch.cuda.current_stream().wait_stream(s)
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            for Lb in self.kv_buckets:
                self.pos.fill_(1)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    self._step(Lb)
                self.graphs[Lb] = g
        self.kc.copy_(save[0]); self.vc.copy_(save[1])
        torch.cuda.synchronize()
        print(f"captured {len(self.graphs)} graphs in {time.time() - t:.1f}s", flush=True)

    # ---- generation: prefill, then replay graphs; read tokens back once per frame (8 tokens)
    @torch.inference_mode()
    def generate(self, ids, seed=0, on_frame=None):
        torch.manual_seed(seed)
        if not self.graphs:
            self.capture()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = int(self.prefill(ids))
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        pos, n, toks, t_frame0 = len(ids), 1, [first], None
        done = first == self.end_id
        # Double buffering: queue frame k+1 on the GPU *before* waiting for frame k, so the GPU never idles while
        # Python handles a frame. Tokens come back through an async copy to pinned memory + a CUDA event (waiting on
        # an event waits only for the work before it, unlike .tolist(), which waits for the whole queue).
        host = torch.empty(self.out.shape[0], dtype=torch.long, pin_memory=True)
        queued = []                                                            # (n, event) per frame in flight

        def launch():
            nonlocal pos, n
            for _ in range(self.Q):
                Lb = next(b for b in self.kv_buckets if b > pos)
                self.graphs[Lb].replay()
                pos += 1
            host[n:n + self.Q].copy_(self.out[n:n + self.Q], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            queued.append((n, ev))
            n += self.Q

        def can_launch():
            return n < self.max_new and pos + self.Q < self.max_seq

        if not done and can_launch():
            launch()
        while queued:
            if not done and can_launch():
                launch()                                                       # keep one frame ahead
            k, ev = queued.pop(0)
            ev.synchronize()
            new = host[k:k + self.Q].tolist()
            if t_frame0 is None:
                t_frame0 = time.perf_counter() - t0
            if self.end_id in new:
                new = new[:new.index(self.end_id)]
                done = True
            toks += new
            if on_frame:
                on_frame(toks)
            if done:
                queued.clear()                                                 # the frame after </audio> is ignored
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        n_dec = max(len(toks) - 1, 1)
        return toks, {"prompt_tokens": len(ids), "tokens": len(toks), "prefill_ms": round(t_prefill * 1e3, 1),
                      "first_frame_ms": round((t_frame0 or total) * 1e3, 1),
                      "decode_tok_s": round(n_dec / (total - t_prefill), 1), "audio_s": round(len(toks) / 100, 2),
                      "total_s": round(total, 2)}

    def to_codes(self, toks):
        """audio token ids -> [1, 8, frames] Mimi codes (whole frames only; resync on a broken round robin)"""
        frames, fr = [], []
        for t in toks:
            if not self.first <= t <= self.last:
                fr = []
                continue
            code, q = divmod(t - self.first, self.Q)
            if q == len(fr):
                fr.append(code)
                if len(fr) == self.Q:
                    frames.append(fr); fr = []
            else:
                fr = [code] if q == 0 else []
        return torch.tensor(frames, dtype=torch.long).T.unsqueeze(0)
