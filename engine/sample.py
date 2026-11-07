"""Fused "tail" of a decode step: stop predictor + top-k + Gumbel sampling + counter updates in 3 Triton kernels
(before: ~25 PyTorch kernels, including torch.topk and an fp32 copy of the stop predictor's 4 MB weights per step).

Same maths as RumikInt4._logits_and_stop / _pick:
  stop   = sigmoid(W2 . gelu(W1 . LayerNorm(h) + b1) + b2) > 0.5     (only after min_new tokens)
  logits = audio head . h  (16,385 audio rows; </audio> forbidden for the first min_new tokens)
  token  = argmax over the top-30 logits of (logit / T + Gumbel noise)  == a sample from softmax(top30 / T)

Top-k without torch.topk: every logit and its index are packed into ONE int64 key, so sorting keys sorts values and
carries the index along. Float bits are made sortable as integers by flipping the 31 value bits of negatives:
  b = bits(x);  key32 = b ^ ((b >> 31) & 0x7FFFFFFF)       (monotonic in x, also for -inf)
  key = key32 << 32 | index
Stage 1 (NCH programs): sort 512 keys, keep them (the first 32 are the chunk's top 32).
Stage 2 (1 program): gather the NCH x 32 survivors, sort, take the top 30 -> sample.
"""
import torch
import triton
import triton.language as tl

CH = 512          # logits per stage-1 program
KEEP = 32         # survivors per chunk (>= top_k)


@triton.jit
def _sortable(x):
    b = x.to(tl.int32, bitcast=True)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@triton.jit
def _unsortable(k32):
    b = k32 ^ ((k32 >> 31) & 0x7FFFFFFF)
    return b.to(tl.float32, bitcast=True)


# ---- stop predictor, hidden layer: z1 = gelu(W1 . LN(h) + b1). Grid: 512 / ROWS programs; each recomputes the
# (cheap) LayerNorm of h and does ROWS rows of W1.
@triton.jit
def stop_hidden(h_ptr, lnw_ptr, lnb_ptr, w1_ptr, b1_ptr, z1_ptr, eps, H: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, H)
    x = tl.load(h_ptr + r).to(tl.float32)
    mu = tl.sum(x, 0) / H
    d = x - mu
    z = d * tl.rsqrt(tl.sum(d * d, 0) / H + eps) * tl.load(lnw_ptr + r).to(tl.float32) + \
        tl.load(lnb_ptr + r).to(tl.float32)
    rows = pid * ROWS + tl.arange(0, ROWS)
    w = tl.load(w1_ptr + rows[:, None] * H + r[None, :]).to(tl.float32)                 # [ROWS, H]
    y = tl.sum(w * z[None, :], 1) + tl.load(b1_ptr + rows).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))                         # exact GELU (F.gelu)
    tl.store(z1_ptr + rows, y)


# ---- stage 1: sort one chunk of logits as packed keys
@triton.jit
def topk_partial(logits_ptr, step_ptr, keys_ptr, NV, END_IDX, MIN_NEW, CH: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * CH + tl.arange(0, CH)
    x = tl.load(logits_ptr + i, mask=i < NV, other=float("-inf")).to(tl.float32)
    early = tl.load(step_ptr) < MIN_NEW
    x = tl.where(early & (i == END_IDX), float("-inf"), x)                              # no </audio> yet
    key = (_sortable(x).to(tl.int64) << 32) | i.to(tl.int64)
    tl.store(keys_ptr + pid * CH + tl.arange(0, CH), tl.sort(key, descending=True))


# ---- stage 2: merge the chunks' top-KEEP, take the top TOPK, Gumbel-max sample, stop rule, write the token
@triton.jit
def pick(keys_ptr, scratch_ptr, u_ptr, z1_ptr, w2_ptr, b2_ptr, ids_ptr, step_ptr, pos_ptr, tok_ptr, out_ptr,
         inv_temp, END_ID, MIN_NEW, NCH, CH: tl.constexpr, KEEP: tl.constexpr, NC: tl.constexpr,
         TOPK: tl.constexpr, NZ: tl.constexpr):
    c = tl.arange(0, NC)                                                 # candidate slot: chunk c // KEEP
    ch = c // KEEP
    cand = tl.load(keys_ptr + ch * CH + c % KEEP, mask=ch < NCH, other=-(2 ** 62))
    tl.store(scratch_ptr + c, tl.sort(cand, descending=True))
    tl.debug_barrier()                                                   # make the sorted list visible to all threads
    j = tl.arange(0, 32)
    top = tl.load(scratch_ptr + j)                                       # the top 32, best first
    val = _unsortable((top >> 32).to(tl.int32))
    idx = (top & 0xFFFFFFFF).to(tl.int32)
    u = tl.load(u_ptr + j, mask=j < TOPK, other=0.5)
    g = -tl.log(-tl.log(tl.minimum(tl.maximum(u, 1e-10), 1.0)))
    score = tl.where(j < TOPK, val * inv_temp + g, float("-inf"))
    best = tl.argmax(score, 0)
    choice = tl.sum(tl.where(j == best, idx, 0), 0)
    nxt = tl.load(ids_ptr + choice)
    zr = tl.arange(0, NZ)                                                # stop predictor output layer
    s = tl.sum(tl.load(z1_ptr + zr) * tl.load(w2_ptr + zr).to(tl.float32), 0) + tl.load(b2_ptr).to(tl.float32)
    step = tl.load(step_ptr)
    nxt = tl.where((s > 0.0) & (step >= MIN_NEW), END_ID, nxt)          # sigmoid(s) > 0.5  <=>  s > 0
    tl.store(out_ptr + step, nxt)
    tl.store(tok_ptr, nxt)
    tl.store(step_ptr, step + 1)


class FusedSampler:
    """Holds the buffers; `__call__` samples the next token from eng.hn and updates eng.out/tok/step_i in place."""

    def __init__(self, eng, logits_fn):
        d = eng.dev
        self.eng, self.logits_fn = eng, logits_fn
        st = eng.stop
        self.lnw, self.lnb = st["0.weight"].float().contiguous(), st["0.bias"].float().contiguous()
        self.w1, self.b1 = st["1.weight"].contiguous(), st["1.bias"].float().contiguous()     # bf16 W1 (exact)
        self.w2, self.b2 = st["3.weight"].reshape(-1).float().contiguous(), st["3.bias"].float().contiguous()
        self.nz = self.w1.shape[0]
        self.nv = eng.audio_ids.numel()
        ends = (eng.audio_ids == eng.end_id).nonzero().flatten().tolist()
        assert len(ends) == 1
        self.end_idx = ends[0]
        self.nch = triton.cdiv(self.nv, CH)
        self.nc = triton.next_power_of_2(self.nch * KEEP)
        self.keys = torch.empty(self.nch * CH, dtype=torch.int64, device=d)
        self.scratch = torch.empty(self.nc, dtype=torch.int64, device=d)
        self.z1 = torch.empty(self.nz, dtype=torch.float32, device=d)
        self.u = torch.empty(32, dtype=torch.float32, device=d)
        assert eng.top_k <= 32

    def __call__(self):
        e = self.eng
        logits = self.logits_fn()                                        # [>= nv], bf16 or fp32
        stop_hidden[(self.nz // 16,)](e.hn, self.lnw, self.lnb, self.w1, self.b1, self.z1, 1e-5, H=e.H, ROWS=16,
                                       num_warps=4)
        topk_partial[(self.nch,)](logits, e.step_i, self.keys, self.nv, self.end_idx, e.min_new, CH=CH, num_warps=4)
        torch.rand(32, device=e.dev, out=self.u)                         # graph-safe RNG (Philox offsets)
        pick[(1,)](self.keys, self.scratch, self.u, self.z1, self.w2, self.b2, e.audio_ids, e.step_i, e.pos, e.tok,
                   e.out, 1.0 / e.temperature, e.end_id, e.min_new, self.nch, CH=CH, KEEP=KEEP, NC=self.nc,
                   TOPK=e.top_k, NZ=self.nz, num_warps=4)
