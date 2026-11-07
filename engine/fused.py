"""Fused Triton kernels for the decode step: everything between the int4 GEMVs.

Before fusion one layer was ~44 small PyTorch kernels (norm pieces, rope pieces, casts, copies, cache writes, an fp32
copy of the whole KV cache for attention, ...). Each one costs a few microseconds just to start and finish, and the
GPU sits mostly idle in between. Here a layer is:

  gemv(w_in) -> post -> attn -> gemv(w_o), gemv(w_down) -> add_ln          (6 launches instead of ~44)

Shapes (Rumik): hidden H=2048, 16 query heads + 4 key/value heads of HD=128, MLP width 11008.
The fused projection output y = [q (16*128) | k (4*128) | v (4*128) | gate (11008) | up (11008)].
"""
import triton
import triton.language as tl


# ---- A. token -> embedding row (int8 * per-row scale) -> x, and h = LayerNorm(x) for layer 0
@triton.jit
def embed_ln(tok_ptr, eq_ptr, es_ptr, w_ptr, x_ptr, h_ptr, eps, H: tl.constexpr):
    r = tl.arange(0, H)
    t = tl.load(tok_ptr)                                                     # the token id, read on the GPU
    e = tl.load(eq_ptr + t * H + r).to(tl.float32) * tl.load(es_ptr + t).to(tl.float32)
    xb = e.to(tl.bfloat16)                                                   # the residual stream is bf16
    tl.store(x_ptr + r, xb)
    xf = xb.to(tl.float32)
    d = xf - tl.sum(xf, 0) / H                                               # Cohere LayerNorm: centre ...
    h = d * tl.rsqrt(tl.sum(d * d, 0) / H + eps) * tl.load(w_ptr + r).to(tl.float32)   # ... scale, no bias
    tl.store(h_ptr + r, h.to(tl.bfloat16))


# ---- D. x = x + attn_out + mlp_out (summed in fp32, rounded once), then h = LayerNorm(x) with the NEXT layer's
# weight (for the last layer: the final norm, written straight into the head's input).
@triton.jit
def add_ln(x_ptr, a_ptr, b_ptr, w_ptr, h_ptr, eps, H: tl.constexpr):
    r = tl.arange(0, H)
    s = tl.load(x_ptr + r).to(tl.float32) + tl.load(a_ptr + r).to(tl.float32) + tl.load(b_ptr + r).to(tl.float32)
    xb = s.to(tl.bfloat16)
    tl.store(x_ptr + r, xb)
    xf = xb.to(tl.float32)
    d = xf - tl.sum(xf, 0) / H
    h = d * tl.rsqrt(tl.sum(d * d, 0) / H + eps) * tl.load(w_ptr + r).to(tl.float32)
    tl.store(h_ptr + r, h.to(tl.bfloat16))


# ---- B. everything right after the big GEMV, in ONE launch. Program ids are split into jobs:
#   0..NH-1            rope one query head  -> q buffer
#   NH..NH+NKV-1       rope one key head    -> KV cache slot `pos`
#   next NKV           copy one value head  -> KV cache slot `pos`
#   the rest           silu(gate) * up for BLOCK MLP channels -> act buffer (input of the down GEMV)
# Rope is the "interleaved" kind: pairs (2j, 2j+1) are rotated by angle pos * theta_j.
@triton.jit
def post(y_ptr, pos_ptr, cos_ptr, sin_ptr, kc_ptr, vc_ptr, q_ptr, act_ptr, S, INTER,
         NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, ROPE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(pos_ptr)
    if pid < NH + NKV:                                         # a q or k head (they sit next to each other in y)
        j = tl.arange(0, HD // 2)
        xe = tl.load(y_ptr + pid * HD + 2 * j).to(tl.float32)  # even elements of the pairs
        xo = tl.load(y_ptr + pid * HD + 2 * j + 1).to(tl.float32)
        if ROPE:
            c = tl.load(cos_ptr + pos * HD + 2 * j).to(tl.float32)
            s = tl.load(sin_ptr + pos * HD + 2 * j).to(tl.float32)
            oe = xe * c - xo * s
            oo = xo * c + xe * s
        else:                                                  # every 4th layer has no position embedding
            oe = xe
            oo = xo
        if pid < NH:
            dst = q_ptr + pid * HD
        else:
            dst = kc_ptr + ((pid - NH) * S + pos) * HD         # cache layout [NKV, S, HD]
        tl.store(dst + 2 * j, oe.to(tl.bfloat16))
        tl.store(dst + 2 * j + 1, oo.to(tl.bfloat16))
    elif pid < NH + 2 * NKV:                                   # a v head: straight into the cache
        vh = pid - NH - NKV
        rv = tl.arange(0, HD)        # (names must differ per branch: Triton unifies same-named variables)
        v = tl.load(y_ptr + (NH + NKV) * HD + vh * HD + rv)
        tl.store(vc_ptr + (vh * S + pos) * HD + rv, v)
    else:                                                      # the MLP activation
        rm = (pid - NH - 2 * NKV) * BLOCK + tl.arange(0, BLOCK)
        m = rm < INTER
        off = (NH + 2 * NKV) * HD
        g = tl.load(y_ptr + off + rm, mask=m, other=0.0).to(tl.float32)
        u = tl.load(y_ptr + off + INTER + rm, mask=m, other=0.0).to(tl.float32)
        tl.store(act_ptr + rm, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=m)


# ---- C. decode attention for one query head per program, over cache slots 0..pos.
# "Online softmax" (the FlashAttention trick): walk the cache in blocks of BT slots, keep a running max m, a running
# sum l and a running weighted sum acc; when a new block raises the max, rescale the old l and acc by exp(m_old-m_new).
# So we never store all the scores, never copy the cache to fp32, and only read the slots that are filled. The loop
# bound comes from `pos` on the GPU, so one CUDA graph works for every length (no KV buckets needed).
@triton.jit
def attn(q_ptr, kc_ptr, vc_ptr, pos_ptr, o_ptr, S, scale, REP: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr):
    h = tl.program_id(0)
    kvh = h // REP                                             # grouped-query: 4 query heads share a k/v head
    pos = tl.load(pos_ptr)
    d = tl.arange(0, HD)
    t = tl.arange(0, BT)
    q = tl.load(q_ptr + h * HD + d).to(tl.float32)
    kb = kc_ptr + kvh * S * HD
    vb = vc_ptr + kvh * S * HD
    m_i = tl.zeros([], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HD], dtype=tl.float32)
    for t0 in range(0, pos + 1, BT):
        tt = t0 + t
        msk = tt <= pos
        k = tl.load(kb + tt[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        s = tl.where(msk, tl.sum(k * q[None, :], 1) * scale, float("-inf"))     # [BT] scores
        m_new = tl.maximum(m_i, tl.max(s, 0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(vb + tt[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m_i = m_new
    tl.store(o_ptr + h * HD + d, (acc / l_i).to(tl.bfloat16))


# ---- C2. flash-decoding: the same attention, split over the cache so many programs share the work.
# Grid (NKV, NSPLIT). Program (g, sp) takes KV head g and the sp-th chunk of slots 0..pos, for ALL the REP query
# heads that share KV head g: K/V are read once (not REP times), and the REP queries form rows of a 16-row tile so
# the products run on the tensor cores (tl.dot needs >= 16 rows; rows REP..15 are zero padding).
# The chunk length is computed from `pos` on the GPU (rounded up to BT), so short and long sequences both spread
# over the NSPLIT chunks. Each program writes a partial result: running max m, running sum l, and the
# un-normalised output acc; attn_combine merges them.
@triton.jit
def attn_split(q_ptr, kc_ptr, vc_ptr, pos_ptr, po_ptr, pm_ptr, pl_ptr, S, scale,
               REP: tl.constexpr, HD: tl.constexpr, NSPLIT: tl.constexpr, BT: tl.constexpr,
               PV_BF16: tl.constexpr = False):
    g = tl.program_id(0)
    sp = tl.program_id(1)
    pos = tl.load(pos_ptr)
    n = pos + 1                                                 # filled slots
    ch = tl.cdiv(tl.cdiv(n, NSPLIT), BT) * BT                   # chunk length, a multiple of BT
    start = sp * ch
    if start < n:
        end = tl.minimum(start + ch, n)
        r = tl.arange(0, 16)
        d = tl.arange(0, HD)
        t = tl.arange(0, BT)
        qm = r < REP
        q = tl.load(q_ptr + (g * REP + r)[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0)   # [16, HD]
        kb = kc_ptr + g * S * HD
        vb = vc_ptr + g * S * HD
        m_i = tl.zeros([16], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([16], dtype=tl.float32)
        acc = tl.zeros([16, HD], dtype=tl.float32)
        for t0 in range(start, end, BT):
            tt = t0 + t
            msk = tt < end
            k = tl.load(kb + tt[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0)            # [BT, HD]
            s = tl.dot(q, tl.trans(k)) * scale                                                      # [16, BT]
            s = tl.where(msk[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            v = tl.load(vb + tt[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            if PV_BF16:                                   # FlashAttention-style: p rounded to bf16, tensor cores
                pv = tl.dot(p.to(tl.bfloat16), v)
            else:                                         # exact fp32 p @ v, like the reference
                pv = tl.dot(p, v.to(tl.float32), input_precision="ieee")
            acc = acc * alpha[:, None] + pv
            m_i = m_new
        h = g * REP + r                                                                             # head ids
        tl.store(po_ptr + (h * NSPLIT + sp)[:, None] * HD + d[None, :], acc, mask=qm[:, None])
        tl.store(pm_ptr + h * NSPLIT + sp, m_i, mask=qm)
        tl.store(pl_ptr + h * NSPLIT + sp, l_i, mask=qm)


# ---- C3. merge the NSPLIT partials of one head: out = sum_i w_i acc_i / sum_i w_i l_i, with w_i = exp(m_i - max m)
@triton.jit
def attn_combine(po_ptr, pm_ptr, pl_ptr, pos_ptr, o_ptr, HD: tl.constexpr, NSPLIT: tl.constexpr,
                 BT: tl.constexpr):
    h = tl.program_id(0)
    n = tl.load(pos_ptr) + 1
    ch = tl.cdiv(tl.cdiv(n, NSPLIT), BT) * BT
    sp = tl.arange(0, NSPLIT)
    valid = sp * ch < n                                         # chunks that had work
    m = tl.load(pm_ptr + h * NSPLIT + sp, mask=valid, other=float("-inf"))
    l = tl.load(pl_ptr + h * NSPLIT + sp, mask=valid, other=0.0)
    w = tl.where(valid, tl.exp(m - tl.max(m, 0)), 0.0)
    d = tl.arange(0, HD)
    acc = tl.load(po_ptr + (h * NSPLIT + sp)[:, None] * HD + d[None, :], mask=valid[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(o_ptr + h * HD + d, o.to(tl.bfloat16))
