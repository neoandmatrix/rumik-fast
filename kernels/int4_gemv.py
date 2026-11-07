"""Batch-1 int4 matrix-vector product (decode step) in Triton, for the layout written by ../modal_export_int4.py:
  qweight uint8 [N, K/2]  (low nibble = even column), scale fp16 [N, K/G], min fp16 [N, K/G];  w = code * scale + min
  y[n] = sum_g ( scale[n,g] * sum_{k in g} x[k] * code[n,k]  +  min[n,g] * sum_{k in g} x[k] )
Decode is bandwidth-bound: the goal is to stream the 0.53 bytes/weight at as close to the GPU's peak as possible.
"""
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

G = 128


def quantize(w: torch.Tensor, group: int = G):
    """bf16/fp32 [N, K] -> (qweight, scale, min), same maths as the exporter"""
    n, k = w.shape
    g = w.float().view(n, k // group, group)
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp(min=1e-8)
    code = ((g - mn) / scale).round().clamp(0, 15).to(torch.uint8).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale.half().view(n, -1).contiguous(), mn.half().view(n, -1).contiguous()


def dequantize(q, s, m, group: int = G):
    n = q.shape[0]
    code = torch.stack([q & 0xF, q >> 4], -1).view(n, -1).float()
    return (code.view(n, -1, group) * s.float()[..., None] + m.float()[..., None]).view(n, -1)


def _configs():
    out = []
    for bn in (8, 16, 32):
        for bk in (128, 256, 512):
            for w in (2, 4):
                for st in (2, 3):
                    out.append(triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st))
    return out


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _gemv_int4(x_ptr, q_ptr, s_ptr, m_ptr, y_ptr, N, K,
               GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    NG: tl.constexpr = BLOCK_K // GROUP
    HALF: tl.constexpr = GROUP // 2
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    rb = tl.arange(0, BLOCK_K // 2)                      # byte index inside the tile
    rg = tl.arange(0, NG)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kb = k0 // 2 + rb
        kmask = kb < K // 2
        q = tl.load(q_ptr + rn[:, None] * (K // 2) + kb[None, :], mask=nmask[:, None] & kmask[None, :], other=0)
        xe = tl.load(x_ptr + 2 * kb, mask=kmask, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + 2 * kb + 1, mask=kmask, other=0.0).to(tl.float32)
        lo = (q & 0xF).to(tl.float32)
        hi = (q >> 4).to(tl.float32)
        p = lo * xe[None, :] + hi * xo[None, :]                                   # [BN, BK/2]
        dot = tl.sum(tl.reshape(p, [BLOCK_N, NG, HALF]), axis=2)                  # [BN, NG]
        xs = tl.sum(tl.reshape(xe + xo, [NG, HALF]), axis=1)                      # [NG]
        g = k0 // GROUP + rg
        gmask = g < K // GROUP
        smask = nmask[:, None] & gmask[None, :]
        s = tl.load(s_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        acc += tl.sum(s * dot + m * xs[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)


# configs pinned by tune_gemv.py (per shape, for this GPU); shapes not in the table fall back to autotune
_CFG_FILE = Path(__file__).parent / "gemv_configs.json"
_PINNED = {}
if _CFG_FILE.exists():
    _j = json.loads(_CFG_FILE.read_text())
    if torch.cuda.is_available() and _j.get("gpu") == torch.cuda.get_device_name():
        _PINNED = {tuple(map(int, k.split("x"))): v for k, v in _j.items() if "x" in k and k[0].isdigit()}


def gemv_int4(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, m: torch.Tensor, out: torch.Tensor | None = None):
    """x [K] (bf16/fp16) -> y [N] in x's dtype"""
    n, k = q.shape[0], q.shape[1] * 2
    assert x.numel() == k and k % G == 0
    y = out if out is not None else torch.empty(n, device=x.device, dtype=x.dtype)
    c = _PINNED.get((n, k))
    if c:
        _gemv_int4.fn[(triton.cdiv(n, c["BLOCK_N"]),)](x, q, s, m, y, n, k, GROUP=G, BLOCK_N=c["BLOCK_N"],
                                                       BLOCK_K=c["BLOCK_K"], num_warps=c["num_warps"],
                                                       num_stages=c["num_stages"])
    else:
        _gemv_int4[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](x, q, s, m, y, n, k, GROUP=G)
    return y


# ---- v2: the same bytes read as int32 words (8 codes each: nibble i of the little-endian word = column i)
@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _gemv_int4_w32(x_ptr, q_ptr, s_ptr, m_ptr, y_ptr, N, K,
                   GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    NG: tl.constexpr = BLOCK_K // GROUP
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    rw = tl.arange(0, BLOCK_K // 8)                      # word index inside the tile
    rk = tl.arange(0, BLOCK_K)
    rg = tl.arange(0, NG)
    shifts = tl.arange(0, 8) * 4
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kw = k0 // 8 + rw
        wmask = kw < K // 8
        w = tl.load(q_ptr + rn[:, None] * (K // 8) + kw[None, :], mask=nmask[:, None] & wmask[None, :], other=0)
        code = ((w[:, :, None] >> shifts[None, None, :]) & 0xF).to(tl.float32)   # [BN, BK/8, 8]
        code = tl.reshape(code, [BLOCK_N, BLOCK_K])
        x = tl.load(x_ptr + k0 + rk, mask=k0 + rk < K, other=0.0).to(tl.float32)
        dot = tl.sum(tl.reshape(code * x[None, :], [BLOCK_N, NG, GROUP]), axis=2)  # [BN, NG]
        xs = tl.sum(tl.reshape(x, [NG, GROUP]), axis=1)
        g = k0 // GROUP + rg
        smask = nmask[:, None] & (g < K // GROUP)[None, :]
        s = tl.load(s_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        acc += tl.sum(s * dot + m * xs[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)


def gemv_int4_w32(x, q, s, m, out=None):
    """same as gemv_int4; q may be the uint8 [N, K/2] tensor (viewed as int32 [N, K/8])"""
    n, k = q.shape[0], q.shape[1] * 2
    qw = q.view(torch.int32)
    y = out if out is not None else torch.empty(n, device=x.device, dtype=x.dtype)
    _gemv_int4_w32[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](x, qw, s, m, y, n, k, GROUP=G)
    return y


# ---- v3: the multiply-adds on the tensor cores. One group (128 columns) per iteration: the codes go in as exact
# fp16 integers 0..15 and x as row 0 of a 16-row tile (tensor cores need M >= 16; 15 rows are wasted, but they
# are idle anyway), so the only per-weight work left on the ALUs is the unpack. Scale and min are applied per group.
def _configs_tc():
    return [triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
            for bn in (32, 64, 128) for w in (2, 4, 8) for st in (2, 3, 4)]


@triton.autotune(configs=_configs_tc(), key=["N", "K"])
@triton.jit
def _gemv_int4_tc(x_ptr, q_ptr, s_ptr, m_ptr, y_ptr, N, K, GROUP: tl.constexpr, BLOCK_N: tl.constexpr):
    HALF: tl.constexpr = GROUP // 2
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    rb = tl.arange(0, HALF)
    rk = tl.arange(0, GROUP)
    row = tl.arange(0, 16)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    NG = K // GROUP
    for gi in range(0, NG):
        q = tl.load(q_ptr + rn[:, None] * (K // 2) + gi * HALF + rb[None, :], mask=nmask[:, None], other=0)
        code = tl.reshape(tl.join(q & 0xF, q >> 4), [BLOCK_N, GROUP]).to(tl.float16)     # [BN, G]
        x = tl.load(x_ptr + gi * GROUP + rk).to(tl.float16)
        xt = tl.where(row[:, None] == 0, x[None, :], 0.0).to(tl.float16)                # [16, G]
        d = tl.dot(xt, tl.trans(code), out_dtype=tl.float32)                            # [16, BN]
        d0 = tl.sum(tl.where(row[:, None] == 0, d, 0.0), axis=0)                        # [BN]
        xs = tl.sum(x.to(tl.float32), axis=0)
        s = tl.load(s_ptr + rn * NG + gi, mask=nmask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + rn * NG + gi, mask=nmask, other=0.0).to(tl.float32)
        acc += s * d0 + m * xs
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)


def gemv_int4_tc(x, q, s, m, out=None):
    """same contract as gemv_int4; x is rounded to fp16 for the tensor-core product"""
    n, k = q.shape[0], q.shape[1] * 2
    y = out if out is not None else torch.empty(n, device=x.device, dtype=x.dtype)
    _gemv_int4_tc[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](x, q, s, m, y, n, k, GROUP=G)
    return y


# ---- v4: v1 with the "magic number" int4 -> float conversion, and optional split-K.
# A float32 whose bits are 0x4B000000 is 2^23; OR-ing a small integer c into its low mantissa bits gives exactly
# 2^23 + c. So code = bitcast_f32((byte & 0xF) | 0x4B000000) - 2^23 needs one integer op (AND+OR fuse into one LOP3)
# and one float subtract, both full rate, instead of an I2F conversion (1/8 rate on Ampere: with 2 weights per byte,
# I2F alone could barely keep up with the memory).
# SPLIT_K > 1: program (n, s) takes the K tiles s, s+SPLIT_K, ... and atomically adds its partial sums into y
# (fp32, must start at zero). For down (K=11008, only N=2048 rows) this gives more programs shorter rows.
@triton.jit
def _gemv_int4_magic(x_ptr, q_ptr, s_ptr, m_ptr, y_ptr, N, K, GROUP: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    NG: tl.constexpr = BLOCK_K // GROUP
    HALF: tl.constexpr = GROUP // 2
    pid = tl.program_id(0)
    pk = tl.program_id(1)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    rb = tl.arange(0, BLOCK_K // 2)
    rg = tl.arange(0, NG)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(pk * BLOCK_K, K, BLOCK_K * SPLIT_K):
        kb = k0 // 2 + rb
        kmask = kb < K // 2
        q = tl.load(q_ptr + rn[:, None] * (K // 2) + kb[None, :], mask=nmask[:, None] & kmask[None, :],
                    other=0).to(tl.int32)
        xe = tl.load(x_ptr + 2 * kb, mask=kmask, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + 2 * kb + 1, mask=kmask, other=0.0).to(tl.float32)
        lo = tl.cast((q & 0xF) | 0x4B000000, tl.float32, bitcast=True) - 8388608.0
        hi = tl.cast((q >> 4) | 0x4B000000, tl.float32, bitcast=True) - 8388608.0
        p = lo * xe[None, :] + hi * xo[None, :]
        dot = tl.sum(tl.reshape(p, [BLOCK_N, NG, HALF]), axis=2)
        xs = tl.sum(tl.reshape(xe + xo, [NG, HALF]), axis=1)
        g = k0 // GROUP + rg
        smask = nmask[:, None] & (g < K // GROUP)[None, :]
        s = tl.load(s_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        acc += tl.sum(s * dot + m * xs[None, :], axis=1)
    if SPLIT_K == 1:
        tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)
    else:
        tl.atomic_add(y_ptr + rn, acc, mask=nmask)


def gemv_int4_magic(x, q, s, m, out, BLOCK_N=16, BLOCK_K=512, num_warps=2, num_stages=2, SPLIT_K=1):
    """same contract as gemv_int4; with SPLIT_K > 1, `out` must be fp32 and zero (results are added into it)"""
    n, k = q.shape[0], q.shape[1] * 2
    _gemv_int4_magic[(triton.cdiv(n, BLOCK_N), SPLIT_K)](x, q, s, m, out, n, k, GROUP=G, BLOCK_N=BLOCK_N,
                                                         BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K, num_warps=num_warps,
                                                         num_stages=num_stages)
    return out


# ---- v5: fp16 maths inside each group. Codes become fp16 with the fp16 magic number (0x6400 = 1024.0: OR-ing a
# 4-bit code into the low mantissa bits gives 1024 + code exactly), x is fp16, the products and the 64-pair sum per
# group are fp16 (Ampere runs fp16 ops two per instruction), and each group's sum goes into an fp32 accumulator.
# The 1024 offset is not subtracted per weight: sum_k x_k (1024 + c_k) = sum x c + 1024 sum x, so it is folded into
# the per-group term: y += s * (dot - 1024 * xs) + m * xs. (dot is kept in fp32 for that subtraction.)
@triton.jit
def _gemv_int4_h(x_ptr, q_ptr, s_ptr, m_ptr, y_ptr, N, K, GROUP: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, FOLD: tl.constexpr):
    NG: tl.constexpr = BLOCK_K // GROUP
    HALF: tl.constexpr = GROUP // 2
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    rb = tl.arange(0, BLOCK_K // 2)
    rg = tl.arange(0, NG)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kb = k0 // 2 + rb
        kmask = kb < K // 2
        q = tl.load(q_ptr + rn[:, None] * (K // 2) + kb[None, :], mask=nmask[:, None] & kmask[None, :],
                    other=0).to(tl.int16)
        xe = tl.load(x_ptr + 2 * kb, mask=kmask, other=0.0).to(tl.float16)
        xo = tl.load(x_ptr + 2 * kb + 1, mask=kmask, other=0.0).to(tl.float16)
        lo = tl.cast((q & 0xF) | 0x6400, tl.float16, bitcast=True)
        hi = tl.cast((q >> 4) | 0x6400, tl.float16, bitcast=True)
        if not FOLD:
            lo = lo - 1024.0
            hi = hi - 1024.0
        p = lo * xe[None, :] + hi * xo[None, :]                                          # fp16
        dot = tl.sum(tl.reshape(p, [BLOCK_N, NG, HALF]), axis=2, dtype=tl.float16).to(tl.float32)
        xs = tl.sum(tl.reshape(xe.to(tl.float32) + xo.to(tl.float32), [NG, HALF]), axis=1)
        g = k0 // GROUP + rg
        smask = nmask[:, None] & (g < K // GROUP)[None, :]
        s = tl.load(s_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + rn[:, None] * (K // GROUP) + g[None, :], mask=smask, other=0.0).to(tl.float32)
        if FOLD:
            dot = dot - 1024.0 * xs[None, :]
        acc += tl.sum(s * dot + m * xs[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)


def gemv_int4_h(x, q, s, m, out, BLOCK_N=32, BLOCK_K=512, num_warps=2, num_stages=2, FOLD=False):
    n, k = q.shape[0], q.shape[1] * 2
    _gemv_int4_h[(triton.cdiv(n, BLOCK_N),)](x, q, s, m, out, n, k, GROUP=G, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                             FOLD=FOLD, num_warps=num_warps, num_stages=num_stages)
    return out
