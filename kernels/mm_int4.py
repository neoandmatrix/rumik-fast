"""int4-weight matrix-matrix product for the prefill (many tokens at once): Y[T, N] = X[T, K] @ W^T, W in the int4
layout of ../modal_export_int4.py (qweight uint8 [N, K/2], fp16 scale/min per 128-column group; w = code*s + min).

Before this, the prefill dequantized every weight matrix to a temporary bf16 copy (with fp32 temporaries) and then
ran a bf16 matmul: ~640 ms per prompt, almost all of it creating and discarding those copies.
Here the weights never leave int4 in memory. Each program computes a [BM tokens x BN outputs] tile and walks K one
group (128 columns) at a time:
    codes  = unpack the [BN, 64] bytes -> [BN, 128] integers 0..15 as bf16 (exact)
    d      = X_tile [BM, 128] @ codes^T          (tensor cores, fp32 accumulate)
    acc   += d * scale[n] + rowsum(X_tile)[m] * min[n]       (the group's affine, applied to the result)
so the per-weight work is just the unpack, and the multiply-adds run on the tensor cores.

  uv run python mm_int4.py      -> correctness + timing vs the old dequant path
"""
import torch
import triton
import triton.language as tl

G = 128


# do_not_specialize: Triton otherwise compiles a separate version when an int argument is a multiple of 16, so a
# 112-token prompt (vs 111) triggered a ~1.3 s compile in the middle of a request.
@triton.jit(do_not_specialize=["M"])
def _mm_int4(x_ptr, q_ptr, s_ptr, m_ptr, y_ptr, M, N, K, sxm, sym,
             GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    rb = tl.arange(0, GROUP // 2)
    rk = tl.arange(0, GROUP)
    NG = K // GROUP
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in range(0, NG):
        q = tl.load(q_ptr + rn[:, None] * (K // 2) + g * (GROUP // 2) + rb[None, :], mask=nmask[:, None], other=0)
        codes = tl.reshape(tl.join(q & 0xF, q >> 4), [BN, GROUP]).to(tl.bfloat16)   # col 2j = low nibble
        x = tl.load(x_ptr + rm[:, None] * sxm + g * GROUP + rk[None, :], mask=mmask[:, None], other=0.0)
        d = tl.dot(x, tl.trans(codes))                                                # [BM, BN] fp32
        xs = tl.sum(x.to(tl.float32), axis=1)
        s = tl.load(s_ptr + rn * NG + g, mask=nmask, other=0.0).to(tl.float32)
        mn = tl.load(m_ptr + rn * NG + g, mask=nmask, other=0.0).to(tl.float32)
        acc += d * s[None, :] + xs[:, None] * mn[None, :]
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :], acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


def warmup(shapes):
    """compile every config mm_int4 can pick for these (N, K) shapes, so no request waits for a compile"""
    for n, k in shapes:
        q = torch.zeros(n, k // 2, device="cuda", dtype=torch.uint8)
        s = torch.zeros(n, k // G, device="cuda", dtype=torch.float16)
        for T in (17, 33, 49, 101):            # one prompt length per config bucket
            mm_int4(torch.zeros(T, k, device="cuda", dtype=torch.bfloat16), q, s, s)
    torch.cuda.synchronize()


def mm_int4(x, q, s, m, BM=None, BN=64, num_warps=4, num_stages=3):
    """x bf16 [T, K] -> y bf16 [T, N]"""
    x = x.contiguous()
    T, K = x.shape
    N = q.shape[0]
    assert q.shape[1] * 2 == K and K % G == 0
    if BM is None:          # best configs from the sweep below (3050): small BM wins except for the long-K down matrix
        if K > 4096:
            BM, BN, num_warps = (64, 64, 8) if T <= 48 else (32, 64, 4) if T <= 100 else (64, 128, 4)
        else:
            BM, BN = 16, 128
    y = torch.empty(T, N, device=x.device, dtype=torch.bfloat16)
    _mm_int4[(triton.cdiv(T, BM), triton.cdiv(N, BN))](x, q, s, m, y, T, N, K, x.stride(0), y.stride(0), GROUP=G,
                                                       BM=BM, BN=BN, num_warps=num_warps, num_stages=num_stages)
    return y


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from int4_gemv import dequantize, quantize

    torch.manual_seed(0)

    def ms(fn, reps=10):
        fn(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3

    for n, k in [(25088, 2048), (2048, 2048), (2048, 11008)]:
        q, s, m = quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16))
        for T in (34, 69, 201):
            x = torch.randn(T, k, device="cuda").to(torch.bfloat16)
            ref = x.float() @ dequantize(q, s, m).t()
            y = mm_int4(x, q, s, m)
            err = ((y.float() - ref).norm() / ref.norm()).item()
            t_new = ms(lambda: mm_int4(x, q, s, m))
            w = dequantize(q, s, m).to(torch.bfloat16)
            t_old = ms(lambda: x @ dequantize(q, s, m).to(torch.bfloat16).t())
            best = min((ms(lambda: mm_int4(x, q, s, m, BM=bm, BN=bn, num_warps=wp)), bm, bn, wp)
                       for bm in (16, 32, 64, 128) for bn in (32, 64, 128) for wp in (4, 8) if bm * bn <= 8192)
            print(f"N={n:5d} K={k:5d} T={T:3d}: err {err:.4f} | old dequant+matmul {t_old:6.2f} ms | "
                  f"int4 {t_new:5.2f} ms (best {best[0]:.2f} ms at BM={best[1]} BN={best[2]} warps={best[3]})",
                  flush=True)
