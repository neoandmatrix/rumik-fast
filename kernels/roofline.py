"""How fast can this GPU *read* a Rumik weight matrix with our tiling, if we do almost no maths?
read_only: same loads as the int4 GEMV (q bytes + scales + mins), integer-sums them. Compared with the real kernel
(v4) and cuBLAS bf16 (reads 4x more bytes, so its GB/s is the hardware's practical ceiling for this kind of read).
  uv run python roofline.py
"""
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import G, gemv_int4_magic, quantize  # noqa: E402

L = 36


@triton.jit
def read_only(q_ptr, s_ptr, m_ptr, y_ptr, N, K, GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rb = tl.arange(0, BLOCK_K // 2)
    rg = tl.arange(0, BLOCK_K // GROUP)
    acc = tl.zeros([BLOCK_N], dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        q = tl.load(q_ptr + rn[:, None] * (K // 2) + (k0 // 2 + rb)[None, :]).to(tl.int32)
        g = k0 // GROUP + rg
        s = tl.load(s_ptr + rn[:, None] * (K // GROUP) + g[None, :], tl.int16 if False else None)
        m = tl.load(m_ptr + rn[:, None] * (K // GROUP) + g[None, :])
        acc += tl.sum(q, 1) + tl.sum((s + m).to(tl.int32), 1)
    tl.store(y_ptr + rn, acc)


def graph_us(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    best = 1e9
    for _ in range(3):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / 20 / L * 1e6)
    return best


torch.manual_seed(0)
for n, k in [(25088, 2048), (2048, 11008)]:
    mats = [quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)) for _ in range(L)]
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    yi = torch.empty(n, device="cuda", dtype=torch.int32)
    yb = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    nb = n * k // 2 + 4 * n * (k // G)
    best = []
    for bn in (8, 16, 32, 64):
        for bk in (256, 512, 1024):
            for w in (2, 4, 8):
                t = graph_us(lambda: [read_only[(n // bn,)](q, s, m, yi, n, k, GROUP=G, BLOCK_N=bn, BLOCK_K=bk,
                                                            num_warps=w) for q, s, m in mats])
                best.append((t, bn, bk, w))
    best.sort()
    t = best[0][0]
    tv4 = graph_us(lambda: [gemv_int4_magic(x, *w_, out=yb, BLOCK_N=32, BLOCK_K=512, num_warps=2) for w_ in mats])
    del mats
    torch.cuda.empty_cache()
    ws = [(torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(12)]
    xb = x.view(1, -1)
    tb = graph_us(lambda: [torch.mm(xb, w_.t(), out=yb.view(1, -1)) for w_ in ws * 3])
    print(f"N={n} K={k}: read-only best {nb / t / 1e3:.0f} GB/s {best[0][1:]} (top3 {[round(nb / b[0] / 1e3) for b in best[:3]]}) | "
          f"real int4 kernel {nb / tv4 / 1e3:.0f} GB/s | cuBLAS bf16 {2 * n * k / tb / 1e3:.0f} GB/s", flush=True)
    del ws
    torch.cuda.empty_cache()
