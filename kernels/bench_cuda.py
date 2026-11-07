"""CUDA int4 GEMV vs Triton v1: error vs fp32 and GB/s per Rumik shape (36 matrices in one CUDA graph).
  ../../04-gemma-stt-hindi/.venv/bin/python bench_cuda.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cuda_gemv import gemv_int4_cuda  # noqa: E402
from int4_gemv import G, dequantize, gemv_int4, quantize  # noqa: E402

L = 36
torch.manual_seed(0)


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


for n, k in [(25088, 2048), (2048, 2048), (2048, 11008)]:
    mats = [quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)) for _ in range(L)]
    x = (torch.randn(k, device="cuda") * (5 if k == 11008 else 1)).to(torch.bfloat16)
    ref = dequantize(*mats[0]) @ x.float()
    y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    nb = n * k // 2 + 4 * n * (k // G)
    e1 = ((gemv_int4(x, *mats[0]).float() - ref).norm() / ref.norm()).item()
    e2 = ((gemv_int4_cuda(x, *mats[0]).float() - ref).norm() / ref.norm()).item()
    t1 = graph_us(lambda: [gemv_int4(x, *w, out=y) for w in mats])
    t2 = graph_us(lambda: [gemv_int4_cuda(x, *w, out=y) for w in mats])
    print(f"N={n:5d} K={k:5d}: Triton v1 {t1:6.1f} us {nb / t1 / 1e3:4.0f} GB/s err {e1:.4f} | "
          f"CUDA {t2:6.1f} us {nb / t2 / 1e3:4.0f} GB/s err {e2:.4f}", flush=True)
    del mats
    torch.cuda.empty_cache()

# ---- block-shape sweep for the CUDA kernel (warps per block x rows per warp), per Rumik shape
if "--shapes" in sys.argv:
    for n, k in [(25088, 2048), (2048, 2048), (2048, 11008)]:
        mats = [quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)) for _ in range(L)]
        x = torch.randn(k, device="cuda").to(torch.bfloat16)
        y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        nb = n * k // 2 + 4 * n * (k // G)
        res = []
        for w, r in [(8, 2), (8, 1), (8, 4), (4, 1), (4, 2), (4, 4), (16, 1), (16, 2)]:
            t = graph_us(lambda: [gemv_int4_cuda(x, *wt, out=y, warps=w, rows=r) for wt in mats])
            res.append((t, w, r))
        res.sort()
        print(f"N={n:5d} K={k:5d}: " + "  ".join(f"{w}x{r}: {nb / t / 1e3:.0f} GB/s" for t, w, r in res), flush=True)
        del mats
        torch.cuda.empty_cache()
