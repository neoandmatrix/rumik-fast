"""v1 (pinned config) vs v4 (magic conversion, + split-K) per Rumik shape: 36 different matrices in one CUDA graph,
best of 2 rounds per config. Checks v4's result against v1 first.
  uv run python bench_v4.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import G, _PINNED, gemv_int4, gemv_int4_magic, quantize  # noqa: E402

H, INTER, L = 2048, 11008, 36
torch.manual_seed(0)


def graph_us(fn, zero=None):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    best = 1e9
    for _ in range(2):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / 20 / L * 1e6)
    return best


for n, k in [(25088, H), (H, H), (H, INTER)]:
    mats = [quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)) for _ in range(L)]
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    yb = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    yf = torch.zeros(L, n, device="cuda")
    ref = gemv_int4(x, *mats[0]).float()
    nbytes = n * k // 2 + 4 * n * (k // G)
    t1 = graph_us(lambda: [gemv_int4(x, *w, out=yb) for w in mats])
    print(f"N={n} K={k}: v1 pinned {t1:6.1f} us ({nbytes / t1 / 1e3:.0f} GB/s)  {_PINNED.get((n, k))}", flush=True)
    res = []
    for sk in ((1, 2, 4) if n == H else (1,)):
        for bn in (8, 16, 32):
            for bk in (256, 512):
                for w in (2, 4):
                    cfg = dict(BLOCK_N=bn, BLOCK_K=bk, num_warps=w, SPLIT_K=sk)
                    if sk == 1:
                        fn = lambda: [gemv_int4_magic(x, *wt, out=yb, **cfg) for wt in mats]
                    else:        # zeroing included (in the engine, the consumer will clear it for free)
                        fn = lambda: [(yf[i].zero_(), gemv_int4_magic(x, *wt, out=yf[i], **cfg))
                                      for i, wt in enumerate(mats)]
                    yf.zero_()
                    (gemv_int4_magic(x, *mats[0], out=yb, **cfg) if sk == 1 else
                     gemv_int4_magic(x, *mats[0], out=yf[0], **cfg))
                    got = (yb if sk == 1 else yf[0]).float()
                    err = ((got - ref).norm() / ref.norm()).item()
                    res.append((graph_us(fn), cfg, err))
    res.sort(key=lambda r: r[0])
    for t, cfg, err in res[:3]:
        print(f"   v4 {t:6.1f} us ({nbytes / t / 1e3:.0f} GB/s)  err vs v1 {err:.4f}  {cfg}", flush=True)
    del mats
    torch.cuda.empty_cache()
