"""Pick one fixed config per Rumik GEMV shape, carefully, and save it (gemv_configs.json).
Why: @triton.autotune re-times every config in each new process with a quick benchmark; on a laptop GPU whose clocks
move around, it sometimes locks in a worse config, so the same engine ran at 13.0 or 14.6 ms/step from run to run.
Here each config is timed the way the engine uses it: 36 different weight matrices (no cache reuse) in a CUDA graph,
over two full sweeps, keeping the best time per config across sweeps.
  ../../04-gemma-stt-hindi/.venv/bin/python tune_gemv.py        (~2 min, run in performance mode)
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import G, _configs, _gemv_int4, quantize  # noqa: E402

H, INTER, LAYERS = 2048, 11008, 36
SHAPES = [(3072 + 2 * INTER, H), (H, H), (H, INTER)]       # fused q|k|v|gate|up, o, down
kern = _gemv_int4.fn                                          # the plain jitted kernel under the autotuner
torch.manual_seed(0)


def timed(n, k, cfg, mats, x, y):
    def launch():
        for q, s, m in mats:
            kern[(triton_cdiv(n, cfg.kwargs["BLOCK_N"]),)](x, q, s, m, y, n, k, GROUP=G, **cfg.kwargs,
                                                         num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    launch(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 20 / len(mats) * 1e6          # us per GEMV


def triton_cdiv(a, b):
    return (a + b - 1) // b


out = {}
for n, k in SHAPES:
    mats = [quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)) for _ in range(LAYERS)]
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    best = {}
    for sweep in range(2):
        for cfg in _configs():
            key = (cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"], cfg.num_warps, cfg.num_stages)
            us = timed(n, k, cfg, mats, x, y)
            best[key] = min(best.get(key, 1e9), us)
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    (bn, bk, w, st), us = ranked[0]
    gbs = (n * k // 2 + 4 * n * (k // G)) / us / 1e3
    print(f"N={n} K={k}: best BLOCK_N={bn} BLOCK_K={bk} warps={w} stages={st}  {us:.1f} us ({gbs:.0f} GB/s); "
          f"worst {ranked[-1][1]:.1f} us; top 3 {[(c, round(t, 1)) for c, t in ranked[:3]]}", flush=True)
    out[f"{n}x{k}"] = {"BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": st, "us": round(us, 1)}
    del mats
    torch.cuda.empty_cache()
(Path(__file__).parent / "gemv_configs.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(), **out},
                                                                    indent=1))
