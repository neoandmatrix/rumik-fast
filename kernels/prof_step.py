"""Where do the ~3 ms between "sum of per-shape kernel times" and "full step" go? One decode step's projections
(random int4 weights, Rumik shapes) as a CUDA graph, run under nsys with per-node tracing:

  nsys profile --cuda-graph-trace=node -o step -f true uv run python prof_step.py
  nsys stats -r cuda_gpu_kern_sum step.nsys-rep
Also prints the step time with the GPU clocks reported before/after (laptop GPUs drop clocks under load).
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import gemv_int4, quantize  # noqa: E402

H, INTER, LAYERS = 2048, 11008, 36
SHAPES = {"qkv_gate_up": (3072 + 2 * INTER, H), "o": (H, H), "down": (H, INTER)}
torch.manual_seed(0)
layers = [{k: quantize((torch.randn(*s, device="cuda") * 0.02).to(torch.bfloat16)) for k, s in SHAPES.items()}
          for _ in range(LAYERS)]
xh = torch.randn(H, device="cuda", dtype=torch.bfloat16)
xi = torch.randn(INTER, device="cuda", dtype=torch.bfloat16)
out = {k: torch.empty(s[0], device="cuda", dtype=torch.bfloat16) for k, s in SHAPES.items()}


def step():
    for L in layers:
        gemv_int4(xh, *L["qkv_gate_up"], out=out["qkv_gate_up"])
        gemv_int4(xh, *L["o"], out=out["o"])
        gemv_int4(xi, *L["down"], out=out["down"])


def clocks():
    q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()


step(); torch.cuda.synchronize()
for k, (q, s, m) in layers[0].items():                # which autotune config won, per shape
    print(k, "N,K =", (q.shape[0], q.shape[1] * 2))
from int4_gemv import _gemv_int4  # noqa: E402
for key, cfg in _gemv_int4.cache.items():
    print("autotune", key, cfg)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    step()
print("before:", clocks())
for rep in (1, 10, 100):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        g.replay()
    torch.cuda.synchronize()
    print(f"{rep:4d} steps: {(time.perf_counter() - t) / rep * 1e3:.2f} ms/step", flush=True)
print("after: ", clocks())
