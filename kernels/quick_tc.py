"""v1 vs v3 (tensor cores): correctness, per shape (cold, do_bench) and a sustained 100-step graph."""
import sys, time
from pathlib import Path
import torch, triton.testing as tt
sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import dequantize, gemv_int4, gemv_int4_tc, quantize
H, I, L = 2048, 11008, 36
S = {"qkv_gate_up": (3072 + 2 * I, H), "o": (H, H), "down": (H, I)}
torch.manual_seed(0)
for k, (n, kk) in S.items():
    q, s, m = quantize((torch.randn(n, kk, device="cuda") * 0.02).to(torch.bfloat16))
    x = torch.randn(kk, device="cuda", dtype=torch.bfloat16)
    ref = dequantize(q, s, m) @ x.float()
    by = n * kk / 2 + 4 * n * kk / 128
    for name, fn in (("v1", gemv_int4), ("v3_tc", gemv_int4_tc)):
        e = ((fn(x, q, s, m).float() - ref).norm() / ref.norm()).item()
        t = tt.do_bench(lambda: fn(x, q, s, m), warmup=50, rep=300) * 1e3
        print(f"{k:12s} {name:6s} err {e:.4f}  {t:6.1f} us  {by / t / 1e3:6.1f} GB/s", flush=True)
layers = [{k: quantize((torch.randn(*s, device="cuda") * 0.02).to(torch.bfloat16)) for k, s in S.items()} for _ in range(L)]
xh = torch.randn(H, device="cuda", dtype=torch.bfloat16); xi = torch.randn(I, device="cuda", dtype=torch.bfloat16)
out = {k: torch.empty(s[0], device="cuda", dtype=torch.bfloat16) for k, s in S.items()}
gb = sum(n * k / 2 + 4 * n * k / 128 for n, k in S.values()) * L / 1e9
for name, fn in (("v1", gemv_int4), ("v3_tc", gemv_int4_tc)):
    def step():
        for Ly in layers:
            fn(xh, *Ly["qkv_gate_up"], out=out["qkv_gate_up"]); fn(xh, *Ly["o"], out=out["o"]); fn(xi, *Ly["down"], out=out["down"])
    step(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    time.sleep(3)                                   # let the GPU cool a bit between the two
    for rep in (1, 100):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(rep):
            g.replay()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) / rep * 1e3
        print(f"step {name:6s} x{rep:3d}: {ms:6.2f} ms  {gb / ms * 1e3:6.1f} GB/s", flush=True)
