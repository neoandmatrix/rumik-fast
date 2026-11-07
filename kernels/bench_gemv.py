"""Rumik decode-step GEMVs on the local GPU, with random weights (no download needed).
  per shape: our Triton int4 kernel vs cuBLAS bf16 vs PyTorch's built-in int4 kernel (tinygemm), GB/s and us
  full step: the 36 layers x 4 projections in int4 + the bf16 audio head, in one CUDA graph -> a tokens/s ceiling
             (attention, norms, sampling and Mimi not included)

  uv run python bench_gemv.py      -> prints a table, writes results_<gpu>.json
"""
import json
import sys
import time
from pathlib import Path

import torch
import triton.testing as tt

sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import G, dequantize, gemv_int4, gemv_int4_w32, quantize  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
H, INTER, LAYERS = 2048, 11008, 36
SHAPES = {"qkv": (3072, H), "o": (H, H), "gate_up": (2 * INTER, H), "down": (H, INTER), "audio_head": (16385, H),
          "qkv_gate_up": (3072 + 2 * INTER, H)}          # parallel block: attention and MLP read the same LN(x)
name = torch.cuda.get_device_name()
PEAK = {"RTX 3050 Laptop": 176, "L4": 300}
peak = next((v for k, v in PEAK.items() if k in name), None)
print(f"{name}, peak DRAM ~{peak} GB/s", flush=True)


def int4_bytes(n, k):
    return n * k // 2 + 2 * n * (k // G) * 2 + 2 * k + 2 * n


def bench(fn):
    return tt.do_bench(fn, warmup=50, rep=300) * 1e3          # us; do_bench flushes L2 between reps


res = {"gpu": name, "peak_gb_s": peak, "shapes": {}}
for key, (n, k) in SHAPES.items():
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(k, device=dev, dtype=torch.bfloat16)
    q, s, m = quantize(w)
    ref = dequantize(q, s, m) @ x.float()
    y = gemv_int4(x, q, s, m)
    err = ((y.float() - ref).norm() / ref.norm()).item()
    r = {"N": n, "K": k, "rel_err_vs_fp32": round(err, 5)}
    t = bench(lambda: gemv_int4(x, q, s, m))
    r["triton_int4_us"], r["triton_int4_gb_s"] = round(t, 1), round(int4_bytes(n, k) / t / 1e3, 1)
    y2 = gemv_int4_w32(x, q, s, m)
    r["w32_rel_err"] = round(((y2.float() - ref).norm() / ref.norm()).item(), 5)
    t = bench(lambda: gemv_int4_w32(x, q, s, m))
    r["triton_w32_us"], r["triton_w32_gb_s"] = round(t, 1), round(int4_bytes(n, k) / t / 1e3, 1)
    t = bench(lambda: torch.nn.functional.linear(x.view(1, -1), w))
    r["cublas_bf16_us"], r["cublas_bf16_gb_s"] = round(t, 1), round(2 * n * k / t / 1e3, 1)
    try:                                                       # PyTorch's tinygemm int4 kernel (what torchao uses)
        code = torch.stack([q & 0xF, q >> 4], -1).view(n, -1).to(torch.int32)
        packed = torch.ops.aten._convert_weight_to_int4pack((code[:, ::2] << 4 | code[:, 1::2]).to(torch.uint8), 8)
        sz = torch.stack([s.t(), (m.float() + 8 * s.float()).half().t()], -1).to(torch.bfloat16).contiguous()
        xb = x.view(1, -1)
        t = bench(lambda: torch.ops.aten._weight_int4pack_mm(xb, packed, G, sz))
        r["tinygemm_us"], r["tinygemm_gb_s"] = round(t, 1), round(int4_bytes(n, k) / t / 1e3, 1)
    except Exception as e:
        r["tinygemm"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
    res["shapes"][key] = r
    print(key, json.dumps(r), flush=True)
    del w, q, s, m

# ---- full decode step: every projection of every layer (1.49 GB of int4) + the bf16 audio head
torch.cuda.empty_cache()
layers = []
for i in range(LAYERS):
    mats = {}
    for key in ("qkv_gate_up", "o", "down"):
        n, k = SHAPES[key]
        mats[key] = quantize((torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16))
    layers.append(mats)
head = (torch.randn(16385, H, device=dev) * 0.02).to(torch.bfloat16)
xh = torch.randn(H, device=dev, dtype=torch.bfloat16)
xi = torch.randn(INTER, device=dev, dtype=torch.bfloat16)
bufs = {key: torch.empty(SHAPES[key][0], device=dev, dtype=torch.bfloat16) for key in SHAPES}


def make_step(fn):
    def step():
        for L in layers:
            fn(xh, *L["qkv_gate_up"], out=bufs["qkv_gate_up"])
            fn(xh, *L["o"], out=bufs["o"])
            fn(xi, *L["down"], out=bufs["down"])
        torch.mm(xh.view(1, -1), head.t(), out=bufs["audio_head"].view(1, -1))
    return step


def time_graph(step):
    step(); torch.cuda.synchronize()                           # autotune done before capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1e3


step_bytes = sum(int4_bytes(*SHAPES[k]) for k in ("qkv_gate_up", "o", "down")) * LAYERS + 2 * 16385 * H
for label, fn in (("full_step_v1", gemv_int4), ("full_step_w32", gemv_int4_w32)):
    ms = time_graph(make_step(fn))
    res[label] = {"ms": round(ms, 2), "gb": round(step_bytes / 1e9, 3), "gb_s": round(step_bytes / ms / 1e6, 1),
                    "tok_s_ceiling": round(1e3 / ms, 1), "pct_of_peak": round(100 * step_bytes / ms / 1e6 / peak, 1)
                    if peak else None, "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(label, json.dumps(res[label]), flush=True)
tag = "3050" if "3050" in name else name.replace(" ", "_")
(Path(__file__).parent / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
