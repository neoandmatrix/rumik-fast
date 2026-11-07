# int4 GEMV kernels for Rumik decode on the RTX 3050 Laptop (4 GB, 176 GB/s)

These benchmarks use random weights of Rumik's exact shapes, so no download is needed. They run in `04-gemma-stt-hindi/.venv`, which has torch 2.14.1+cu130 and Triton 3.8, so no new installs either.

```
cd 06-rumik-fast/kernels && ../../04-gemma-stt-hindi/.venv/bin/python bench_gemv.py   # ~3 min, writes results_3050.json
```

## Files
- `int4_gemv.py`: quantize/dequantize in the exporter's layout, plus two kernels:
  - `gemv_int4` (v1): byte loads, one fp32 reduction per group.
  - `gemv_int4_w32` (v2): int32 word loads. It's 3× slower, so it's kept only as a negative result.
- `bench_gemv.py`: per-shape GB/s for our kernel, cuBLAS bf16 and PyTorch's tinygemm int4. Then a full decode step with all 36 layers' projections plus the audio head, captured in one CUDA graph.

## Results, 2026-11-06 (GB/s, peak ~176)
| shape | ours v1 | tinygemm int4 | cuBLAS bf16 | v2 (int32 loads) |
|---|---|---|---|---|
| qkv 3072×2048 | 104 | 115 | 149 | 44 |
| o 2048×2048 | 99 | 106 | 140 | 42 |
| gate_up 22016×2048 | 150 | 158 | 168 | 50 |
| down 2048×11008 | 127 | 145 | 157 | 46 |
| qkv+gate_up fused 25088×2048 | 152 | 158 | 168 | 51 |

Full step, 1.56 GB per token (projections + head only; no attention, Mimi or sampling):
- v1: **14.0 ms, 112 GB/s (64% of peak), at most 72 tok/s**;
- v2: 33.7 ms.

Fusing qkv with gate_up didn't change the full step: 13.95 ms fused vs 14.01 ms unfused.

Points to chase next:
- The per-shape times add up to only ~10.8 ms, but the graph takes 14 ms. About 3 ms goes somewhere between or around kernels: tails, clock drop under sustained load, or a different autotune choice. Profile with nsys.
- `o` and `down` (only 2048 rows) are the weakest. Try split-K or more rows per program.
- cuBLAS bf16 reaches 85–95% of peak, so the hardware can do it. Our int4 kernels lose on dequant work and smaller per-thread loads.

## Where the lost time goes (`prof_step.py`, nsys, 2026-11-06)
- There are **no gaps** between kernels in the graph: the last step's span equals its busy time, 11.6 ms.
- The lost time is **the laptop GPU slowing under sustained load**:
  - 1 step: 10.1 ms; 100 steps in a row: 12.7 ms per step.
  - SM clock fell from 1560 to 1282 MHz. nvidia-smi showed throttle reason 0x4, the software power cap, at about 35 W. Memory clock stayed at 5501 MHz.
- So v1 is partly **ALU-bound**: the unpack and convert work per weight makes it sensitive to SM clock. The fix is fewer instructions per weight, not better memory access.
- v3 (`gemv_int4_tc`, the multiply-adds on tensor cores, one 128-column group per iteration) was **slower**: 122/70/86 GB/s per shape and 17.3 ms sustained, against 13.2 ms for v1. The per-group loop makes the loads too small.
- Laptop numbers are noisy with temperature. The same v1 step measured 10.1 ms cold and 14.9 ms after a few minutes of benchmarks. Always compare variants back to back, with a pause between them.
- Next: a CUDA kernel using the standard int4→fp16 bit trick (OR the code into the mantissa of 1024.0 as fp16, then subtract 1024; two weights per 32-bit operation) with 128-bit loads. That's the approach of Marlin and FasterTransformer, and it costs a few instructions per weight instead of about 6.

## Power / clocks on this laptop (ASUS Vivobook Pro M6500QC), 2026-11-06
60 s GEMV-only step (36 layers, v1 kernel, CUDA graph):
- Balanced mode: the first ~15 s run at a 30 W cap with the memory at 5001 MHz (15.5 ms/step). Then Dynamic Boost
  raises the cap to 40 W and the memory to 5501 MHz (12.1 ms/step).
- Performance mode: the memory is at 5501 MHz from the start, the cap is 40 W, and the step is flat at
  11.7–12.0 ms for 60 s (125 GB/s, 71% of peak). The core clock drops from 1665 to ~1340 MHz at the cap, but the
  step time barely moves.
- Conclusion: the memory clock matters and the core clock hardly does. More watts would not help; better kernels
  will. Benchmark in performance mode only.

## Why Triton stopped at ~112–140 GB/s, and the CUDA kernel (2026-11-07)
- `roofline.py`: a read-only kernel with our exact loads reaches 153 GB/s = cuBLAS bf16 (154). So the loads were fine.
- `sass_count.py`: the v1 inner loop is 5.97 instructions per weight (256 I2F, 298 PRMT, 194 FADD, 260 FMUL/FFMA
  per 256 weights per thread); v4 with the fp32 magic-number trick is 6.86 (no I2F, but more LOP3/FADD/SGXT).
- v5 (fp16 maths in Triton): slower, 6.4–6.9 instructions per weight; the FOLD variant has 19–22% error.
- `cuda_gemv.py` (load_inline, sm_86): ~1.6 instructions per weight; 167 / 138 / 155 GB/s on in / o / down.
  `bench_cuda.py` checks the error vs fp32 and times both kernels in a CUDA graph.
- Lesson: when a kernel is compute-bound on unpacking, count the instructions first; Triton gives little control
  over packed fp16 maths, while hand-written CUDA does.
