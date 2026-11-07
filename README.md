# Rumik-oss-1, real-time on a 4 GB laptop GPU

A from-scratch inference engine for [rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1) (a 3B Hindi / English /
Indic text-to-speech model) that streams speech at **~98 audio tokens/s on an RTX 3050 Laptop GPU (4 GB, 35–40 W)**.
Real time is 100 tok/s. It uses int4 weights, fused Triton kernels, flash-decoding, hand-written CUDA int4/int8 GEMVs,
an int4 prefill matmul, exact streaming Mimi decoding, and a small local web UI.

| RTX 3050 Laptop, AC power | first engine (step 4) | now |
|---|---|---|
| decode speed | 59 tok/s | **95–98 tok/s** (live, with streaming Mimi: ~97) |
| prompt (prefill), 34 tokens | 642 ms | **50 ms** |
| first audio frame ready | ~770 ms | **~130–190 ms** |
| first sound from the speaker (5–8 s lines, no stalls) | — | **~0.45–0.7 s** |
| VRAM | 2.5 GB | 2.5 GB |

int4 weights on Hugging Face: [bringagi/rumik-oss-1-int4](https://huggingface.co/bringagi/rumik-oss-1-int4).

## Quick start
Needs an NVIDIA GPU with compute capability 8.0+ (RTX 30xx/40xx/50xx, A10, L4, A100, H100…), ~3 GB of VRAM,
the CUDA toolkit (`nvcc`) for the CUDA kernels, and Linux. Everything was tested only on an RTX 3050 Laptop.

```bash
pip install torch            # the build for your CUDA, see pytorch.org
pip install -r requirements.txt
python download_weights.py   # ~2.55 GB into ./weights (int4 file + Rumik's tokenizer/config/codec)

cd engine
python say.py                # 4 test lines -> out/*.wav, prints tok/s and timings
python say.py --hard         # 4 long lines (numbers, code-switching, emotion tags)
python speak.py              # live streaming to your speakers; type a line, hear it
cd ../ui && python server.py # web UI at http://127.0.0.1:8765
```

The first run compiles the CUDA kernels (about 1 minute; cached in `~/.cache/torch_extensions`) and the Triton kernels.

**On a different GPU.** It runs without changes; the CUDA code is compiled for your GPU's architecture. Some
settings were tuned on the 3050 and are only applied there: `kernels/gemv_configs.json` (the Triton fallback GEMV)
is keyed by GPU name, and the CUDA kernel's block shape (`SHAPE` in `kernels/cuda_gemv.py`) defaults to 8 warps x
2 rows. `python kernels/bench_cuda.py --shapes` sweeps block shapes, and `python engine/tune_step.py` tunes the
Triton GEMV inside the real step. Decode is memory-bandwidth bound (each token reads ~1.53 GB of weights), so expect
roughly tok/s ≈ 0.6 x your GPU's GB/s. That is a rule of thumb from the 3050, not a measurement.

## What is where

| path | what |
|---|---|
| `engine/rumik_int4.py` | the base engine: int4 weight loading, static KV cache, CUDA graphs, Gumbel sampling, the model explained |
| `engine/rumik_fused.py` | the fast engine used everywhere: fused step, flash-decoding, CUDA GEMVs, int4 prefill, int8 head |
| `engine/fused.py` | Triton kernels: embed+LayerNorm, rope/KV-write/SiLU "post", decode attention, flash-decoding, add+LayerNorm |
| `engine/sample.py` | stop predictor + top-k (sorted int64 keys) + Gumbel sampling in 3 Triton kernels |
| `engine/stream_mimi.py` | exact streaming Mimi decoder (our own conv state on top of HF's transformer cache) |
| `engine/say.py`, `speak.py` | offline clips / live streaming playback with an adaptive prebuffer |
| `engine/check_*.py`, `test_attn.py` | correctness checks: vs the reference Cohere2 layer, old vs new kernels, KL of sampling distributions |
| `kernels/cuda_gemv.py` | hand-written CUDA int4 GEMV (fp16 magic-number LOP3 + HFMA2) and int8 GEMV (PRMT) |
| `kernels/mm_int4.py` | int4-weight matmul for the prefill (tensor cores, weights never leave int4) |
| `kernels/int4_gemv.py` | the Triton GEMV versions v1–v5, kept for the write-up |
| `kernels/roofline.py`, `sass_count.py` | how the bottleneck was found: read-only roofline, instruction counts in SASS |
| `ui/` | Rumik Studio: local web UI (standard library server + one HTML page) |
| `mimi/` | Mimi codec speed on CPU vs GPU |
| `modal_*.py` | steps 1–3 on a cloud L4 (Modal): baseline, torchao int8/int4, the int4 export |

`engine/README.md` is the step-by-step log (steps 4–12): every measurement, dead end and bug, with numbers.
`kernels/README.md` covers the GEMV work in detail.

## The int4 format (`modal_export_int4.py`)
- Every projection of the 36 layers is int4 with group 128, min/max round-to-nearest:
  `w = code * scale + min`. Codes are packed two per byte (low nibble = even column); scale and min are fp16 per
  group. That is about 1.49 GB.
- The embedding table is int8 with one fp16 scale per row (0.57 GB); it is only looked up, never multiplied.
- The 16,385 audio rows of the tied output head are kept bf16 (67 MB); the engine re-quantizes them to int8 per row
  at load time.
- Mean relative weight error is 10.5% (worst: late v_proj, ~15%). On an L4 with torchao's int4 kernel, the same
  recipe gave the same CER as bf16 (Indic-Transcribe, en 0.000 / hi 0.073).

## Earlier steps on a cloud L4 (Modal)
| | tok/s | x real time | time to first audio | CER en / hi |
|---|---|---|---|---|
| bf16 (step 1, compiled) | 41.8 | 0.40 | 216 ms | - |
| int8 (torchao) | 64.8 | 0.62 | 162 ms | 0.008 / 0.073 |
| int4 g128 (torchao tinygemm) | 130.3 | 1.22 | 99 ms | 0.000 / 0.073 |

Steps 1–2 ran on Ashwin Hegde's [rumik-latency](https://github.com/Ashwinhegde19/rumik-latency) engine, which has no
licence file, so `modal_bench.py` / `modal_quant.py` fetch it at a pinned commit instead of copying it. The engine in
`engine/` does not use it.

## Licences and credits
- **The model weights** (including the int4 version) are rumik-oss-1 by Rumik AI, a derivative of Cohere Labs'
  tiny-aya-fire: **CC-BY-NC 4.0 + the Cohere Labs acceptable-use policy. Non-commercial use only**, which per the
  model card includes self-hosted deployments. The Mimi codec is Kyutai's, CC-BY-4.0.
- **The code in this repository:** no licence chosen yet.
- Thanks to Rumik AI, Cohere Labs, Kyutai, Ashwin Hegde (rumik-latency) and Mrinaal Arora (latency write-up).
