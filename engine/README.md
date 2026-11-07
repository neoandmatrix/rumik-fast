# Our Rumik int4 engine (laptop, RTX 3050 4 GB)

```
cd 06-rumik-fast/engine
uv run python check.py      # maths vs the reference Cohere2 code (~1 min)
uv run python say.py        # 4 test lines -> out/*.wav + speed
uv run python say.py "नमस्ते, कैसे हैं आप?"
```
It needs `../weights/` (see STORAGE.md). The code is written from the model's published code, not copied from Ashwin's unlicensed repo.

## How it's put together (`rumik_int4.py`)
- **Weights:** our int4 file. Per layer, q/k/v/gate/up are concatenated into **one** 25,088×2,048 matrix, because in Rumik's parallel block they all multiply the same normed vector. So each layer is 3 GEMVs: in, o, down.
- **Prefill:** the whole prompt at once. Weights are dequantized to bf16 and multiplied with ordinary matmuls.
- **Decode step:** one token through 36 layers using our Triton int4 GEMV (`../kernels`). The K/V cache is static, attention runs in fp32 over a fixed-size *bucket* of the cache (256/512/1024/2048 slots, masked), and the head is audio-only (16,385 rows).
- **Sampling on the GPU:** top-30, temperature 0.8, by the Gumbel-max trick. The stop predictor runs every step.
- **CUDA graphs:** one per bucket. Replaying a graph launches the ~1,000 small kernels of a step with a single CPU call. Tokens are read back once per 8-token frame.
- **Gotcha:** `t[gpu_scalar_tensor]` silently syncs with the CPU, which breaks graph capture. Use `index_select`/`gather`.

## Results, 2026-11-06
- `check.py`: layer outputs match the reference within 0.8–1.2% (bf16 rounding). The decode step matches prefill logits within 1%, with the same top-1 token.
- `say.py`: **58–60 tok/s (0.6× real time)**. Prefill 640 ms (slow: dequant in plain torch), first frame 770 ms, VRAM 2.5 GiB including Mimi. The clips end on their own via the stop predictor: 6.2–7.7 s, speech from 0.2 s.

Step time is ~16.8 ms. The GEMVs alone take ~12.7 ms sustained (`../kernels`), so about 4 ms goes to the ~30 small ops per layer: norms, rope, cache writes, attention, adds, sampling.

## Next
1. Fuse the small ops: LN+rope+cache write, attention, residual adds.
2. A CUDA int4 GEMV.
3. A faster prefill dequant.
4. Streaming Mimi on a side stream, plus the start-prefix trick.

## Hard set (`say.py --hard`, results in out/runs_hard.json)
4 harder lines: a long Hindi story (Zoya, 201 prompt tokens), numbers/money/phone numbers (Ira), fast Hinglish
code-switching (Aisha), emotional Hindi + English with `<sigh>`/`<laugh>` (Siya).
- Speed: 54–56 tok/s (lower than the short lines: longer KV buckets, plus the GPU throttles over 30 s runs).
- Only hard_switch stopped on its own (15.1 s). The other three ran to the 1,800-token cap (18 s), with steady
  loudness to the end (no silent stall), so they look cut mid-speech rather than broken: the long story needs ~25 s.
- hard_emotion is very quiet (peak 0.21, "slow, emotional" description); listen to check if it is soft speech or noise.
- Fix for long text: split by sentence and generate each piece (what a real server does anyway), rather than
  raising max_seq (more KV memory; the 4 GB card is already near full with Mimi loaded).
- Listening verdict (user): speech quality ~9/10. Two issues, both outside the engine maths:
  1. numbers are read digit by digit (no text normalisation in front of the model);
  2. the emotional clip is quiet (needs output loudness normalisation).

## Step 5: fused decode step (`fused.py`, `rumik_fused.py`), 2026-11-06
nsys on the real step (`prof_engine.py`) showed 16.3 ms = 11.5 ms of int4 GEMVs + **4.8 ms of ~1,500 small
kernels**. The biggest pieces were 1.6 ms of copies (`.float()`/`.contiguous()`, including an fp32 copy of the KV
cache for attention every layer), 0.64 ms of fp32 attention matmuls, and ~2 ms of tiny elementwise ops.
Four Triton kernels replace them, so a layer is 6 launches instead of ~44 (1,596 -> 260 kernels per step):
- `embed_ln`: int8 embedding row + layer 0's LayerNorm;
- `post`: rope on q and k, k/v into the cache, silu(gate)*up, in one launch (program ids split into jobs);
- `attn`: decode attention with an online softmax straight over the bf16 cache, looping up to `pos` read on the
  GPU, so there are no KV buckets any more (one CUDA graph);
- `add_ln`: x += attn + mlp, then the next layer's LayerNorm.
`check_fused.py`: same sampled token as the old step on every step tested; hidden state within ~1% (bf16 rounding).

Two more fixes found along the way:
- **Pinned GEMV configs.** Triton's autotune re-times configs in every process and on this laptop sometimes picked
  bad ones (the same engine ran at 13.0 or 14.6 ms/step). Configs tuned *in isolation* (`kernels/tune_gemv.py`)
  were also worse in the real step (14.2 ms). `tune_step.py` tunes inside the real graph and writes
  `kernels/gemv_configs.json`; `gemv_int4` uses it when present.
- The generate loop is now double-buffered (async copy + event), so Python never makes the GPU wait. Small gain.

| | step 4 | step 5 |
|---|---|---|
| step at pos 300 | 16.3 ms | **13.1 ms** |
| say.py decode | 58–60 tok/s | **74.5–75.2 tok/s** (0.75x real time) |
| first frame | ~770 ms | ~740 ms |

What is left in the 13.1 ms: GEMVs ~11.6 ms (down GEMV only 111 GB/s), audio head 0.40 ms (bf16), attention
0.56 ms at pos 300, growing to ~2.7 ms at pos 2000 (16 programs only), top-k 0.14 ms.
Next: split-K attention (flash-decoding), then a better GEMV (split-K for down/o, wider loads), then an int8 audio head.

## Step 6: flash-decoding attention (`attn_split` + `attn_combine` in fused.py), 2026-11-07
The step-5 `attn` used one program per query head (16 programs on 16 SMs), and each of the 4 query heads sharing a
KV head read that K/V separately, so attention grew from 0.3 ms to 2.6 ms per step between pos 100 and pos 2047.
Now: grid (4 KV heads x 16 chunks). Each program takes one chunk of the cache for all 4 query heads of its KV head
(K/V read once, the 4 queries padded to a 16-row tile so q.k and p.v run as tl.dot on the tensor cores), and writes a
partial (max m, sum l, un-normalised acc). `attn_combine` merges them: w_i = exp(m_i - max m),
out = sum w_i acc_i / sum w_i l_i. The chunk length comes from `pos` on the GPU.
- `test_attn.py`: error vs fp32 attention 0.17% (exact p.v) / 0.21% (bf16 p.v); same tokens in `check_fused.py`.
- Shape sweep inside a CUDA graph (36 layers, 36 different caches): 16 chunks, 16 slots per block, 2 warps, bf16 p.v
  was best. BT=64 does not fit in shared memory.

| attention, 36 layers | pos 100 | 300 | 800 | 1500 | 2047 |
|---|---|---|---|---|---|
| one program per head | 0.30 ms | 0.66 | 1.49 | 2.55 | 2.61 |
| flash-decoding | **0.19** | **0.30** | **0.54** | **0.85** | **1.09** |

Whole step: pos 50 12.7 ms, pos 2000 13.35 ms (was 13.1 / 15.3). say.py: short lines 75–76 tok/s; the 18 s hard
lines **73.6–75.3 tok/s** (were 54–56 in step 4). The first run after the GPU has been idle is ~5% slower (warm-up).

## Step 7: hand-written CUDA int4 GEMV (`../kernels/cuda_gemv.py`), 2026-11-07
Finding the bottleneck (kernels/README.md has the details):
- read-only twin of the Triton kernel (same loads, no maths): 153 GB/s, the same as cuBLAS bf16. Our kernel: 114.
  So the memory side was fine and the maths per weight was the limit.
- SASS of the Triton kernel: **~6 instructions per weight** (I2F conversions, byte extracts, separate mul/add).
  The magic-number conversion in Triton (v4) and fp16 maths in Triton (v5) did not help: Triton does not pack fp16
  ops two per instruction here, and folding the 1024 offset into fp16 sums destroys precision (19–22% error).
- The CUDA kernel: 16-byte loads, `(w & 0x000F000F) | 0x64006400` makes two fp16 codes in one LOP3, HFMA2 does
  two multiply-adds per instruction. ~1.6 instructions per weight. x is staged once per 16-row block in shared
  memory in the matching pair order.
- Bug found with known inputs: the min term was counted 4x (a group of 128 spans 4 lanes); fixed with per-chunk sums.

| GEMV | Triton v1 | CUDA |
|---|---|---|
| fused in 25088x2048 | 140 GB/s | **167 GB/s** (95% of peak) |
| o 2048x2048 | 125 | **138** |
| down 2048x11008 | 127 | **155** |

Accuracy: error vs fp32 0.0023 (Triton 0.0017). `check_dist.py`: KL(Triton||CUDA) 0.007 nats on average over
40 steps, top-1 token the same 40/40.
Result: step 10.4 ms at pos 50, 11.3 ms at pos 2000; **say.py 93 tok/s** (short lines), **92 tok/s** (18 s lines).
First frame ~720 ms (now mostly prefill).

## Step 8: int4 prefill (`../kernels/mm_int4.py`, `RumikFused.prefill`), 2026-11-07
The old prefill dequantized every weight matrix to a temporary bf16 copy (with fp32 temporaries) for each prompt.
The new Triton kernel keeps W in int4. Per 128-column group it unpacks the codes (0..15, exact in bf16), runs
`tl.dot(X_tile, codes^T)` on the tensor cores, and applies the group's scale/min to the result:
acc += dot * scale[n] + rowsum(X)[m] * min[n].
- Per matrix at T=69: fused-in 13.8 -> 1.3 ms, o 7.7 -> 0.16 ms, down 41.6 -> 1.0 ms. Error vs fp32 0.0017.
- `check_prefill.py`: whole prefill 705 -> 50 ms (21 tokens), 2,190 -> 126 ms (106 tokens). The first audio
  token's distribution is unchanged (KL < 1e-4, same top-1).
- Bug found: Triton compiles a separate kernel when an int argument is a multiple of 16, so a 112-token prompt
  paid a ~1.3 s compile mid-request. Fixed with `do_not_specialize=["M"]` plus `mm_int4.warmup()` at capture time.
- Measure on AC power only: on battery the GPU is power-limited and decode fell to 50–81 tok/s (the say.py run
  with the laptop unplugged is not a valid result).
- On AC power (2026-11-07): prefill 50–109 ms (34–122 tokens), 189 ms (201 tokens); **first frame 131–191 ms**
  (271 ms for the 201-token prompt); decode 95 tok/s on short lines, 93.4–93.9 tok/s on the 18 s lines.

| | step 4 | now (step 8) |
|---|---|---|
| decode | 59 tok/s | **95 tok/s** |
| prefill, 34-token prompt | 642 ms | **50 ms** |
| first frame | ~770 ms | **~130–190 ms** |

## Step 9: live streaming (`stream_mimi.py`, `speak.py`), 2026-11-07
- `stream_mimi.MimiStreamer`: exact streaming Mimi. HF streams only the decoder transformer (KV cache). The SEANet
  convolutions got our own state: causal convs keep (kernel-1)*dilation input samples; transposed convs carry
  their overhanging tail (overlap-add, bias counted once). `check_stream.py`: 66 dB SNR vs a full decode over an
  18 s clip (past the 250-step window); ~6.4 ms per call, on its own CUDA stream.
- `speak.py`: PortAudio (sounddevice 0.5.6, installed in the venv) output stream kept open; ring buffer; pause and
  refill on underrun; first sound timed with PortAudio's DAC timestamp; runs saved to out/stream_<time>/.
  The audio device latency is 20 ms, and audio is ready ~14 ms after the first frame.
- **The limit is the speed.** At ~94 tok/s a frame takes 84.5 ms to make and 80 ms to play, so playback falls
  behind by ~5.6% of the clip (more at long positions, where decode is ~92 tok/s). Fixed prebuffers: 100 ms gives
  7–18 stalls per clip, 300 ms gives 0–1 stalls with first sound at ~510–555 ms.
  Adaptive prebuffer (6% of the length estimated from the text, + 50 ms):

| line | first frame | first sound | stalls |
|---|---|---|---|
| hinglish (5 s) | 132 ms | 526 ms | 0 |
| mixed (5.7 s) | 146 ms | 699 ms | 0 |
| hi (8.4 s) | 160 ms | 810 ms | 0 |
| 18 s lines | 180–270 ms | 1.25–1.5 s | 1 stall of ~1.1–1.3 s near the end |

  (One `en` run fell to 55 tok/s, with 5 s of stalls, right after start-up; unexplained so far. Another run of the
  same line: 94 tok/s, 1 stall.)
- To get a near-instant start with no stalls, decode must be faster than real time (>100 tok/s at every position).
  Candidates: int4 audio head (-0.3 ms), one fused sampling kernel (-0.25 ms), int8 KV cache for long positions.

## Step 10: faster tail, int8 audio head (`sample.py`, `gemv_int8` in ../kernels/cuda_gemv.py), 2026-11-07
- `sample.py`: stop predictor + top-30 + Gumbel sampling + counters in 3 Triton kernels (plus one torch.rand), down
  from ~25 kernels, including torch.topk and a per-step fp32 copy of the stop predictor's weights. Top-k sorts int64
  keys (sortable float bits << 32 | index) in 33 chunks of 512, then merges. `check_sample.py`: the top-30 values are
  equal in 80/80 cases (so the sampling distribution is identical; bf16 logits always had ties, so token-by-token
  comparison with the same random numbers is not meaningful), same stop decision 80/80.
- Audio head: int4 is too lossy here (simulated KL vs bf16 0.15 for g128, 0.10 even for g32). int8 with a per-row
  scale: KL 0.013, top-1 78/80. New CUDA kernel: PRMT places each byte under 0x64 (fp16 1024), minus 1152 = the
  signed weight. 200 us vs 398 us for bf16 (168 GB/s).
- Step: pos 50 9.81 ms, pos 800 10.21, pos 2000 10.72 (was 10.18 / 10.69 / 11.08).
- speak.py, live (Mimi running alongside): **97 tok/s** (5–8 s lines), 95–96 tok/s (18 s lines). **0 stalls on all
  8 lines.** Prebuffer adapts to the measured speed (running average over lines of >= 200 tokens):

| line | first frame | first sound | stalls |
|---|---|---|---|
| hinglish 5.8 s | 129 ms | **435 ms** | 0 |
| mixed 5.8 s | 143 ms | **528 ms** | 0 |
| en 6.3 s | 129 ms | 679 ms | 0 |
| hi 8.0 s | 156 ms | 688 ms | 0 |
| 18 s lines | 178–267 ms | 1.29–1.53 s | 0 |

  The first sound is now mostly the prebuffer, which is (1 - tok_s/100) x estimated length (from the text,
  deliberately on the long side) x 1.3. Going above 100 tok/s, or a better length estimate, removes most of it.

## Step 11 (attempt): squeezing past 100 tok/s live, 2026-11-07
- `bench_mimi_overlap.py`: no Mimi 98.2 tok/s; streaming Mimi fp32 every 1/2/4 frames 96.7/97.3/97.6; fp16 Mimi no
  faster (58.7 dB SNR vs fp32). Mimi costs only ~1.5%.
- CUDA int4 GEMV block shape (warps per block x rows per warp, now a template parameter): no shape is more than
  ~1–2% better than 8x2. fused-in 168 GB/s, o 143, down 159.
- So the step is ~95% weight reads at ~92% of the 176 GB/s peak. The ceiling for 1.53 GB per token is ~115 tok/s
  (theory), ~100–104 tok/s in practice for the bare step.
- What is left that moves bytes: 8-bit scales/mins ("double quantisation": the fp16 scale+min are ~6% of the bytes
  read), and an int8 KV cache (only matters at long positions).
