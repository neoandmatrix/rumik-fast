# Mimi decoder on the laptop (`bench_mimi.py`, 2026-11-06)
This uses the HF `MimiModel.decode` with random codes (8 codebooks), the plain non-streaming call, with no CUDA graph and no other GPU work running. One frame is 80 ms of audio, so that's the per-frame time budget.

| | 1 frame | per frame, 100 at once |
|---|---|---|
| CPU fp32, 6 threads (Ryzen 5 5600H) | 67 ms | 4.5 ms |
| CPU fp32, 12 threads | 48 ms | 5.0 ms |
| 3050 fp32 | 6.4 ms | 0.34 ms |
| 3050 fp16 | 6.6 ms | 0.18 ms |

- On the GPU, one frame takes about 8% of the 80 ms budget when it runs alone. The L4's 24 ms in step 1 was measured while the LM was running at the same time, so it included contention.
- On the CPU, one frame fits the budget but only just (48–67 ms), and the CPU also has to drive the LM loop. Decoding 2–4 frames per call costs much less per frame. That works after the first frame, which should be decoded on its own so first audio isn't delayed.
- Most of the single-frame cost is per-call overhead: 100 frames cost 4.5 ms each.
- Next: measure Mimi while the int4 LM runs, i.e. real contention on the GPU.
