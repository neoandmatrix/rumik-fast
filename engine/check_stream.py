"""Streaming Mimi vs a full decode of the same codes: SNR of the streamed audio, and time per call.
Uses an 18 s clip (longer than the transformer's 250-step window) and chunks of 1 frame, then 3 frames.
  uv run python check_stream.py
"""
import time

import torch
from transformers import AutoTokenizer, MimiModel

from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS
from lines import HARD
from stream_mimi import SAMPLES_PER_FRAME, MimiStreamer

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused()
eng.capture()
mimi = MimiModel.from_pretrained(WEIGHTS / "hf/codec").eval().to("cuda")
_, spk, desc, text = HARD[0]
toks, _ = eng.generate(tok(f'<text>{spk}: <description="{desc}"> {text}<audio>').input_ids, seed=0)
codes = eng.to_codes(toks).cuda()                                             # [1, 8, F]
F = codes.shape[-1]
with torch.inference_mode():
    full = mimi.decode(codes).audio_values[0, 0]
print(f"{F} frames = {F * 0.08:.1f} s; full decode {full.shape[-1]} samples (expected {F * SAMPLES_PER_FRAME})")


def snr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[:n].float(), b[:n].float()
    return (10 * torch.log10(b.pow(2).sum() / (a - b).pow(2).sum().clamp_min(1e-20))).item()


for chunk in (1, 3):
    st = MimiStreamer(mimi)
    parts, times = [], []
    for f0 in range(0, F, chunk):
        torch.cuda.synchronize(); t = time.perf_counter()
        parts.append(st.decode(codes[..., f0:f0 + chunk]))
        torch.cuda.synchronize(); times.append((time.perf_counter() - t) * 1e3)
    out = torch.cat(parts)
    first_s = snr(out[:SAMPLES_PER_FRAME * 125], full[:SAMPLES_PER_FRAME * 125])
    t = torch.tensor(times[2:])
    print(f"chunk {chunk} frame(s): SNR vs full decode {snr(out, full):.1f} dB (first 10 s {first_s:.1f} dB); "
          f"per call median {t.median():.1f} ms, max {t.max():.1f} ms ({t.median() / chunk:.1f} ms per frame)",
          flush=True)
