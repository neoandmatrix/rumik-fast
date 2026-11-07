"""How much does streaming Mimi slow the language model? Same line, same seed, decode tok/s with:
no Mimi | Mimi every 1/2/4 frames | Mimi in fp16. Also the SNR of fp16 Mimi vs fp32.
  ../../04-gemma-stt-hindi/.venv/bin/python bench_mimi_overlap.py
"""
import torch
from transformers import AutoTokenizer, MimiModel

from lines import LINES
from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS
from stream_mimi import MimiStreamer

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused()
eng.capture()
m32 = MimiModel.from_pretrained(WEIGHTS / "hf/codec").eval().to("cuda")
m16 = MimiModel.from_pretrained(WEIGHTS / "hf/codec", torch_dtype=torch.float16).eval().to("cuda")
mstream = torch.cuda.Stream()
_, spk, desc, text = LINES[1]
ids = tok(f'<text>{spk}: <description="{desc}"> {text}<audio>').input_ids


def run(mimi, chunk):
    st_m = MimiStreamer(mimi) if mimi is not None else None
    done = [0]
    wavs = []

    def on_frame(toks):
        if st_m is None:
            return
        codes = eng.to_codes(toks)
        if codes.shape[-1] - done[0] >= chunk:
            with torch.cuda.stream(mstream):
                wavs.append(st_m.decode(codes[..., done[0]:].cuda()).float().cpu())
            done[0] = codes.shape[-1]

    toks, st = eng.generate(ids, seed=0, on_frame=on_frame)
    return st["decode_tok_s"], (torch.cat(wavs) if wavs else None)


run(None, 1); run(m32, 1); run(m16, 1)                                       # warm-up
res = {}
for name, mimi, chunk in [("no Mimi", None, 1), ("fp32 chunk 1", m32, 1), ("fp32 chunk 2", m32, 2),
                          ("fp32 chunk 4", m32, 4), ("fp16 chunk 1", m16, 1), ("fp16 chunk 2", m16, 2)]:
    v = [run(mimi, chunk) for _ in range(2)]
    res[name] = v[0][1]
    print(f"{name:13s}: {max(x[0] for x in v):5.1f} tok/s", flush=True)
a, b = res["fp32 chunk 1"], res["fp16 chunk 1"]
n = min(len(a), len(b))
snr = 10 * torch.log10(a[:n].pow(2).sum() / (a[:n] - b[:n]).pow(2).sum())
print(f"fp16 Mimi vs fp32: SNR {snr:.1f} dB")
