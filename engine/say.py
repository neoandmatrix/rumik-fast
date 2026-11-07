"""Text -> speech on the laptop with our int4 engine; prints speed, writes wavs to out/.
  uv run python say.py                       # the built-in test lines
  uv run python say.py --hard                # 4 harder lines (long, numbers, switching, emotion)
  uv run python say.py "नमस्ते, कैसे हैं आप?"   # your own line
Mimi decodes the whole clip at the end here (not streamed yet).
"""
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, MimiModel

from rumik_fused import RumikFused
from lines import HARD, LINES
from rumik_int4 import WEIGHTS, RumikInt4

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

args = sys.argv[1:]
if args == ["--hard"]:
    lines = HARD
elif args:
    lines = [("custom", "Ira", "calm, Hindi accent, steady pace", " ".join(args))]
else:
    lines = LINES
tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)   # repo code read: config only
OLD = os.environ.get("ENGINE") == "old"                   # ENGINE=old: the unfused step (step 4)
eng = RumikInt4() if OLD else RumikFused()
eng.capture()
mimi = MimiModel.from_pretrained(WEIGHTS / "hf/codec").eval().to("cuda")
print(f"VRAM in use {torch.cuda.memory_allocated() / 2**30:.2f} GiB (peak {torch.cuda.max_memory_allocated() / 2**30:.2f})")
eng.generate(tok('<text>Ira: <description="calm, steady pace"> Warm up.<audio>').input_ids, seed=1)   # warm-up
rows = []
for tag, spk, desc, text in lines:
    ids = tok(f'<text>{spk}: <description="{desc}"> {text}<audio>').input_ids
    toks, st = eng.generate(ids, seed=0)
    t = time.perf_counter()
    with torch.inference_mode():
        wav = mimi.decode(eng.to_codes(toks).cuda()).audio_values[0, 0].float().cpu()
    st["mimi_ms"] = round((time.perf_counter() - t) * 1e3)
    sf.write(OUT / f"{tag}{'_old' if OLD else ''}.wav", wav.numpy(), 24000)
    st["rtf_x"] = round(st["decode_tok_s"] / 100, 2)
    st["hit_max"] = st["tokens"] >= eng.max_new          # True = never stopped on its own (cut at 18 s)
    rows.append({"id": tag, **st})
    print(tag, json.dumps(st), flush=True)
(OUT / (("runs_hard" if lines is HARD else "runs") + ("_old" if OLD else "") + ".json")).write_text(json.dumps(rows, indent=1))
