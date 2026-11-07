"""Download what the engine needs into ./weights (about 2.55 GB):
  weights/int4/  rumik-oss-1-int4.safetensors (2.13 GB) + export.json   from bringagi/rumik-oss-1-int4
  weights/hf/    config, tokenizer, model code, Mimi codec (~410 MB)     from rumik-ai/rumik-oss-1 (not the bf16 weights)

  python download_weights.py            (or set RUMIK_WEIGHTS=/some/dir to put them elsewhere)
Both repos are public; the model is CC-BY-NC 4.0 (non-commercial), see the model cards.
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get("RUMIK_WEIGHTS", Path(__file__).resolve().parent / "weights"))
INT4_REPO = "bringagi/rumik-oss-1-int4"
BASE_REPO = "rumik-ai/rumik-oss-1"
BASE_FILES = ["config.json", "generation_config.json", "configuration_rumik_oss.py", "modeling_rumik_oss.py",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "LICENSE", "NOTICE",
              "codec/config.json", "codec/preprocessor_config.json", "codec/model.safetensors"]

for f in ("rumik-oss-1-int4.safetensors", "export.json"):
    print("int4:", f, flush=True)
    hf_hub_download(INT4_REPO, f, local_dir=ROOT / "int4")
for f in BASE_FILES:
    print("hf:", f, flush=True)
    hf_hub_download(BASE_REPO, f, local_dir=ROOT / "hf")
print(f"done: {ROOT}")
