"""06 Rumik fast inference: export an int4 copy of Rumik-oss-1's LM for the laptop (RTX 3050, 4 GB). CPU only.

Same recipe that step 2 showed lossless on the L4 (torchao int4 weight-only, group 128, round-to-nearest min-max), but
in a plain layout of our own so a hand-written kernel can read it:
  <proj>.qweight  uint8 [out, in/2]    two 4-bit codes per byte, low nibble = even column
  <proj>.scale    fp16  [out, in/128]  w ~= code * scale + min
  <proj>.min      fp16  [out, in/128]
  proj = model.layers.{i}.{self_attn.{q,k,v,o}_proj | mlp.{gate,up,down}_proj}
  model.embed_tokens.qweight int8 [V, 2048] + .scale fp16 [V]  (per-row int8: only looked up, never multiplied)
  head.audio_weight bf16 [16385, 2048] + head.audio_ids int64   (the tied head's audio rows, kept exact)
  everything else (norms, stop predictor, ...) copied as-is.
Config, tokenizer, model code and the Mimi codec are small: download them from the HF repo without the big weights.

  modal run modal_export_int4.py   -> voice-results volume: rumik-fast/int4/{rumik-oss-1-int4.safetensors, export.json}
"""
import modal

REPO = "rumik-ai/rumik-oss-1"
HF, RES = "/models", "/results"
models_vol = modal.Volume.from_name("voice-models")
results_vol = modal.Volume.from_name("voice-results")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "safetensors", "huggingface_hub<1.0", "numpy",
                      index_url="https://download.pytorch.org/whl/cpu", extra_index_url="https://pypi.org/simple")
         .env({"HF_HOME": HF}))
app = modal.App("rumik-export-int4")
GROUP = 128
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@app.function(image=image, cpu=8, memory=32768, volumes={HF: models_vol, RES: results_vol}, timeout=3600,
              secrets=[modal.Secret.from_name("huggingface")])
def export():
    import json
    import time
    from pathlib import Path
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(8)
    src = Path(snapshot_download(REPO, local_files_only=True))
    cfg = json.loads((src / "config.json").read_text())
    files = sorted(src.glob("*.safetensors"))
    print("source:", [f.name for f in files], flush=True)

    def q4(w):
        out_f, in_f = w.shape
        g = w.float().view(out_f, in_f // GROUP, GROUP)
        mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale = ((mx - mn) / 15).clamp(min=1e-8)
        code = ((g - mn) / scale).round().clamp(0, 15).to(torch.uint8)
        scale16, mn16 = scale.half(), mn.half()
        deq = code.float() * scale16.float() + mn16.float()                 # what the kernel will compute
        err = ((deq - g).norm() / g.norm()).item()
        code = code.view(out_f, in_f)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
        return packed, scale16.view(out_f, -1).contiguous(), mn16.view(out_f, -1).contiguous(), err

    t0 = time.time()
    out, errs, n_q = {}, {}, 0
    for f in files:
        with safe_open(str(f), "pt") as st:
            for name in st.keys():
                w = st.get_tensor(name)
                base = name[:-len(".weight")] if name.endswith(".weight") else None
                if base and base.split(".")[-1] in PROJ and w.dim() == 2:
                    assert w.shape[1] % GROUP == 0, (name, w.shape)
                    out[base + ".qweight"], out[base + ".scale"], out[base + ".min"], errs[base] = q4(w)
                    n_q += w.numel()
                elif name == "model.embed_tokens.weight":
                    a, e, last = cfg["first_unit_id"], cfg["audio_end_token_id"], cfg["last_unit_id"]
                    ids = torch.tensor(list(range(a, last + 1)) + ([e] if e > last else []), dtype=torch.int64)
                    out["head.audio_ids"] = ids
                    out["head.audio_weight"] = w.index_select(0, ids).to(torch.bfloat16).contiguous()
                    wf = w.float()
                    s = (wf.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
                    out["model.embed_tokens.qweight"] = (wf / s).round().clamp(-127, 127).to(torch.int8)
                    out["model.embed_tokens.scale"] = s.half().view(-1)
                    errs["model.embed_tokens"] = ((out["model.embed_tokens.qweight"].float() * s - wf).norm()
                                                  / wf.norm()).item()
                elif name == "lm_head.weight" and cfg.get("tie_word_embeddings"):
                    continue
                else:
                    out[name] = w.contiguous()
                print(f"{name} {tuple(w.shape)} {time.time() - t0:.0f}s", flush=True)

    dst = Path(RES) / "rumik-fast" / "int4"
    dst.mkdir(parents=True, exist_ok=True)
    meta = {"format": "rumik-int4-g128-minmax-v1", "group": GROUP, "source": REPO,
            "audio_rows": int(out["head.audio_ids"].numel()), "quantized_params": n_q}
    save_file(out, str(dst / "rumik-oss-1-int4.safetensors"), metadata={k: str(v) for k, v in meta.items()})
    size = (dst / "rumik-oss-1-int4.safetensors").stat().st_size
    worst = sorted(errs.items(), key=lambda kv: -kv[1])[:5]
    summary = {**meta, "file_gb": round(size / 1e9, 3), "seconds": round(time.time() - t0),
               "rel_err_mean": round(sum(v for k, v in errs.items() if "embed" not in k) / max(len(errs) - 1, 1), 5),
               "rel_err_embed_int8": round(errs.get("model.embed_tokens", float("nan")), 5),
               "worst": {k: round(v, 5) for k, v in worst}}
    (dst / "export.json").write_text(json.dumps({**summary, "rel_err": errs}, indent=1))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(export.remote(), indent=1))
