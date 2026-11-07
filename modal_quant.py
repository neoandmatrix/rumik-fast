"""06 Rumik fast inference, step 2: weight-only int8 / int4 on Ashwin Hegde's engine, speed + quality, Modal L4.

The engine (fetched at a pinned commit, not copied: the repo has no licence) keeps raw weight tensors and calls
F.linear. We subclass it so the four projection matrices of every layer are swapped for torchao quantized tensors
just before graph capture + compile; everything else (audio-only head, KV cache, Mimi, sampling) is unchanged.
  bf16: the baseline for quality (no compile: speed was measured in step 1)
  int8: torchao Int8WeightOnly (per-channel scales, dequant fused by inductor)
  int4: torchao Int4WeightOnly, group 128 (tinygemm kernel)
Quality: every clip is transcribed by Indic-Transcribe-core and scored against the input text (CER).

  modal run modal_quant.py                          # all three, 2 seeds per text
-> results volume rumik-fast/quant/<q>/*.wav, runs.json; rumik-fast/quant/summary.json
"""
import modal

SHA = "15037e1c8040f03ba2b597f654699f17c15def09"
RAW = f"https://raw.githubusercontent.com/Ashwinhegde19/rumik-latency/{SHA}"
REPO, IT_REPO = "rumik-ai/rumik-oss-1", "bodhan-ai/indic-transcribe-core"
HF, RES = "/models", "/results"
OUT = f"{RES}/rumik-fast/quant"
models_vol = modal.Volume.from_name("voice-models")
results_vol = modal.Volume.from_name("voice-results")
VOLS = {HF: models_vol, RES: results_vol}
ENV = {"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false"}

image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "libsndfile1")
         .pip_install("torch==2.8.0", "triton==3.4.0", "torchao==0.13.0", "transformers==4.57.6", "numpy",
                      "soundfile", "huggingface_hub<1.0", "sentencepiece")
         .run_commands(f"mkdir -p /eng && curl -fsSL {RAW}/rumik_fast.py -o /eng/rumik_fast.py "
                       f"&& curl -fsSL {RAW}/texts.json -o /eng/texts.json")
         .env(ENV))
asr_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1", "curl")
             .pip_install("torch==2.9.1", "torchaudio==2.9.1", "torchcodec", "transformers==4.57.3",
                          "huggingface_hub<1.0", "sentencepiece", "soundfile", "numpy", "scipy", "jiwer")
             .run_commands(f"mkdir -p /eng && curl -fsSL {RAW}/texts.json -o /eng/texts.json")
             .env(ENV))
app = modal.App("rumik-quant")
KW = dict(gpu="L4", volumes=VOLS, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)

# bytes per decode step: 2.81B projection params at the stored width (+ scales), plus the bf16 audio head
HEAD = 16385 * 2048 * 2
STEP_BYTES = {"bf16": 2.81e9 * 2 + HEAD, "int8": 2.81e9 * (1 + 2 / 2048) + HEAD,
              "int4": 2.81e9 * (0.5 + 4 / 128) + HEAD}


@app.function(image=image, **KW)
def gen(quant: str, seeds: int = 2, warmup: int = 3):
    import json
    import statistics as st
    import sys
    import time
    from pathlib import Path
    import soundfile as sf
    import torch
    sys.path.insert(0, "/eng")
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel
    from rumik_fast import FastLM, StreamingMimi, StreamingTTS

    def qweight(w):
        """one weight matrix -> torchao quantized tensor (F.linear dispatches to its kernel)"""
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
        if quant == "int8":
            cfg = Int8WeightOnlyConfig()
        else:
            try:
                cfg = Int4WeightOnlyConfig(group_size=128, version=1)      # tinygemm path
            except TypeError:
                cfg = Int4WeightOnlyConfig(group_size=128)
        with torch.inference_mode(False):
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta")
            lin.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
            quantize_(lin, cfg)
        return lin.weight

    class QLM(FastLM):
        def _capture(self):
            for l in model.model.layers:             # the engine holds its own (concatenated) copies by now
                for mod in (l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj, l.mlp.gate_proj,
                            l.mlp.up_proj):
                    mod.weight = None
            if quant != "bf16":
                t = time.time()
                for name in ("w_qkv", "w_o", "w_gu", "w_down"):
                    setattr(self, name, [qweight(w) for w in getattr(self, name)])
                torch.cuda.empty_cache()
                print(f"quantized to {quant} in {time.time() - t:.0f}s: {type(self.w_qkv[0]).__name__}", flush=True)
            super()._capture()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    mimi = MimiModel.from_pretrained(REPO, subfolder="codec").eval().cuda()
    t = time.time()
    kw = dict(kv_buckets=(256, 512, 1024), attn="f32", prefill_buckets=(32, 64, 128, 256), max_new_tokens=2048,
              min_new_tokens=8, temperature=0.8, top_k=30)
    lm = QLM(model, use_compile=quant != "bf16", compile_prefill=quant != "bf16", **kw)
    tts = StreamingTTS(lm, tok, smimi=StreamingMimi(mimi))
    capture_s = time.time() - t
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"{quant}: capture + compile {capture_s:.0f}s, peak VRAM {vram:.1f} GiB", flush=True)

    texts = json.loads(Path("/eng/texts.json").read_text())
    prompt = lambda x: f'<text>{x["speaker"]}: <description="{x["description"]}"> {x["text"]}<audio>'  # noqa: E731
    for i in range(warmup):
        tts.run(prompt(texts[i % len(texts)]), seed=1)
    out = Path(OUT) / quant
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for x in texts:
        for s in range(seeds):
            r = tts.run(prompt(x), seed=1000 + s, keep_audio=True)
            pcm = r.pop("_pcm", None)
            if pcm is not None:
                sf.write(out / f"{x['id']}_s{s}.wav", pcm.float().clamp(-1, 1).numpy(), 24000)
            for k in ("tokens", "frame_ready_s", "frame_dbfs"):
                r.pop(k, None)
            rows.append({"id": x["id"], "seed": s, **r})
            print(x["id"], s, r.get("decode_tok_per_s"), r.get("stopped_by"), flush=True)
    (out / "runs.json").write_text(json.dumps(rows, indent=1))
    results_vol.commit()

    med = lambda k: st.median([r[k] for r in rows if r.get(k) is not None])  # noqa: E731
    ms = med("per_token_ms")
    return {"quant": quant, "compiled": quant != "bf16", "capture_s": round(capture_s), "vram_gib": round(vram, 1),
            "decode_tok_per_s_p50": round(med("decode_tok_per_s"), 1), "per_token_ms_p50": round(ms, 2),
            "rtfx_p50": round(med("rtfx"), 2), "ttfa_ms_p50": round(med("ttfa_s") * 1e3, 1),
            "mimi_frame_ms_p50": round(med("mimi_frame_ms_p50"), 2),
            "stall_free": f"{sum(bool(r.get('stall_free')) for r in rows)}/{len(rows)}",
            "hit_max_tokens": sum(r.get("stopped_by") == "max_new_tokens" for r in rows),
            "weight_gb_per_s": round(STEP_BYTES[quant] / (ms / 1e3) / 1e9)}


@app.function(image=asr_image, **KW)
def score(quants: list[str]):
    import inspect
    import json
    import re
    import sys
    import unicodedata
    from pathlib import Path
    import jiwer
    from huggingface_hub import snapshot_download
    mdir = snapshot_download(IT_REPO, local_files_only=True)
    sys.path.insert(0, mdir)
    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained(mdir)
    print("transcribe signature:", inspect.signature(asr.transcribe), flush=True)
    texts = {x["id"]: x["text"] for x in json.loads(Path("/eng/texts.json").read_text())}

    def norm(t):
        t = unicodedata.normalize("NFC", t or "").lower()
        t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
        return re.sub(r"\s+", " ", t).strip()

    def tr(path, tid):
        if tid.startswith("en"):
            return str(asr(path, lang="en"))
        if tid.startswith("hinglish"):                 # Latin-script reference: ask for romanized output
            for kw in ({"romanized": True}, {"mode": "romanized"}):
                try:
                    return str(asr(path, lang="hi", **kw))
                except Exception:
                    pass
        return str(asr(path, lang="hi"))

    summary = {}
    for q in quants:
        rows = []
        for wav in sorted((Path(OUT) / q).glob("*.wav")):
            tid = wav.stem.rsplit("_s", 1)[0]
            hyp = tr(str(wav), tid)
            c = min(jiwer.cer(norm(texts[tid]), norm(hyp)), 1.0)
            rows.append({"id": tid, "file": wav.name, "hyp": hyp, "cer": round(c, 4)})
            print(q, wav.name, round(c, 3), hyp[:80], flush=True)
        (Path(OUT) / q / "asr.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        groups = {}
        for r in rows:
            groups.setdefault(r["id"].split("_")[0], []).append(r["cer"])
        summary[q] = {"cer_mean": round(sum(r["cer"] for r in rows) / max(len(rows), 1), 4),
                      **{f"cer_{g}": round(sum(v) / len(v), 4) for g, v in sorted(groups.items())}}
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(quants: str = "bf16,int8,int4", seeds: int = 2):
    import json
    qs = quants.split(",")
    res = list(gen.map(qs, kwargs={"seeds": seeds}, return_exceptions=True))
    for q, r in zip(qs, res):
        if not isinstance(r, dict):
            print(q, "FAILED:", repr(r)[:2000])
    speed = {r["quant"]: r for r in res if isinstance(r, dict)}
    print(json.dumps(speed, indent=1))
    quality = score.remote([q for q in qs if q in speed])
    out = {q: {**speed[q], **quality.get(q, {})} for q in speed}
    print(json.dumps(out, indent=1))
    with results_vol.batch_upload(force=True) as b:
        import io
        b.put_file(io.BytesIO(json.dumps(out, indent=1).encode()), "rumik-fast/quant/summary.json")
