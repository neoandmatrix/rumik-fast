"""06 Rumik fast inference: step 1, reproduce Ashwin Hegde's streaming engine (github.com/Ashwinhegde19/rumik-latency)
off the H100 it was built and measured on.

His repo has no licence file, so its code is not copied here: the image fetches rumik_fast.py and texts.json from
GitHub at a pinned commit. Rumik-oss-1 weights (CC-BY-NC) come from the voice-models volume (cached by 05).

  modal run modal_bench.py                 # L4, all texts x 3 seeds after warm-up
  modal run modal_bench.py --gpu A10G      # another GPU
-> prints a table; JSON + a few wavs on the voice-results volume under rumik-fast/<gpu>/
"""
import modal

SHA = "15037e1c8040f03ba2b597f654699f17c15def09"     # rumik-latency HEAD on 2026-11-06
RAW = f"https://raw.githubusercontent.com/Ashwinhegde19/rumik-latency/{SHA}"
REPO = "rumik-ai/rumik-oss-1"
HF, RES = "/models", "/results"
models_vol = modal.Volume.from_name("voice-models")
results_vol = modal.Volume.from_name("voice-results")

image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "libsndfile1")
         .pip_install("torch==2.8.0", "triton==3.4.0", "transformers==4.57.6", "numpy", "soundfile",
                      "huggingface_hub<1.0", "sentencepiece")
         .run_commands(f"mkdir -p /eng && curl -fsSL {RAW}/rumik_fast.py -o /eng/rumik_fast.py "
                       f"&& curl -fsSL {RAW}/texts.json -o /eng/texts.json")
         .env({"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false"}))
app = modal.App("rumik-fast")

# bytes read per decode step: 2.81B non-embedding params + the 16,385 audio rows of the tied head, bf16
STEP_BYTES = (2.81e9 + 16385 * 2048) * 2
PEAK_GBS = {"L4": 300, "A10G": 600, "T4": 320, "L40S": 864, "A100": 1555, "H100": 3350}


def bench(gpu: str, seeds: int, warmup: int):
    import json
    import statistics as st
    import sys
    import time
    from pathlib import Path
    import soundfile as sf
    import torch
    sys.path.insert(0, "/eng")
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel
    from rumik_fast import START_PREFIX, FastLM, StreamingMimi, StreamingTTS

    dev = torch.cuda.get_device_properties(0)
    print(f"{dev.name}, {dev.total_memory / 2**30:.1f} GiB, sm_{dev.major}{dev.minor}", flush=True)
    t = time.time()
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    mimi = MimiModel.from_pretrained(REPO, subfolder="codec").eval().cuda()
    load_s = time.time() - t
    t = time.time()
    lm = FastLM(model, use_compile=True, kv_buckets=(256, 512, 1024), attn="f32",
                compile_prefill=True, prefill_buckets=(32, 64, 128, 256),
                max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30)
    tts = StreamingTTS(lm, tok, smimi=StreamingMimi(mimi))
    capture_s = time.time() - t
    print(f"load {load_s:.0f}s, graph capture + compile {capture_s:.0f}s, "
          f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    texts = json.loads(Path("/eng/texts.json").read_text())
    prompt = lambda x: f'<text>{x["speaker"]}: <description="{x["description"]}"> {x["text"]}<audio>'  # noqa: E731
    for i in range(warmup):
        tts.run(prompt(texts[i % len(texts)]), seed=1)
    out = Path(RES) / "rumik-fast" / gpu
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for x in texts:
        for s in range(seeds):
            for prefix in (False, True):
                r = tts.run(prompt(x), seed=1000 + s, keep_audio=(s == 0),
                            start_prefix=START_PREFIX if prefix else None)
                pcm = r.pop("_pcm", None)
                if pcm is not None:
                    sf.write(out / f"{x['id']}{'_prefix' if prefix else ''}.wav",
                             pcm.float().clamp(-1, 1).numpy(), 24000)
                r.pop("tokens", None); r.pop("frame_ready_s", None); r.pop("frame_dbfs", None)
                rows.append({"id": x["id"], "prefix": prefix, **r})
    (out / "runs.json").write_text(json.dumps(rows, indent=1))
    results_vol.commit()

    def med(key, prefix=False):
        v = [r[key] for r in rows if r.get(key) is not None and r["prefix"] == prefix]
        return st.median(v) if v else float("nan")
    ms = med("per_token_ms")
    summary = {
        "gpu": dev.name, "load_s": round(load_s), "capture_s": round(capture_s),
        "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "ttfa_ms_p50": round(med("ttfa_s") * 1e3, 1),
        "first_audible_ms_p50": round(med("ttfa_nonsilent_s") * 1e3, 1),
        "first_audible_ms_p50_with_prefix": round(med("ttfa_nonsilent_s", True) * 1e3, 1),
        "prefill_ms_p50": round(med("prefill_s") * 1e3, 1),
        "mimi_frame_ms_p50": round(med("mimi_frame_ms_p50"), 2),
        "decode_tok_per_s_p50": round(med("decode_tok_per_s"), 1),
        "per_token_ms_p50": round(ms, 2),
        "rtfx_p50": round(med("rtfx"), 2),
        "stall_free": f"{sum(r.get('stall_free', False) for r in rows)}/{len(rows)}",
        "weight_read_gb_per_s": round(STEP_BYTES / (ms / 1e3) / 1e9),
        "peak_gb_per_s": PEAK_GBS.get(gpu),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    return summary


KW = dict(image=image, volumes={HF: models_vol, RES: results_vol}, secrets=[modal.Secret.from_name("huggingface")],
          timeout=3600)


@app.function(gpu="L4", **KW)
def bench_l4(seeds: int = 3, warmup: int = 3):
    return bench("L4", seeds, warmup)


@app.function(gpu="A10G", **KW)
def bench_a10g(seeds: int = 3, warmup: int = 3):
    return bench("A10G", seeds, warmup)


FNS = {"L4": bench_l4, "A10G": bench_a10g}


@app.local_entrypoint()
def main(gpu: str = "L4", seeds: int = 3, warmup: int = 3):
    import json
    print(json.dumps(FNS[gpu].remote(seeds=seeds, warmup=warmup), indent=1))
