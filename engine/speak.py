"""Live streaming TTS on the laptop: text in, audio out of the speakers while it is being generated.

  ../../04-gemma-stt-hindi/.venv/bin/python speak.py                  # interactive: type a line, hear it
  ../../04-gemma-stt-hindi/.venv/bin/python speak.py "Namaste ji"     # one line
  ../../04-gemma-stt-hindi/.venv/bin/python speak.py --set test        # the 4 test lines (--set hard: the 4 long ones)
  options: --speaker Ira --desc "calm, Hindi accent, steady pace" --prebuffer MS (default: adaptive) --chunk 1

Pipeline: engine.generate(on_frame=...) hands over every 80 ms frame (8 tokens) as soon as it is sampled ->
MimiStreamer decodes it (exact streaming, own CUDA stream so it overlaps the language model) -> a ring buffer ->
a PortAudio callback that plays it. Playback starts once `prebuffer` ms are buffered; if the buffer runs dry it
pauses and refills (counted as a stall) instead of crackling.
Adaptive prebuffer: at ~94 tok/s each 80 ms frame takes ~84.5 ms to make, so playback falls behind by ~5.6% of the
clip's length. The clip length is estimated from the text (Rumik speaks ~6.9 Devanagari or ~10-13 Latin characters
per second; the estimate is deliberately on the long side), and the prebuffer is est * (1 - tok_s/100) * 1.3 + 60 ms,
with tok_s = a running average of the measured speed (lines of >= 200 tokens; starts at 95).
Reports, per line: time to first frame, time to the first sound at the speaker (PortAudio's DAC timestamp),
stalls, and decode speed. Every run is saved to out/stream_<time>/ (wavs + runs.json).
"""
import argparse
import collections
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoTokenizer, MimiModel

from lines import HARD, LINES
from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS
from stream_mimi import SAMPLES_PER_FRAME, MimiStreamer

SR = 24000


def estimate_audio_s(text, cap=18.0):
    dev = len(re.findall(r"[\u0900-\u097F]", text))
    lat = len(re.findall(r"[A-Za-z0-9]", text))
    return min(dev / 6.5 + lat / 9.5, cap)


def adaptive_prebuffer_ms(text, tok_s):
    """playback falls behind by (1 - tok_s/100) of the clip; buffer that (x1.3 margin) + 60 ms"""
    return 1000 * estimate_audio_s(text) * max(0.0, 1 - tok_s / 100) * 1.3 + 60


class Player:
    """Ring buffer + PortAudio output stream (opened once, plays silence while idle)."""

    def __init__(self, prebuffer_ms):
        self.pre = int(SR * prebuffer_ms / 1000)
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=480, latency="low",
                                      callback=self._cb)
        self.stream.start()
        print(f"audio device latency {self.stream.latency * 1e3:.0f} ms", flush=True)
        self.reset()

    def reset(self):
        with self.lock:
            self.q, self.avail = collections.deque(), 0
            self.playing = self.started = self.finished = False
            self.stalls, self.stall_s, self.first_sound, self.done = 0, 0.0, None, threading.Event()

    def push(self, x):
        with self.lock:
            self.q.append(x)
            self.avail += len(x)

    def end(self):
        with self.lock:
            self.finished = True

    def _cb(self, out, frames, tinfo, status):
        out.fill(0)
        with self.lock:
            if not self.playing:
                if self.avail >= self.pre or (self.finished and self.avail > 0):
                    self.playing = True
                else:
                    if self.started and not self.finished:
                        self.stall_s += frames / SR
                    if self.finished and self.avail == 0 and self.started:
                        self.done.set()
                    return
            n = 0
            while n < frames and self.q:
                x = self.q[0]
                k = min(len(x), frames - n)
                out[n:n + k, 0] = x[:k]
                n += k
                self.avail -= k
                if k == len(x):
                    self.q.popleft()
                else:
                    self.q[0] = x[k:]
            if n and not self.started:
                self.started = True
                # when these samples reach the DAC, on the perf_counter clock
                self.first_sound = time.perf_counter() + (tinfo.outputBufferDacTime - tinfo.currentTime)
            if n < frames:
                if self.finished:
                    self.done.set()
                else:
                    self.stalls += 1
                    self.stall_s += (frames - n) / SR
                    self.playing = False                                    # refill to `pre` before resuming


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--set", choices=["test", "hard"])
    ap.add_argument("--speaker", default="Ira")
    ap.add_argument("--desc", default="calm, Hindi accent, steady pace")
    ap.add_argument("--prebuffer", type=float, default=None, help="ms; default adaptive")
    ap.add_argument("--chunk", type=int, default=1)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
    eng = RumikFused()
    eng.capture()
    mimi = MimiModel.from_pretrained(WEIGHTS / "hf/codec").eval().to("cuda")
    mstream = torch.cuda.Stream()
    player = Player(a.prebuffer or 300)
    rate = {"tok_s": 95.0}
    out_dir = Path(__file__).parent / "out" / time.strftime("stream_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)

    def speak(tag, spk, desc, text, play=True):
        streamer = MimiStreamer(mimi)
        player.reset()
        pre_ms = a.prebuffer if a.prebuffer is not None else adaptive_prebuffer_ms(text, rate["tok_s"])
        player.pre = int(SR * pre_ms / 1000)
        done_frames, pieces = 0, []
        stamps = {}
        t0 = time.perf_counter()

        def emit(codes):
            with torch.cuda.stream(mstream):
                wav = streamer.decode(codes.to("cuda", non_blocking=True)).float().cpu().numpy()
            pieces.append(wav)
            if play:
                player.push(wav)
            stamps.setdefault("first_audio_ready", time.perf_counter() - t0)

        def on_frame(toks):
            nonlocal done_frames
            stamps.setdefault("first_frame", time.perf_counter() - t0)
            codes = eng.to_codes(toks)
            new = codes.shape[-1] - done_frames
            if new >= (1 if done_frames == 0 else a.chunk):               # first frame alone: fastest first sound
                emit(codes[..., done_frames:])
                done_frames = codes.shape[-1]

        ids = tok(f'<text>{spk}: <description="{desc}"> {text}<audio>').input_ids
        toks, st = eng.generate(ids, seed=0, on_frame=on_frame)
        codes = eng.to_codes(toks)
        if codes.shape[-1] > done_frames:
            emit(codes[..., done_frames:])
        gen_done = time.perf_counter() - t0
        if st["tokens"] >= 200:                      # short lines (the warm-up) measure a misleadingly low speed
            rate["tok_s"] = 0.5 * rate["tok_s"] + 0.5 * min(st["decode_tok_s"], 100.0)
        player.end()
        if play:
            player.done.wait(timeout=60)
        wav = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
        sf.write(out_dir / f"{tag}.wav", wav, SR)
        row = {"id": tag, "speaker": spk, "text": text, "prompt_tokens": st["prompt_tokens"],
               "prefill_ms": st["prefill_ms"], "first_frame_ms": round(stamps["first_frame"] * 1e3),
               "first_audio_ready_ms": round(stamps["first_audio_ready"] * 1e3),
               "first_sound_ms": round((player.first_sound - t0) * 1e3) if play and player.first_sound else None,
               "decode_tok_s": st["decode_tok_s"], "audio_s": round(len(wav) / SR, 2),
               "generation_s": round(gen_done, 2), "stalls": player.stalls, "stall_ms": round(player.stall_s * 1e3),
               "prebuffer_ms": round(pre_ms), "estimated_audio_s": round(estimate_audio_s(text), 2)}
        print(f"{tag}: first sound {row['first_sound_ms']} ms (first frame {row['first_frame_ms']} ms), "
              f"{row['audio_s']} s audio in {row['generation_s']} s, {row['decode_tok_s']} tok/s, "
              f"stalls {row['stalls']} ({row['stall_ms']} ms), prebuffer {row['prebuffer_ms']} ms", flush=True)
        return row

    speak("warmup", "Ira", "calm, steady pace", "Warm up.", play=False)           # compile / allocate once
    rows = []
    if a.set:
        for tag, spk, desc, text in (LINES if a.set == "test" else HARD):
            rows.append(speak(tag, spk, desc, text))
    elif a.text:
        rows.append(speak("line", a.speaker, a.desc, " ".join(a.text)))
    else:
        print("type a line and press enter (empty line or Ctrl-D quits)")
        i = 0
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                break
            i += 1
            rows.append(speak(f"line{i}", a.speaker, a.desc, text))
    (out_dir / "runs.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    player.stream.stop()
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
