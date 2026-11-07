"""Local web UI for our Rumik engine: type a line, pick a voice / tone / accent / pace, add <laugh> <chuckle> <sigh>,
and hear it stream from the laptop GPU while it is generated.

  cd 06-rumik-fast/ui && ../../04-gemma-stt-hindi/.venv/bin/python server.py      -> open http://127.0.0.1:8765

Standard library only (http.server). Listens on 127.0.0.1 only. One generation at a time (one GPU).
POST /speak streams a chunked binary response of frames: 1 type byte + 4-byte little-endian length + payload.
  M  JSON  the prompt that was sent to the model + the prebuffer the browser should wait for
  A  PCM   int16 mono 24 kHz audio, one Mimi frame (80 ms) at a time, as soon as it is decoded
  S  JSON  stats at the end (prefill, first frame, tok/s, length, ...)
Every clip is saved to ui/out/ and listed by GET /history.
"""
import json
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, MimiModel  # noqa: E402

from rumik_fused import RumikFused  # noqa: E402
from rumik_int4 import WEIGHTS  # noqa: E402
from speak import adaptive_prebuffer_ms, estimate_audio_s  # noqa: E402
from stream_mimi import MimiStreamer  # noqa: E402

SR = 24000
PORT = 8765
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
HISTORY = OUT / "history.json"
SPEAKERS = ["Ira", "Aisha", "Siya", "Zoya"]
TONES = ["", "happy", "sad", "angry", "excited", "professional"]
ACCENTS = ["Hindi", "Telugu", "Tamil", "Kannada", "Bengali", "Punjabi", "Indian English"]
PACES = ["slow", "steady", "fast"]

print("loading the engine ...", flush=True)
tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused()
eng.capture()
mimi = MimiModel.from_pretrained(WEIGHTS / "hf/codec").eval().to("cuda")
mstream = torch.cuda.Stream()
gpu_lock = threading.Lock()
rate = {"tok_s": 95.0}
history = json.loads(HISTORY.read_text()) if HISTORY.exists() else []


def description(tone, accent, pace):
    parts = ([tone] if tone else []) + [f"{accent} accent", f"{pace} pace"]
    return ", ".join(parts)


def frame(kind, payload):
    return kind.encode() + struct.pack("<I", len(payload)) + payload


def generate(req, send):
    spk = req.get("speaker") if req.get("speaker") in SPEAKERS else "Ira"
    tone = req.get("tone") if req.get("tone") in TONES else ""
    accent = req.get("accent") if req.get("accent") in ACCENTS else "Hindi"
    pace = req.get("pace") if req.get("pace") in PACES else "steady"
    seed = int(req.get("seed", 0)) if str(req.get("seed", "0")).lstrip("-").isdigit() else 0
    text = " ".join(str(req.get("text", "")).split())[:1000]
    if not text:
        raise ValueError("empty text")
    desc = description(tone, accent, pace)
    prompt = f'<text>{spk}: <description="{desc}"> {text}<audio>'
    pre_ms = adaptive_prebuffer_ms(text, rate["tok_s"])
    send(frame("M", json.dumps({"prompt": prompt, "prebuffer_ms": round(pre_ms),
                                 "estimated_audio_s": round(estimate_audio_s(text), 1)}).encode()))
    streamer = MimiStreamer(mimi)
    done, pieces, stamps = [0], [], {}
    t0 = time.perf_counter()

    def emit(codes):
        with torch.cuda.stream(mstream):
            wav = streamer.decode(codes.to("cuda", non_blocking=True)).float().cpu().numpy()
        pieces.append(wav)
        send(frame("A", (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()))
        stamps.setdefault("first_audio_ms", round((time.perf_counter() - t0) * 1e3))

    def on_frame(toks):
        stamps.setdefault("first_frame_ms", round((time.perf_counter() - t0) * 1e3))
        codes = eng.to_codes(toks)
        if codes.shape[-1] > done[0]:
            emit(codes[..., done[0]:])
            done[0] = codes.shape[-1]

    ids = tok(prompt).input_ids
    toks, st = eng.generate(ids, seed=seed, on_frame=on_frame)
    codes = eng.to_codes(toks)
    if codes.shape[-1] > done[0]:
        emit(codes[..., done[0]:])
    total = time.perf_counter() - t0
    if st["tokens"] >= 200:
        rate["tok_s"] = 0.5 * rate["tok_s"] + 0.5 * min(st["decode_tok_s"], 100.0)
    wav = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    name = time.strftime("%Y%m%d_%H%M%S") + f"_{spk}.wav"
    sf.write(OUT / name, wav, SR)
    row = {"file": name, "speaker": spk, "description": desc, "text": text, "seed": seed,
           "prompt_tokens": st["prompt_tokens"], "prefill_ms": st["prefill_ms"],
           "first_frame_ms": stamps.get("first_frame_ms"), "first_audio_ms": stamps.get("first_audio_ms"),
           "decode_tok_s": st["decode_tok_s"], "audio_s": round(len(wav) / SR, 2), "total_s": round(total, 2),
           "hit_cap": st["tokens"] >= eng.max_new, "time": time.strftime("%H:%M:%S")}
    history.insert(0, row)
    HISTORY.write_text(json.dumps(history[:200], indent=1, ensure_ascii=False))
    send(frame("S", json.dumps(row, ensure_ascii=False).encode()))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/history":
            return self._send(200, json.dumps(history, ensure_ascii=False).encode(), "application/json")
        if self.path == "/options":
            opts = {"speakers": SPEAKERS, "tones": TONES, "accents": ACCENTS, "paces": PACES}
            return self._send(200, json.dumps(opts).encode(), "application/json")
        if self.path.startswith("/audio/"):
            f = OUT / Path(self.path[len("/audio/"):]).name                  # no directory traversal
            if f.suffix == ".wav" and f.exists():
                return self._send(200, f.read_bytes(), "audio/wav")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/speak":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(min(n, 100_000)) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b"bad json", "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(data):
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            with gpu_lock:
                generate(req, send)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:                                                  # report to the page
            try:
                send(frame("E", f"{type(e).__name__}: {e}".encode()))
            except (BrokenPipeError, ConnectionResetError):
                return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    with torch.inference_mode():
        generate({"text": "Warm up.", "speaker": "Ira"}, lambda b: None)          # compile + allocate once
    (OUT / history.pop(0)["file"]).unlink(missing_ok=True)                  # the warm-up is not a take
    HISTORY.write_text(json.dumps(history[:200], indent=1, ensure_ascii=False))
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"ready: open http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
