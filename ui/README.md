# Rumik Studio: local web UI

Type a line, pick a voice, tone, accent and pace, insert `<laugh>` `<chuckle>` `<sigh>` at the cursor, and hear it
stream from the laptop GPU (our int4 engine, ~97 tok/s) while it is being generated.

    cd 06-rumik-fast/ui && uv run python server.py
    # open http://127.0.0.1:8765   (listens on this laptop only; Ctrl-C to stop)

- `server.py`: standard library only (http.server). It loads RumikFused + Mimi once and serves `index.html`.
  POST /speak streams binary frames over a chunked response: `M` (prompt + prebuffer), `A` (int16 PCM, one 80 ms
  frame at a time), `S` (stats), `E` (error). One generation at a time.
- `index.html`: Web Audio playback with the same adaptive prebuffer as engine/speak.py, a live waveform, stats
  (first sound, first frame, tok/s, length, prebuffer, stalls), examples, and a history with replay/download and
  "use these settings". Same seed + same settings = the same take.
- Takes are saved to `ui/out/` with `history.json` (last 200).
- Test, 2026-11-07: Siya, excited, fast, a Hindi + English line with `<laugh>`. 97.1 tok/s; the first audio bytes
  reached the client after 288 ms; 6.8 s of audio.

Controls (from the model card): speakers Ira / Aisha / Siya / Zoya; tone happy / sad / angry / excited /
professional (or neutral); accent Hindi / Telugu / Tamil / Kannada / Bengali / Punjabi / Indian English;
pace slow / steady / fast. The rumik-oss-1 licence is non-commercial (research / personal use).
