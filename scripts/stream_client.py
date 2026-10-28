#!/usr/bin/env python3
"""Minimal client for POST /v1/audio/speech/stream (standard library only).

Reads the chunked PCM as it arrives, prints per-chunk timing, and writes a WAV
so you can listen to the result.

    python scripts/stream_client.py --text "Hello. This streams sentence by sentence." -o out.wav
"""
import argparse
import http.client
import json
import os
import time
import wave


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", 9090)))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--voice", default=os.getenv("DEFAULT_VOICE", "Emilia"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--l16", action="store_true",
                    help="Opt in to big-endian audio/L16; convert back to WAV little-endian")
    ap.add_argument("--text", default=(
        "Hello, this is a streaming test. The quick brown fox jumps over the lazy dog. "
        "Streaming synthesis processes one sentence at a time."))
    ap.add_argument("-o", "--output", default="stream_test_output.wav")
    a = ap.parse_args()

    conn = http.client.HTTPConnection(a.host, a.port, timeout=300)
    headers = {"Authorization": f"Bearer {a.api_key}", "Content-Type": "application/json"}
    if a.l16:
        headers["Accept"] = "audio/L16"
    conn.request("POST", "/v1/audio/speech/stream",
                 json.dumps({"input": a.text, "voice": a.voice, "speed": a.speed}),
                 headers)
    t0 = time.time()
    resp = conn.getresponse()
    if resp.status != 200:
        raise SystemExit(f"HTTP {resp.status}: {resp.read().decode(errors='replace')}")
    rate = int(resp.getheader("X-Sample-Rate", "16000"))
    media_type = resp.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
    encoding = resp.getheader("X-Audio-Encoding", "").lower()
    if a.l16:
        if media_type != "audio/l16" or encoding != "s16be":
            raise SystemExit("Server did not honor explicit audio/L16 request")
    elif encoding and encoding != "s16le":
        raise SystemExit("Unexpected PCM encoding (default mode expects s16le)")
    pcm = bytearray()
    while True:
        data = resp.read1(1 << 16)
        if not data:
            break
        pcm += data
        print(f"+{time.time() - t0:6.2f}s  chunk {len(data):>7} bytes  (total {len(pcm)})")
    if a.l16:
        if len(pcm) % 2:
            raise SystemExit("Truncated L16 sample")
        pcm[::2], pcm[1::2] = pcm[1::2], pcm[::2]
    with wave.open(a.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm))
    print(f"{len(pcm) / (rate * 2):.2f}s of audio in {time.time() - t0:.2f}s -> {a.output}")


if __name__ == "__main__":
    main()
