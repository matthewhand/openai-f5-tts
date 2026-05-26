#!/usr/bin/env python3
"""
Test client for the /v1/audio/speech/stream endpoint.

Streams int16 PCM from the server and writes a valid WAV file for playback verification.

Usage:
    python test_stream_client.py [--host HOST] [--port PORT] [--voice VOICE]
                                 [--text TEXT] [--output OUTPUT]
"""
import argparse
import struct
import wave
import requests
import os
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes


def stream_to_wav(host, port, api_key, text, voice, speed, output_path):
    url = f"http://{host}:{port}/v1/audio/speech/stream"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": text, "voice": voice, "speed": speed}

    print(f"Connecting to {url} ...")
    chunks = []
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        reported_rate = resp.headers.get("X-Sample-Rate", str(SAMPLE_RATE))
        print(f"Server sample rate: {reported_rate} Hz — receiving chunks ...")
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                chunks.append(chunk)
                print(f"  received {len(chunk)} bytes (total {sum(len(c) for c in chunks)})", end="\r")

    pcm_data = b"".join(chunks)
    print(f"\nTotal PCM received: {len(pcm_data)} bytes "
          f"({len(pcm_data) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH):.2f}s)")

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(int(reported_rate))
        wf.writeframes(pcm_data)

    print(f"WAV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stream TTS to WAV file")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 9090)))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "your_api_key_here"))
    parser.add_argument("--voice", default=os.getenv("DEFAULT_VOICE", "Emilia"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--text",
        default=(
            "Hello, this is a streaming test. "
            "The quick brown fox jumps over the lazy dog. "
            "Streaming synthesis processes one sentence at a time."
        ),
    )
    parser.add_argument("--output", default="stream_test_output.wav")
    args = parser.parse_args()

    stream_to_wav(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        text=args.text,
        voice=args.voice,
        speed=args.speed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
