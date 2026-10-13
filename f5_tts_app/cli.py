#!/usr/bin/env python3
"""Small CLI wrapper for the supported Flask server (no model imports here).

Use os.execv to avoid a second independent app initialization, and let the
server's existing --host/--port flags control the listener.
"""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenAI F5-TTS local REST server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    target = Path(__file__).resolve().parents[1] / "app" / "server.py"
    if args.debug:
        os.environ["DEBUG_MODE"] = "true"
    os.execv(sys.executable, [sys.executable, str(target),
                             "--host", args.host, "--port", str(args.port)])


if __name__ == "__main__":
    main()
