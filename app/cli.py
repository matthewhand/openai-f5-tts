#!/usr/bin/env python
import argparse
import os
import sys
try:
    from gevent.pywsgi import WSGIServer
except ImportError:
    WSGIServer = None  # Gevent not installed, will error in production mode
from app.server import app, PORT, ENGINES, DEFAULT_RESPONSE_FORMAT, DEFAULT_SPEED
from app.utils import AUDIO_FORMAT_MIME_TYPES

def run_server(args):
    """Run the Flask app as a server."""
    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        print(f"Starting OpenAI F5-TTS REST service on http://{args.host}:{args.port}")
        http_server = WSGIServer((args.host, args.port), app)
        http_server.serve_forever()

def cli_list_engines(args):
    """CLI: list available TTS engines."""
    for name in ENGINES:
        print(name)

def cli_list_voices(args):
    """CLI: list voices for a given engine."""
    engine = ENGINES.get(args.engine)
    if not engine:
        print(f"Engine '{args.engine}' not found", file=sys.stderr)
        sys.exit(1)
    for v in engine.list_voices():
        print(v)

def cli_speak(args):
    """CLI: generate speech and save to output file."""
    engine = ENGINES.get(args.engine)
    if not engine:
        print(f"Engine '{args.engine}' not found", file=sys.stderr)
        sys.exit(1)
    sample_rate, audio = engine.generate(
        args.text,
        args.voice,
        response_format=args.response_format,
        speed=args.speed,
        ref_audio=args.ref_audio
    )
    with open(args.output, 'wb') as f:
        f.write(audio)
    print(f"Generated speech saved to {args.output}")

def cli_mix(args):
    """CLI: mix multiple audio files."""
    import sys
    import soundfile as sf
    from app.engine import mix_audio
    arrays = []
    sr = args.sr
    for path in args.inputs:
        try:
            data, s = sf.read(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}", file=sys.stderr)
            sys.exit(1)
        if sr is None:
            sr = s
        elif sr != s:
            print(f"Sample rate mismatch: {path} is {s}Hz, expected {sr}Hz", file=sys.stderr)
            sys.exit(1)
        arrays.append(data)
    mixed = mix_audio(arrays, mode=args.mode)
    # Write floats directly without normalization
    sf.write(args.output, mixed, sr, subtype='FLOAT')
    print(f"Mixed output saved to {args.output}")

def main():
    parser = argparse.ArgumentParser(prog='openai-f5-tts')
    subparsers = parser.add_subparsers(dest='cmd', help='Commands')

    # Serve
    serve = subparsers.add_parser('serve', help='Run the REST server')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=PORT)
    serve.add_argument('--debug', action='store_true')
    serve.set_defaults(func=run_server)

    # List engines
    le = subparsers.add_parser('list-engines', help='List installed TTS engines')
    le.set_defaults(func=cli_list_engines)

    # List voices
    lv = subparsers.add_parser('list-voices', help='List voices for a TTS engine')
    lv.add_argument('--engine', required=True)
    lv.set_defaults(func=cli_list_voices)

    # Speak
    sp = subparsers.add_parser('speak', help='Generate speech from text')
    sp.add_argument('--engine', required=True)
    sp.add_argument('--text', required=True)
    sp.add_argument('--voice', default=os.getenv('DEFAULT_VOICE', 'Emilia'))
    sp.add_argument('--output', required=True)
    sp.add_argument('--format', dest='response_format', choices=list(AUDIO_FORMAT_MIME_TYPES.keys()), default=os.getenv('DEFAULT_RESPONSE_FORMAT', DEFAULT_RESPONSE_FORMAT), help='Response audio format')
    sp.add_argument('--speed', type=float, default=float(os.getenv('DEFAULT_SPEED', DEFAULT_SPEED)), help='Speech speed multiplier')
    sp.add_argument('--ref-audio', dest='ref_audio', help='Reference audio override path')
    sp.set_defaults(func=cli_speak)

    # Mix
    mix = subparsers.add_parser('mix', help='Mix multiple audio files')
    mix.add_argument('--inputs', nargs='+', required=True, help='Paths to input audio files')
    mix.add_argument('--mode', choices=['overlay', 'concat'], default='overlay', help='Mixing mode')
    mix.add_argument('--output', required=True, help='Output audio file path')
    mix.add_argument('--sr', type=int, help='Sample rate for output (defaults to inputs sample rate)')
    mix.set_defaults(func=cli_mix)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(0)
    args.func(args)

if __name__ == "__main__":
    main()
