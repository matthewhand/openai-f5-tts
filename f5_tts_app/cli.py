#!/usr/bin/env python
import argparse
import os
from gevent.pywsgi import WSGIServer
from app.app import app

def main():
    parser = argparse.ArgumentParser(description='OpenAI F5-TTS REST Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        # Run with Flask's built-in server in debug mode
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # Run with production WSGI server
        print(f"Starting OpenAI F5-TTS REST service on http://{args.host}:{args.port}")
        http_server = WSGIServer((args.host, args.port), app)
        http_server.serve_forever()

if __name__ == "__main__":
    main()