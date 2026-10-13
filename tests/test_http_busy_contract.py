"""Exercise real Flask route bodies without importing Flask or model dependencies.

Extract only the two route function bodies from the pinned app/server.py AST.
The fake request and handler ensure no network, model import, or inference.
"""
import ast
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from generation_gate import SynthesisBusyError
from request_validation import parse_speech_request, parse_timestamp_request

server_source = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
module = ast.parse(server_source)
route_names = {"text_to_speech", "text_to_speech_with_timestamps"}
routes = []
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in route_names:
        node.decorator_list = []  # pure route body; Flask middleware not tested here
        routes.append(node)
assert len(routes) == 2, "Expected both routes in current server source"


class BusyEndpointTests(unittest.TestCase):
    def build_routes(self, payload):
        class _SyntheticHandler:
            calls = []
            def generate_speech(self, **kwargs):
                self.calls.append(kwargs)
                raise SynthesisBusyError("synthetic contention")

        handler = _SyntheticHandler()
        namespace = {
            "request": types.SimpleNamespace(
                get_json=lambda **kwargs: payload,
                args={},
            ),
            "jsonify": lambda response: response,
            "tts_handler": handler,
            "parse_speech_request": parse_speech_request,
            "parse_timestamp_request": parse_timestamp_request,
            "RequestValidationError": ValueError,
            "SynthesisBusyError": SynthesisBusyError,
            "DEFAULT_VOICE": "Emilia",
            "DEFAULT_RESPONSE_FORMAT": "mp3",
            "DEFAULT_SPEED": 1.0,
            "MAX_TEXT_CHARS": 2000,
            "AUDIO_FORMAT_MIME_TYPES": {"mp3": "audio/mpeg"},
            "logging": types.SimpleNamespace(error=lambda *args, **kwargs: None),
        }
        ast_module = ast.fix_missing_locations(ast.Module(body=routes, type_ignores=[]))
        exec(compile(ast_module, "app/server.py", "exec"), namespace)
        return namespace, handler

    def test_openai_style_route_returns_503_without_creating_audio(self):
        namespace, handler = self.build_routes({"input": "fictional hello"})
        payload, status = namespace["text_to_speech"]()
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "Speech generator busy; try again later"})
        self.assertEqual(len(handler.calls), 1)

    def test_timestamp_route_returns_503_before_audio_alignment(self):
        namespace, handler = self.build_routes({"text": "fictional hello"})
        payload, status = namespace["text_to_speech_with_timestamps"]("Emilia")
        self.assertEqual(status, 503)
        self.assertEqual(payload, {"error": "Speech generator busy; try again later"})
        self.assertEqual(len(handler.calls), 1)

    def test_invalid_payload_never_enters_synthesis(self):
        namespace, handler = self.build_routes({"input": [], "speed": 1.0})
        payload, status = namespace["text_to_speech"]()
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(handler.calls, [])


if __name__ == "__main__":
    unittest.main()
