"""Dependency-free source invariants; no Flask/model imports or downloads."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def read(self, path):
        return (ROOT / path).read_text(encoding="utf-8")

    def test_all_repository_python_files_parse(self):
        sources = list((ROOT / "app").glob("*.py"))
        sources += list((ROOT / "f5_tts_app").glob("*.py"))
        sources += list((ROOT / "tests").glob("*.py"))
        self.assertGreaterEqual(len(sources), 8)
        for path in sources:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_each_flask_route_is_api_key_protected(self):
        module = ast.parse(self.read("app/server.py"))
        routes = []
        for fn in module.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = fn.decorator_list
            is_route = any(
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.func.attr == "route"
                for dec in decorators
            )
            if is_route:
                routes.append(fn)
                self.assertTrue(
                    any(isinstance(dec, ast.Name) and dec.id == "require_api_key"
                        for dec in decorators), f"Unauthenticated route {fn.name}"
                )
        self.assertGreaterEqual(len(routes), 6)

    def test_http_debug_does_not_log_request_bodies_or_secret_headers(self):
        server = self.read("app/server.py")
        # Reading Accept for explicit PCM negotiation is safe; logging its value is not.
        for node in ast.walk(ast.parse(server)):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "logging":
                continue
            for argument in node.args:
                for value in ast.walk(argument):
                    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                        if value.value.id == "request":
                            self.assertNotIn(value.attr, ("headers", "url", "path"))
        self.assertNotIn("logging.debug('HTTP Request: %s %s'", server)
        self.assertNotIn("logging.debug(f\"Payload:", server)
        self.assertNotIn("logging.debug(f\"Headers:", server)
        handler = self.read("app/tts_handler.py")
        self.assertNotIn("text={text}", handler)
        self.assertNotIn("{gen_text}", handler)
        self.assertNotIn("Transcription result: {transcription}", handler)

    def test_gevent_access_and_exception_logging_disabled(self):
        # Gevent's default access log includes request URIs with private voices.
        # WSGI exception output may include paths or sensitive exception data.
        module = ast.parse(self.read("app/server.py"))
        calls = [node for node in ast.walk(module)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "WSGIServer"]
        self.assertEqual(len(calls), 1)
        settings = {kw.arg: kw.value for kw in calls[0].keywords}
        for option in ("log", "error_log"):
            self.assertIn(option, settings)
            self.assertIsInstance(settings[option], ast.Constant)
            self.assertIsNone(settings[option].value)

    def test_per_request_file_cleanup_has_no_global_after_request_race(self):
        server = self.read("app/server.py")
        self.assertNotIn("_temp_files_to_cleanup", server)
        self.assertNotIn("@app.after_request", server)
        self.assertIn("finally:\n        _delete_request_file(path)", server)
        self.assertIn("finally:\n            _delete_request_file(output_file_path)", server)

    def test_inference_and_cache_mutations_are_serialized(self):
        handler = self.read("app/tts_handler.py")
        self.assertIn("self._inference_lock = Semaphore(1)", handler)
        self.assertIn("with self._inference_lock:", handler)
        self.assertIn("sample_rate, audio_data = self.infer(", handler)

    def test_docker_defaults_to_loopback_without_inference_preflight(self):
        compose = self.read("docker-compose.yml")
        self.assertIn("127.0.0.1:9090:9090", compose)
        self.assertIn("DEFAULT_VOICE=${DEFAULT_VOICE:-Emilia}", compose)
        self.assertNotIn("app/test.py", compose)
        self.assertNotIn("sleep 60", compose)

    def test_readme_license_and_integrity_warning_present(self):
        readme = self.read("README.md")
        self.assertIn("CC-BY-NC", readme)
        self.assertIn("670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38", readme)
        self.assertIn("community beta", readme.lower())

    def test_stream_client_l16_opt_in_converts_to_little_endian_wav(self):
        """Exercise actual client with fictional HTTP and in-memory wave writer."""
        import importlib.util
        import sys
        from unittest import mock
        source = ROOT / "scripts" / "stream_client.py"
        spec = importlib.util.spec_from_file_location("synthetic_stream_client", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        requests, output = [], []
        class FakeResponse:
            status = 200
            chunks = [b"\x01\x02\xff\xfe", b""]
            def getheader(self, name, default=None):
                return {"X-Sample-Rate": "16000",
                        "X-Audio-Encoding": "s16be",
                        "Content-Type": "audio/L16;rate=16000;channels=1"}.get(name, default)
            def read1(self, count):
                return self.chunks.pop(0)
        class FakeConnection:
            def __init__(self, *args, **kwargs): pass
            def request(self, method, path, body, headers):
                requests.append((method, path, headers))
            def getresponse(self): return FakeResponse()
        class FakeWave:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def setnchannels(self, count): assert count == 1
            def setsampwidth(self, width): assert width == 2
            def setframerate(self, rate): assert rate == 16000
            def writeframes(self, data): output.append(data)
        with mock.patch.object(sys, "argv", ["stream_client.py", "--l16", "-o", "fictional.wav"]), \
             mock.patch.object(module.http.client, "HTTPConnection", FakeConnection), \
             mock.patch.object(module.wave, "open", return_value=FakeWave()):
            module.main()
        self.assertEqual(requests[0][2]["Accept"], "audio/L16")
        self.assertEqual(output, [b"\x02\x01\xfe\xff"])

    def test_no_hosted_workflow_is_introduced(self):
        workflow_dir = ROOT / ".github" / "workflows"
        self.assertFalse(workflow_dir.exists() and list(workflow_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
