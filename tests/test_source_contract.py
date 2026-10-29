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

    def test_readme_authenticated_route_inventory_tracks_server(self):
        """A newly added API route must update the documented auth count."""
        server = ast.parse(self.read("app/server.py"))
        handlers = []
        for fn in server.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            routes = [dec for dec in fn.decorator_list
                      if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                      and isinstance(dec.func.value, ast.Name)
                      and dec.func.value.id == "app" and dec.func.attr == "route"]
            if routes:
                self.assertIn("require_api_key", [dec.id for dec in fn.decorator_list
                                                  if isinstance(dec, ast.Name)])
                handlers.append(fn.name)
        readme = self.read("README.md")
        self.assertIn(f"all {len(handlers)} HTTP routes", readme)
        self.assertIn("/v1/audio/speech/stream`, and `/v1/text-to-speech/", readme)
        self.assertIn("REQUIRE_API_KEY=False", readme)
        self.assertIn("numeric loopback", readme)
        # Each API endpoint has one authoritative section; a duplicate heading
        # silently creates ambiguous markdown navigation links.
        endpoint_headings = [line for line in readme.splitlines()
                             if line.startswith("### `/v1/")]
        self.assertEqual(len(endpoint_headings), len(set(endpoint_headings)))
        self.assertEqual(readme.count("### `/v1/loaded_models`"), 1)

    def test_readme_shared_synthesis_admission_and_stream_cleanup(self):
        readme = self.read("README.md")
        source = self.read("app/server.py")
        handler = self.read("app/tts_handler.py")
        validator = self.read("app/request_validation.py")
        self.assertIn("all three synthesis endpoints return HTTP `503`", readme)
        self.assertIn("All three speech endpoints", readme)
        self.assertIn("MAX_REQUEST_BYTES", readme)
        self.assertIn("MAX_TEXT_CHARS", readme)
        self.assertIn("response.call_on_close(chunks.close)", source)
        self.assertEqual(handler.count("with self._generation_gate.claim():"), 2)
        for name in ("parse_speech_request", "parse_timestamp_request", "parse_stream_request"):
            self.assertIn(name + "(", source)
        self.assertIn("max_text_chars=MAX_TEXT_CHARS", source)
        self.assertIn("def parse_stream_request(", validator)
        self.assertIn("before body iteration", readme)
        self.assertIn("cross-process", readme)

    def test_readme_pcm_headers_and_timestamp_profiles_follow_source(self):
        readme = self.read("README.md")
        server = self.read("app/server.py")
        contract = self.read("app/pcm_contract.py")
        self.assertIn("application/octet-stream", readme)
        self.assertIn("X-Audio-Encoding: s16le", readme)
        self.assertIn("audio/L16; rate=16000; channels=1", readme)
        self.assertIn("X-Audio-Encoding: s16be", readme)
        self.assertIn("Accept: */*", readme)
        self.assertIn("explicit_l16_accepted", server)
        self.assertIn("pcm_le_to_l16", server)
        self.assertIn('media.strip().lower() != "audio/l16"', contract)

        validation = ast.parse(self.read("app/request_validation.py"))
        profiles = None
        for assign in validation.body:
            if isinstance(assign, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "STRICT_TIMESTAMP_PROFILES"
                for target in assign.targets
            ):
                profiles = ast.literal_eval(assign.value)
        self.assertIsNotNone(profiles)
        for name in profiles:
            with self.subTest(profile=name):
                self.assertIn("`" + name + "`", readme)
        for legacy in ("mp3_44100_128", "ogg_48000_128", "wav_44100_128"):
            self.assertIn("`" + legacy + "`", readme)
        self.assertIn("codec selectors only", readme)
        self.assertIn("encoding_profile", readme)

    def test_readme_privacy_scope_and_historical_evidence_remain_distinct(self):
        readme = self.read("README.md")
        server = self.read("app/server.py")
        handler = self.read("app/tts_handler.py")
        for text in ("14 October 2026", "20 October 2026", "28 October 2026",
                     "223 passed, 197 subtests", "zero skips", "loopback",
                     "third-party logger", "reverse-proxy", "no pretrained weights"):
            with self.subTest(text=text):
                self.assertIn(text.lower(), readme.lower())
        self.assertIn("log=None, error_log=None", server)
        self.assertIn("_upstream_infer_module.print = _discard_upstream_print", handler)

    def test_no_hosted_workflow_is_introduced(self):
        workflow_dir = ROOT / ".github" / "workflows"
        self.assertFalse(workflow_dir.exists() and list(workflow_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
