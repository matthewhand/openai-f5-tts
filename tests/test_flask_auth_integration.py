"""Real Flask test-client authentication and routing, with inert model/gevent imports.

No socket, GPU, checkpoint, Whisper, ffmpeg or remote request. If Flask is not
already installed, this module skips rather than installing dependencies.
"""
import importlib.util
import logging
import os
from pathlib import Path
import base64
import tempfile
import sys
import types
import unittest
from unittest import mock

try:
    from flask import Flask
except ImportError:  # pragma: no cover - no installs in source-only review
    Flask = None


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SECRET = "synthetic-f5-auth-secret-not-a-live-key"


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(Flask is None, "Flask is not preinstalled; do not install to run offline tests")
class RealFlaskAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_gevent = types.ModuleType("gevent")
        fake_gevent.__path__ = []
        fake_pywsgi = types.ModuleType("gevent.pywsgi")
        fake_pywsgi.WSGIServer = object
        fake_pywsgi.WSGIHandler = object

        class FakeTTSHandler:
            def __init__(self, **kwargs):
                self.calls = []
                self.loaded_models = {"Emilia": object()}
            def list_available_models(self):
                return [{"name": "Emilia", "status": "loaded"}]
            def generate_speech(self, **kwargs):
                self.calls.append(kwargs)
                # Use precisely the exception class captured by the real server.
                # Transient sys.modules patching can otherwise create a second
                # class identity when this suite runs alone.
                raise self.busy_error_type("synthetic busy; no inference")

        fake_tts = types.ModuleType("tts_handler")
        fake_tts.TTSHandler = FakeTTSHandler
        fake_alignment = types.ModuleType("alignment")
        fake_alignment.build_alignment = lambda *args: (None, [], "unavailable")

        cls.env = mock.patch.dict(os.environ, {
            "API_KEY": SECRET, "REQUIRE_API_KEY": "True",
            "DEBUG_MODE": "false", "DEFAULT_VOICE": "Emilia",
            "MAX_REQUEST_BYTES": "65536", "MAX_TEXT_CHARS": "2000",
        })
        cls.env.start()
        cls.addClassCleanup(cls.env.stop)
        cls.path_patch = mock.patch.object(sys, "path", [str(APP)] + list(sys.path))
        cls.path_patch.start()
        cls.addClassCleanup(cls.path_patch.stop)

        with mock.patch.dict(sys.modules, {
            "gevent": fake_gevent, "gevent.pywsgi": fake_pywsgi,
            "tts_handler": fake_tts, "alignment": fake_alignment,
        }):
            cls.utils = import_source("f5_auth_real_utils", APP / "utils.py")
            with mock.patch.dict(sys.modules, {"utils": cls.utils}):
                with mock.patch.object(sys, "argv", ["mock-f5-server"]):
                    cls.server = import_source("f5_mock_backed_server", APP / "server.py")

        cls.client = cls.server.app.test_client()
        cls.fake_handler = cls.server.tts_handler
        cls.fake_handler.busy_error_type = cls.server.SynthesisBusyError

    def setUp(self):
        self.fake_handler.calls.clear()

    def test_every_published_route_requires_auth(self):
        endpoints = [
            ("get", "/v1/models", None),
            ("get", "/v1/voices", None),
            ("get", "/v1/voices/all", None),
            ("get", "/v1/loaded_models", None),
            ("post", "/v1/audio/speech", {"input": "synthetic"}),
            ("post", "/v1/text-to-speech/Emilia/with-timestamps", {"text": "synthetic"}),
        ]
        for method, endpoint, body in endpoints:
            with self.subTest(route=endpoint):
                response = getattr(self.client, method)(endpoint, json=body)
                self.assertEqual(response.status_code, 401)
                self.assertNotIn(SECRET, response.get_data(as_text=True))
        self.assertEqual(self.fake_handler.calls, [])

    def test_valid_bearer_and_xi_api_key_are_accepted(self):
        for headers in (
            {"Authorization": "Bearer " + SECRET},
            {"xi-api-key": SECRET},
        ):
            with self.subTest(headers=list(headers)):
                response = self.client.get("/v1/models", headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["models"][0]["name"], "Emilia")

    def test_incorrect_and_non_ascii_keys_return_401_not_500(self):
        for headers in (
            {"Authorization": "Bearer incorrect"},
            {"xi-api-key": "café"},
            {"Authorization": "Bearer café"},
            {"Authorization": "Basic " + SECRET},
            {"Authorization": "Bearer incorrect", "xi-api-key": SECRET},
        ):
            with self.subTest(headers=list(headers)):
                response = self.client.get("/v1/models", headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertNotIn(SECRET, response.get_data(as_text=True))

    def test_post_routes_with_valid_auth_return_mock_busy_503(self):
        routes = [
            ("/v1/audio/speech", {"input": "fictional text"}),
            ("/v1/text-to-speech/Emilia/with-timestamps", {"text": "fictional text"}),
        ]
        for route, payload in routes:
            with self.subTest(route=route):
                response = self.client.post(route, json=payload,
                       headers={"Authorization": "Bearer " + SECRET})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.get_json()["error"], "Speech generator busy; try again later")
        self.assertEqual(len(self.fake_handler.calls), 2)

    def test_valid_auth_bad_json_never_enters_handler(self):
        response = self.client.post("/v1/audio/speech", json={"input": []},
                         headers={"xi-api-key": SECRET})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.fake_handler.calls, [])

    def test_max_request_bytes_return_413_without_inference(self):
        response = self.client.post("/v1/audio/speech",
            data='{"input":"' + ("x" * 70000) + '"}',
            headers={"xi-api-key": SECRET, "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.fake_handler.calls, [])

    def test_raw_audio_oversize_returns_503_and_deletes_request_file(self):
        with tempfile.TemporaryDirectory(prefix="f5-mock-output-") as tmp:
            path = Path(tmp) / "synthetic.mp3"
            path.write_bytes(b"x" * 18)
            with mock.patch.object(self.fake_handler, "generate_speech", return_value=str(path)), \
                 mock.patch.object(self.server, "MAX_AUDIO_OUTPUT_BYTES", 8):
                response = self.client.post("/v1/audio/speech", json={"input": "fictional"},
                    headers={"Authorization": "Bearer " + SECRET})
            self.assertEqual(response.status_code, 503)
            self.assertFalse(path.exists())
            self.assertEqual(response.get_json()["error"], "Generated audio exceeds configured output limit")

    def test_timestamp_audio_oversize_returns_503_and_deletes_request_file(self):
        with tempfile.TemporaryDirectory(prefix="f5-mock-output-") as tmp:
            path = Path(tmp) / "synthetic.mp3"
            path.write_bytes(b"x" * 18)
            with mock.patch.object(self.fake_handler, "generate_speech", return_value=str(path)), \
                 mock.patch.object(self.server, "MAX_AUDIO_OUTPUT_BYTES", 8):
                response = self.client.post("/v1/text-to-speech/Emilia/with-timestamps",
                    json={"text": "fictional"}, headers={"xi-api-key": SECRET})
            self.assertEqual(response.status_code, 503)
            self.assertFalse(path.exists())

    def test_bounded_audio_and_timestamp_success_with_fake_file(self):
        for route, payload in (
            ("/v1/audio/speech", {"input": "fictional"}),
            ("/v1/text-to-speech/Emilia/with-timestamps", {"text": "fictional"}),
        ):
            with self.subTest(route=route), tempfile.TemporaryDirectory(prefix="f5-mock-") as tmp:
                path = Path(tmp) / "synthetic.mp3"
                path.write_bytes(b"12345678")
                with mock.patch.object(self.fake_handler, "generate_speech", return_value=str(path)), \
                     mock.patch.object(self.server, "MAX_AUDIO_OUTPUT_BYTES", 8):
                    response = self.client.post(route, json=payload,
                            headers={"Authorization": "Bearer " + SECRET})
                self.assertEqual(response.status_code, 200)
                self.assertFalse(path.exists())
                if route.endswith("with-timestamps"):
                    self.assertEqual(base64.b64decode(response.get_json()["audio_base64"]), b"12345678")
                else:
                    self.assertEqual(response.data, b"12345678")

    def test_disabled_auth_setting_is_explicit_and_visible(self):
        with mock.patch.object(self.utils, "REQUIRE_API_KEY", False):
            response = self.client.get("/v1/models")
            self.assertEqual(response.status_code, 200)


    def test_malformed_auth_setting_rejected_during_configuration_import(self):
        for value in ("tru", "enabled", " ", "2"):
            with self.subTest(value=value), mock.patch.dict(
                    os.environ, {"API_KEY": SECRET, "REQUIRE_API_KEY": value}):
                with self.assertRaisesRegex(ValueError, "Invalid boolean configuration"):
                    import_source("f5_rejected_auth_utils", APP / "utils.py")

    def test_backend_errors_are_sanitized_for_speech_and_timestamps(self):
        sentinel = "SYNTHETIC_SECRET_PATH_TEXT_6789"
        for error in (ValueError(sentinel), RuntimeError(sentinel)):
            for url, payload in (
                    ("/v1/audio/speech", {"input": "hello"}),
                    ("/v1/text-to-speech/Emilia/with-timestamps", {"input": "hello"})):
                with self.subTest(url=url, error=type(error).__name__), \
                     mock.patch.object(self.fake_handler, "generate_speech", side_effect=error), \
                     self.assertLogs(level="ERROR") as logs:
                    response = self.client.post(url, json=payload,
                                                headers={"Authorization": "Bearer " + SECRET})
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.get_json(), {"error": "Failed to generate speech"})
                self.assertNotIn(sentinel, response.get_data(as_text=True))
                self.assertNotIn(sentinel, " ".join(logs.output))

    def test_stream_backend_error_is_sanitized_before_headers(self):
        sentinel = "SYNTHETIC_SECRET_PATH_STREAM_6789"
        def fail_stream(*args, **kwargs):
            raise ValueError(sentinel)
            yield b""  # must fail during first iteration
        with mock.patch.object(self.fake_handler, "available_models", {"Emilia": True}, create=True), \
             mock.patch.object(self.fake_handler, "generate_speech_stream", fail_stream, create=True), \
             self.assertLogs(level="ERROR") as logs:
            response = self.client.post("/v1/audio/speech/stream", json={"input": "hello"},
                                        headers={"Authorization": "Bearer " + SECRET})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(sentinel, response.get_data(as_text=True))
        self.assertNotIn(sentinel, " ".join(logs.output))


    def test_debug_request_logging_excludes_voice_path_text_and_credentials(self):
        voice = "PRIVATE_VOICE_IN_PATH_SENTINEL_14"
        payload = "PRIVATE_REQUEST_TEXT_SENTINEL_26"
        with mock.patch.object(self.server, "DEBUG_MODE", True), \
             self.assertLogs(level="DEBUG") as logs:
            response = self.client.post(
                "/v1/text-to-speech/" + voice + "/with-timestamps",
                json={"text": payload},
                headers={"Authorization": "Bearer " + SECRET},
            )
        self.assertIn(response.status_code, (400, 503))
        combined = " ".join(logs.output)
        for marker in (voice, payload, SECRET):
            self.assertNotIn(marker, combined)
        self.assertIn("HTTP request: method=POST endpoint=text_to_speech_with_timestamps",
                      combined)

    def _import_with_bind(self, host):
        """Import the actual server with fake model/gevent and auth disabled."""
        created = []
        class StartupSpy:
            def __init__(self, **kwargs):
                created.append("constructed")
                self.loaded_models = {}
            def list_available_models(self):
                return []
        fake_tts = types.ModuleType("tts_handler")
        fake_tts.TTSHandler = StartupSpy
        fake_gevent = types.ModuleType("gevent")
        fake_gevent.__path__ = []
        fake_pywsgi = types.ModuleType("gevent.pywsgi")
        fake_pywsgi.WSGIServer = object
        fake_pywsgi.WSGIHandler = object
        fake_alignment = types.ModuleType("alignment")
        fake_alignment.build_alignment = lambda *args: (None, [], "unavailable")
        with mock.patch.dict(os.environ, {"HOST": host, "REQUIRE_API_KEY": "False"}), \
             mock.patch.object(self.utils, "REQUIRE_API_KEY", False), \
             mock.patch.dict(sys.modules, {
                 "gevent": fake_gevent, "gevent.pywsgi": fake_pywsgi,
                 "tts_handler": fake_tts, "alignment": fake_alignment,
                 "utils": self.utils,
             }), \
             mock.patch.object(sys, "argv", ["f5-fake-server"]):
            try:
                server = import_source("f5_unauth_bind_probe", APP / "server.py")
            except ValueError:
                self.assertEqual(created, [], "Unsafe bind reached model initialization")
                raise
        return server, created

    def test_unauthenticated_wildcard_and_lan_rejected_before_model_startup(self):
        for host in ("0.0.0.0", "::", "192.168.1.25", "localhost"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                self._import_with_bind(host)

    def test_unauthenticated_loopback_import_and_real_flask_route(self):
        server, created = self._import_with_bind("127.0.0.1")
        self.assertEqual(created, ["constructed"])
        # The fixture temporarily patches an already imported utils module.
        # Keep that synthetic auth state active across the client request.
        with mock.patch.object(self.utils, "REQUIRE_API_KEY", False):
            response = server.app.test_client().get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"models": []})


if __name__ == "__main__":
    unittest.main()
