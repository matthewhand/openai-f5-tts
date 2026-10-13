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


if __name__ == "__main__":
    unittest.main()
