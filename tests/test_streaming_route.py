"""Route-level contract for POST /v1/audio/speech/stream.

No GPU / model needed: TTSHandler is replaced by a fake that records when each
"sentence" is synthesised, so we can prove the response is really incremental
(chunk 1 is delivered before sentence 2 is synthesised) and that closing the
connection stops further synthesis (barge-in).
"""
import importlib
import sys
from pathlib import Path

import pytest

APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
KEY = "test-key-123"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeTTS:
    """Stands in for tts_handler.TTSHandler."""

    instance = None

    def __init__(self, *args, **kwargs):
        self.available_models = {"Scarlett": "x", "Emilia": "y"}
        self.loaded_models = {}
        self.events = []
        self.fail_on_sentence = None
        self.last_speed = None
        FakeTTS.instance = self

    def list_available_models(self):
        return [{"name": n, "status": "unloaded"} for n in self.available_models]

    def generate_speech_stream(self, text, voice="Scarlett", speed=1.0):
        self.last_speed = speed
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        try:
            for i, _ in enumerate(sentences):
                if self.fail_on_sentence == i:
                    raise RuntimeError("boom")
                self.events.append(f"synth:{i}")
                yield (b"\x01\x00" * 160)  # 160 int16 samples
        finally:
            self.events.append("closed")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.syspath_prepend(APP_DIR)
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setenv("REQUIRE_API_KEY", "True")
    monkeypatch.setattr(sys, "argv", ["server.py"])
    for name in ("server", "utils", "tts_handler"):
        sys.modules.pop(name, None)
    fake_mod = type(sys)("tts_handler")
    fake_mod.TTSHandler = FakeTTS
    monkeypatch.setitem(sys.modules, "tts_handler", fake_mod)
    server = importlib.import_module("server")
    server.app.testing = True
    yield server.app.test_client()
    for name in ("server", "utils"):
        sys.modules.pop(name, None)


URL = "/v1/audio/speech/stream"


def test_requires_auth(client):
    assert client.post(URL, json={"input": "Hi."}).status_code == 401


def test_rejects_missing_or_empty_input(client):
    assert client.post(URL, json={}, headers=AUTH).status_code == 400
    assert client.post(URL, json={"input": ""}, headers=AUTH).status_code == 400
    assert client.post(URL, json={"input": "   "}, headers=AUTH).status_code == 400
    assert client.post(URL, json={"input": 123}, headers=AUTH).status_code == 400


def test_rejects_non_json_body(client):
    r = client.post(URL, data="not json", headers=AUTH)
    assert r.status_code in (400, 415)


def test_rejects_bad_speed(client):
    for bad in ("fast", 0, -1, 99):
        r = client.post(URL, json={"input": "Hi.", "speed": bad}, headers=AUTH)
        assert r.status_code == 400, f"speed={bad!r} should be 400, got {r.status_code}"


def test_unknown_voice_is_http_400_not_empty_200(client):
    """Errors must surface as a status code *before* streaming starts."""
    r = client.post(URL, json={"input": "Hi.", "voice": "Nope"}, headers=AUTH)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_headers_describe_the_pcm(client):
    r = client.post(URL, json={"input": "One. Two."}, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("audio/L16")
    assert "rate=16000" in r.headers["Content-Type"]
    assert r.headers["X-Sample-Rate"] == "16000"
    assert r.headers["X-Audio-Channels"] == "1"
    assert r.headers["X-Audio-Encoding"] == "int16"


def test_body_is_concatenated_chunks(client):
    r = client.post(URL, json={"input": "One. Two. Three."}, headers=AUTH)
    assert len(r.data) == 3 * 320


def test_speed_is_forwarded(client):
    client.post(URL, json={"input": "Hi.", "speed": 1.25}, headers=AUTH)
    assert FakeTTS.instance.last_speed == pytest.approx(1.25)


def test_is_incremental_chunk1_before_sentence2_is_synthesised(client):
    r = client.post(URL, json={"input": "One. Two. Three."}, headers=AUTH, buffered=False)
    it = iter(r.response)
    first = next(it)
    assert len(first) == 320
    assert FakeTTS.instance.events == ["synth:0"], "sentence 2 ran before chunk 1 was consumed"
    second = next(it)
    assert len(second) == 320
    assert FakeTTS.instance.events == ["synth:0", "synth:1"]
    r.close()


def test_closing_connection_stops_synthesis(client):
    """Barge-in: client hangs up after first chunk => no more sentences, generator closed."""
    r = client.post(URL, json={"input": "One. Two. Three. Four."}, headers=AUTH, buffered=False)
    it = iter(r.response)
    next(it)
    r.close()
    ev = FakeTTS.instance.events
    assert ev == ["synth:0", "closed"], ev


def test_midstream_failure_does_not_crash_and_keeps_earlier_audio(client):
    client.application  # ensure imported
    FakeTTS.instance.fail_on_sentence = 1
    r = client.post(URL, json={"input": "One. Two. Three."}, headers=AUTH)
    assert r.status_code == 200
    assert len(r.data) == 320  # sentence 0 delivered, stream ended cleanly
