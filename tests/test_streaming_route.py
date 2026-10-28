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
        self.available_models = {"V0": "x", "Emilia": "y"}
        self.loaded_models = {}
        self.events = []
        self.fail_on_sentence = None
        self.last_speed = None
        FakeTTS.instance = self

    def list_available_models(self):
        return [{"name": n, "status": "unloaded"} for n in self.available_models]

    def generate_speech_stream(self, text, voice="V0", speed=1.0):
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
    # Route tests are pure Flask tests; never install or load real Gevent.
    fake_gevent = type(sys)("gevent")
    fake_gevent.__path__ = []
    fake_pywsgi = type(sys)("gevent.pywsgi")
    fake_pywsgi.WSGIServer = object
    fake_pywsgi.WSGIHandler = object
    monkeypatch.setitem(sys.modules, "gevent", fake_gevent)
    monkeypatch.setitem(sys.modules, "gevent.pywsgi", fake_pywsgi)
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


def test_stream_rejects_shared_validation_failures_before_generator(client):
    for invalid in ({"input": "a" * 2001}, {"input": "Hi.", "voice": []},
                    {"input": "Hi.", "speed": True},
                    {"input": "Hi.", "speed": float("nan")},
                    {"input": "Hi.", "response_format": "wav"},
                    {"input": "Hi.", "output_format": "mp3"}):
        r = client.post(URL, json=invalid, headers=AUTH)
        assert r.status_code == 400, invalid
        assert FakeTTS.instance.events == []


def test_stream_preheader_backend_failure_is_sanitized(client, caplog):
    sentinel = "SYNTHETIC_PRIVATE_PATH_AND_TEXT_ABC123"
    def fail_generator(text, voice, speed):
        raise ValueError(sentinel)
        yield b""  # generator must fail during first next(), not construction
    FakeTTS.instance.generate_speech_stream = fail_generator
    with caplog.at_level("ERROR"):
        response = client.post(URL, json={"input": "Hi."}, headers=AUTH)
    assert response.status_code == 500
    assert sentinel not in response.get_data(as_text=True)
    assert sentinel not in caplog.text


def test_stream_busy_is_503_before_sending_audio(client):
    from generation_gate import SynthesisBusyError
    def busy_generator(text, voice, speed):
        raise SynthesisBusyError("Synthetic competing generation")
        yield b""
    FakeTTS.instance.generate_speech_stream = busy_generator
    r = client.post(URL, json={"input": "Hi."}, headers=AUTH)
    assert r.status_code == 503
    assert r.get_json() == {"error": "Speech generator busy; try again later"}


def test_prefetched_response_close_before_iteration_releases_gate(client, monkeypatch):
    """The WSGI iterable can be closed before body() gets its first next()."""
    import sys
    from generation_gate import GenerationGate, SynthesisBusyError

    gate = GenerationGate()
    events = []
    retained = []  # Keep a second owner: refcount cleanup must not mask the bug.

    def guarded_stream(text, voice, speed):
        def generate():
            with gate.claim():
                events.append("started")
                try:
                    yield b"\x01\x00"
                    yield b"\x02\x00"
                finally:
                    events.append("closed")
        stream = generate()
        retained.append(stream)
        return stream

    monkeypatch.setattr(FakeTTS.instance, "generate_speech_stream", guarded_stream)
    server = sys.modules["server"]
    with server.app.test_request_context(URL, method="POST", headers=AUTH,
                                         json={"input": "One. Two."}):
        response = server.text_to_speech_stream()
        assert response.status_code == 200
        assert events == ["started"]
        with pytest.raises(SynthesisBusyError):
            with gate.claim():
                pytest.fail("Second request entered before stream closed")
        response.close()  # Important: never iterate response.response.
        assert events == ["started", "closed"]
        with gate.claim():
            pass
        with gate.claim():
            response.close()  # Must not release a new owner's lease.
            with pytest.raises(SynthesisBusyError):
                with gate.claim():
                    pytest.fail("Double close released another owner's slot")


def test_prefetched_wsgi_app_iter_close_releases_gate(client, monkeypatch):
    """Close via actual WSGI iterable, not just Flask Response.close()."""
    import sys
    from generation_gate import GenerationGate

    gate = GenerationGate()
    events = []
    retained = []  # Keep a second owner: refcount cleanup must not mask the bug.

    def guarded_stream(text, voice, speed):
        def generate():
            with gate.claim():
                events.append("started")
                try:
                    yield b"\x01\x00"
                    yield b"\x02\x00"
                finally:
                    events.append("closed")
        stream = generate()
        retained.append(stream)
        return stream

    monkeypatch.setattr(FakeTTS.instance, "generate_speech_stream", guarded_stream)
    server = sys.modules["server"]
    with server.app.test_request_context(URL, method="POST", headers=AUTH,
                                         json={"input": "One. Two."}):
        response = server.text_to_speech_stream()
        app_iter = response.get_app_iter({"REQUEST_METHOD": "POST"})
        assert events == ["started"]
        app_iter.close()
        assert events == ["started", "closed"]
        with gate.claim():
            pass


@pytest.mark.parametrize("termination", ("normal", "failure", "disconnect"))
def test_gated_stream_releases_after_delivery_failure_or_disconnect(
        client, monkeypatch, caplog, termination):
    """WSGI delivery owns the lease through the last chunk or client hangup."""
    from generation_gate import GenerationGate, SynthesisBusyError

    gate = GenerationGate()
    events = []
    sentinel = "SYNTHETIC_PRIVATE_INFERENCE_ERROR_NOT_FOR_LOGS"

    def synth(text, voice, speed):
        with gate.claim():
            events.append("claimed")
            try:
                yield b"\x01\x00"
                if termination == "failure":
                    raise RuntimeError(sentinel)
                events.append("second")
                yield b"\x02\x00"
            finally:
                events.append("released")

    monkeypatch.setattr(FakeTTS.instance, "generate_speech_stream", synth)
    with caplog.at_level("ERROR"):
        response = client.post(URL, json={"input": "One. Two."}, headers=AUTH,
                               buffered=False)
        assert response.status_code == 200
        assert events == ["claimed"]
        with pytest.raises(SynthesisBusyError):
            with gate.claim():
                pytest.fail("Concurrent generation admitted during stream")
        if termination == "disconnect":
            response.close()
            assert events == ["claimed", "released"]
        elif termination == "normal":
            assert response.data == b"\x01\x00\x02\x00"
            response.close()
            assert events == ["claimed", "second", "released"]
        else:
            assert response.data == b"\x01\x00"
            response.close()
            assert events == ["claimed", "released"]
    assert sentinel not in caplog.text
    with gate.claim():
        response.close()  # Re-closing must not free the *new* owner's lease.
        with pytest.raises(SynthesisBusyError):
            with gate.claim():
                pytest.fail("Response cleanup released somebody else's lease")


def test_nonstreaming_raw_pcm_mime_is_truthful(client, monkeypatch):
    import tempfile
    generated = []
    def speech(**kwargs):
        file = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
        with file:
            file.write(b"\x02\x01\xfe\xff")
        generated.append(Path(file.name))
        return file.name
    monkeypatch.setattr(FakeTTS.instance, "generate_speech", speech, raising=False)
    try:
        response = client.post("/v1/audio/speech", json={"input": "one", "response_format": "pcm"},
                               headers=AUTH)
        assert response.status_code == 200
        assert response.mimetype == "application/octet-stream"
        assert response.headers["X-Audio-Encoding"] == "s16le"
        assert response.data == b"\x02\x01\xfe\xff"
        assert all(not path.exists() for path in generated)
    finally:
        for path in generated:
            path.unlink(missing_ok=True)


def test_timestamp_strict_profile_is_forwarded_and_legacy_is_unchanged(client, monkeypatch):
    import sys
    import tempfile
    import base64
    captured, generated = [], []
    server = sys.modules["server"]
    monkeypatch.setattr(server, "build_alignment", lambda *args: (None, [], "unavailable"))
    def speech(**kwargs):
        captured.append(kwargs)
        file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with file:
            file.write(b"fictional-audio")
        generated.append(Path(file.name))
        return file.name
    monkeypatch.setattr(FakeTTS.instance, "generate_speech", speech, raising=False)
    try:
        for profile, strict in (("mp3_44100_128_strict", True),
                                ("mp3_44100_128", False)):
            response = client.post(
                "/v1/text-to-speech/Emilia/with-timestamps?output_format=" + profile,
                json={"text": "fictional speech"}, headers=AUTH)
            assert response.status_code == 200
            body = response.get_json()
            assert base64.b64decode(body["audio_base64"]) == b"fictional-audio"
            assert captured[-1]["response_format"] == "mp3"
            if strict:
                assert captured[-1]["encoding_profile"] == profile
                assert body["encoding_profile"] == profile
            else:
                assert "encoding_profile" not in captured[-1]
                assert "encoding_profile" not in body
        assert all(not path.exists() for path in generated)
    finally:
        for path in generated:
            path.unlink(missing_ok=True)


def test_headers_describe_the_pcm(client):
    r = client.post(URL, json={"input": "One. Two."}, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("application/octet-stream")
    assert r.headers["X-Sample-Rate"] == "16000"
    assert r.headers["X-Audio-Channels"] == "1"
    assert r.headers["X-Audio-Encoding"] == "s16le"


@pytest.mark.parametrize("accept,l16", [
    ("audio/L16", True),
    ("application/json, audio/L16; rate=16000; channels=1; q=0.8", True),
    ("*/*", False),
    ("audio/*", False),
    ("audio/L16;q=0", False),
    ("audio/L16;rate=8000", False),
])
def test_explicit_l16_negotation_and_sample_bytes(client, accept, l16):
    response = client.post(URL, json={"input": "One. Two."},
                           headers={**AUTH, "Accept": accept})
    assert response.status_code == 200
    if l16:
        assert response.headers["Content-Type"].startswith("audio/L16;")
        assert response.headers["X-Audio-Encoding"] == "s16be"
        assert response.data[:4] == b"\x00\x01\x00\x01"
    else:
        assert response.headers["Content-Type"].startswith("application/octet-stream")
        assert response.headers["X-Audio-Encoding"] == "s16le"
        assert response.data[:4] == b"\x01\x00\x01\x00"
    assert len(response.data) == 640


def test_l16_odd_width_rejected_before_headers(client):
    def malformed(text, voice, speed):
        yield b"\x01"
    FakeTTS.instance.generate_speech_stream = malformed
    response = client.post(URL, json={"input": "One."},
                           headers={**AUTH, "Accept": "audio/L16"})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid audio sample width"}


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


def test_hangup_is_detected_before_next_sentence_even_without_a_failed_write(client):
    """TCP accepts the first write after the peer's FIN, so waiting for a write error
    wastes a whole extra sentence of GPU time. The route must peek at the client
    socket (exposed by the gevent handler as environ['gevent.socket']) before each
    sentence instead."""
    import socket
    server_side, client_side = socket.socketpair()
    try:
        r = client.post(URL, json={"input": "One. Two. Three. Four."}, headers=AUTH,
                        buffered=False, environ_overrides={"gevent.socket": server_side})
        it = iter(r.response)
        assert len(next(it)) == 320
        client_side.close()  # barge-in: client goes away while sentence 2 would be next
        assert list(it) == [], "kept synthesising after the client hung up"
        assert FakeTTS.instance.events == ["synth:0", "closed"]
    finally:
        server_side.close()


def test_open_connection_keeps_streaming(client):
    import socket
    server_side, client_side = socket.socketpair()
    try:
        r = client.post(URL, json={"input": "One. Two. Three."}, headers=AUTH,
                        buffered=False, environ_overrides={"gevent.socket": server_side})
        assert [len(c) for c in r.response] == [320, 320, 320]
    finally:
        server_side.close()
        client_side.close()


def test_midstream_failure_does_not_crash_and_keeps_earlier_audio(client):
    FakeTTS.instance.fail_on_sentence = 1
    r = client.post(URL, json={"input": "One. Two. Three."}, headers=AUTH)
    assert r.status_code == 200
    assert len(r.data) == 320  # sentence 0 delivered, stream ended cleanly
