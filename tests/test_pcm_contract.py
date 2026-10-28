"""Issue #18: synthetic PCM wire-contract regressions, no GPU, ports, or real models."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
from pcm_contract import explicit_l16_accepted, pcm_le_to_l16


@pytest.mark.parametrize("value,expected", [
    ("audio/L16", True),
    ("audio/L16; rate=16000; channels=1; q=0.8", True),
    ("audio/mpeg, audio/L16;q=1", True),
    ("*/*", False), ("audio/*", False),
    ("audio/L16;q=0", False),
    ("audio/L16;rate=8000", False),
    ("audio/L16;channels=2", False),
    ("audio/L16;other=value", False),
    ("audio/L16;q=garbage", False),
])
def test_explicit_l16_accept_only(value, expected):
    assert explicit_l16_accepted(value) is expected


def test_actual_stream_encoder_emits_little_endian_on_any_host():
    """Run real producer method via AST with fictional in-memory inference."""
    import ast
    import logging
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    tree = ast.parse((APP / "tts_handler.py").read_text())
    handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TTSHandler")
    method = next(n for n in handler.body if isinstance(n, ast.FunctionDef) and n.name == "generate_speech_stream")
    method.decorator_list = []
    namespace = {
        "np": np,
        "torch": SimpleNamespace(Tensor=type("SyntheticTorchNever", (), {})),
        "split_sentences": lambda text: [text],
        "infer_process": lambda *args, **kwargs: (np.array([1.0, -1.0, 0.0]), 16000, None),
        "logging": logging,
    }
    ast.fix_missing_locations(method)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "tts_handler.py", "exec"), namespace)
    owner = SimpleNamespace(
        available_models={"Emilia": object()},
        load_voice_model=lambda voice: object(),
        transcribe_audio=lambda voice: "fictional transcription",
        process_reference_audio=lambda voice: "fictional-path",
        vocoder=object(),
    )
    owner._generation_gate = SimpleNamespace(claim=lambda: nullcontext())
    chunks = namespace["generate_speech_stream"](owner, "Hello.", "Emilia", 1.0)
    try:
        assert next(chunks) == bytes.fromhex("ff7f 0180 0000")
        with pytest.raises(StopIteration): next(chunks)
    finally:
        chunks.close()


def test_network_order_byte_samples_preserve_signed_values():
    # 0x0102, -2 and extrema, each serialized as signed PCM16 little endian.
    little = bytes.fromhex("0201 feff 0080 ff7f 0000")
    big = bytes.fromhex("0102 fffe 8000 7fff 0000")
    assert pcm_le_to_l16(little) == big
    assert pcm_le_to_l16(big) == little
    assert pcm_le_to_l16(b"") == b""
    with pytest.raises(ValueError, match="sample width"):
        pcm_le_to_l16(b"\x42")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    monkeypatch.setenv("API_KEY", "synthetic-123")
    monkeypatch.setenv("REQUIRE_API_KEY", "True")
    monkeypatch.setattr(sys, "argv", ["server.py"])
    for name in ("server", "utils", "tts_handler"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    gevent = ModuleType("gevent")
    gevent.__path__ = []
    pywsgi = ModuleType("gevent.pywsgi")
    pywsgi.WSGIServer = object
    pywsgi.WSGIHandler = object
    monkeypatch.setitem(sys.modules, "gevent", gevent)
    monkeypatch.setitem(sys.modules, "gevent.pywsgi", pywsgi)
    class TTS:
        available_models = {"Emilia": "test"}
        def __init__(self, *args, **kwargs): pass
        def generate_speech_stream(self, *args, **kwargs):
            yield bytes.fromhex("0201 feff")
            yield bytes.fromhex("0080 ff7f")
    fake = ModuleType("tts_handler")
    fake.TTSHandler = TTS
    monkeypatch.setitem(sys.modules, "tts_handler", fake)
    server = importlib.import_module("server")
    server.app.testing = True
    yield server, server.app.test_client()
    monkeypatch.delitem(sys.modules, "server", raising=False)
    monkeypatch.delitem(sys.modules, "utils", raising=False)


@pytest.mark.parametrize("accept,be", [
    (None, False), ("*/*", False), ("audio/L16;q=0", False),
    ("audio/L16", True), ("audio/L16;rate=16000;channels=1", True),
])
def test_stream_headers_match_exact_bytes(client, accept, be):
    _, c = client
    headers = {"Authorization": "Bearer synthetic-123"}
    if accept is not None:
        headers["Accept"] = accept
    response = c.post("/v1/audio/speech/stream", json={"input": "One. Two."}, headers=headers)
    assert response.status_code == 200
    assert response.headers["X-Sample-Rate"] == "16000"
    assert response.headers["X-Audio-Channels"] == "1"
    assert response.headers["X-Audio-Encoding"] == ("s16be" if be else "s16le")
    assert response.mimetype.lower() == ("audio/l16" if be else "application/octet-stream")
    assert response.data == (bytes.fromhex("0102 fffe 8000 7fff") if be
                             else bytes.fromhex("0201 feff 0080 ff7f"))
    response.close()


def test_invalid_first_l16_chunk_closes_upstream_generator(client, monkeypatch):
    server, c = client
    closed = []
    def malformed(*args, **kwargs):
        try:
            yield b"\x01"  # one byte: cannot be PCM16
        finally:
            closed.append(True)
    monkeypatch.setattr(server.tts_handler, "generate_speech_stream", malformed)
    response = c.post("/v1/audio/speech/stream", json={"input": "Hi."},
                      headers={"Authorization": "Bearer synthetic-123", "Accept": "audio/L16"})
    assert response.status_code == 400
    assert closed == [True]


def test_nonstreaming_raw_pcm_headers_and_cleanup(client, monkeypatch, tmp_path):
    server, c = client
    target = tmp_path / "fictional.pcm"
    target.write_bytes(bytes.fromhex("0201 fffe"))
    monkeypatch.setattr(server.tts_handler, "generate_speech", lambda **kwargs: str(target), raising=False)
    response = c.post("/v1/audio/speech", json={"input": "fiction", "response_format": "pcm"},
                      headers={"Authorization": "Bearer synthetic-123"})
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    assert response.headers["X-Audio-Encoding"] == "s16le"
    assert response.data == bytes.fromhex("0201 fffe")
    assert not target.exists()


def test_client_opt_in_converts_l16_to_wav_le(monkeypatch):
    spec = importlib.util.spec_from_file_location("pcm_client_fixture", ROOT / "scripts/stream_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outgoing, captured = [], []
    class Response:
        status = 200
        def __init__(self): self.chunks = [bytes.fromhex("0102 fffe"), b""]
        def getheader(self, name, default=None):
            return {"Content-Type": "audio/L16;rate=16000;channels=1",
                    "X-Sample-Rate": "16000", "X-Audio-Encoding": "s16be"}.get(name, default)
        def read1(self, n): return self.chunks.pop(0)
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, body, headers): outgoing.append((method, path, headers))
        def getresponse(self): return Response()
    class WaveWriter:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def setnchannels(self, value): assert value == 1
        def setsampwidth(self, value): assert value == 2
        def setframerate(self, value): assert value == 16000
        def writeframes(self, data): captured.append(data)
    with mock.patch.object(sys, "argv", ["stream_client.py", "--l16", "-o", "fiction.wav"]), \
         mock.patch.object(module.http.client, "HTTPConnection", Connection), \
         mock.patch.object(module.wave, "open", return_value=WaveWriter()):
        module.main()
    assert outgoing[0][2]["Accept"] == "audio/L16"
    assert captured == [bytes.fromhex("0201 feff")]
