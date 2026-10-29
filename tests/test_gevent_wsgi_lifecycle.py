"""Actual Gevent WSGI/Flask stream lifetime with fictional CPU-only output.

No F5 model, Whisper, external networking, CUDA or Docker. Tests deliberately
bind one ephemeral IPv4 loopback socket and use a bounded client thread.
"""
import http.client
import importlib
import io
import json
import logging
import socket
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

gevent = pytest.importorskip("gevent")

APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
TOKEN = "fictional-test-key"
URL = "/v1/audio/speech/stream"


@pytest.fixture(params=("False", "True"), ids=("production", "debug"))
def native_route(monkeypatch, request):
    """The actual Flask routes and real Gevent handler, but no model startup."""
    monkeypatch.syspath_prepend(APP_DIR)
    monkeypatch.setenv("API_KEY", TOKEN)
    monkeypatch.setenv("REQUIRE_API_KEY", "True")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("DEBUG_MODE", request.param)
    monkeypatch.setattr(sys, "argv", ["server.py", "--host", "127.0.0.1"])
    for name in ("server", "tts_handler", "utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    from generation_gate import GenerationGate

    class FakeTTS:
        def __init__(self, *args, **kwargs):
            self.available_models = {"Emilia": "synthetic"}
            self._generation_gate = GenerationGate()
            self.events = []
            self.start_delay = 0

        def generate_speech_stream(self, text, voice, speed):
            with self._generation_gate.claim():
                self.events.append("claimed")
                try:
                    if self.start_delay:
                        gevent.sleep(self.start_delay)
                    yield b"\x01\x00" * 160
                    self.events.append("second")
                    yield b"\x02\x00" * 160
                finally:
                    self.events.append("released")

    mod = ModuleType("tts_handler")
    mod.TTSHandler = FakeTTS
    monkeypatch.setitem(sys.modules, "tts_handler", mod)
    server = importlib.import_module("server")
    server.app.testing = True
    try:
        yield server
    finally:
        monkeypatch.delitem(sys.modules, "server", raising=False)
        monkeypatch.delitem(sys.modules, "tts_handler", raising=False)
        monkeypatch.delitem(sys.modules, "utils", raising=False)


def test_real_gevent_import_wsgi_close_before_body_and_double_close(native_route):
    """Prefetch holds the lease; closing an uniterated WSGI iterable frees it."""
    from generation_gate import SynthesisBusyError
    server = native_route
    gate = server.tts_handler._generation_gate
    with server.app.test_request_context(URL, method="POST",
            headers={"Authorization": "Bearer " + TOKEN},
            json={"input": "Fictional one. Fictional two."}):
        response = server.text_to_speech_stream()
        assert response.status_code == 200
        assert server.tts_handler.events == ["claimed"]
        app_iter = response.get_app_iter({"REQUEST_METHOD": "POST"})
        with pytest.raises(SynthesisBusyError):
            with gate.claim():
                pass
        app_iter.close()  # No body chunk ever iterated.
        assert server.tts_handler.events == ["claimed", "released"]
        with gate.claim():
            app_iter.close()
            response.close()
            with pytest.raises(SynthesisBusyError):
                with gate.claim():
                    pass


def test_real_gevent_server_closes_uniterated_wsgi_response(native_route):
    """Live Gevent WSGI calls the real Flask route then closes it without yielding."""
    server = native_route
    output, errors, observed = [], [], []

    def close_response_before_first_byte(environ, start_response):
        iterable = server.app(environ, start_response)
        observed.append(tuple(server.tts_handler.events))
        iterable.close()
        iterable.close()  # Repeated close is idempotent.
        return []

    service = server.WSGIServer(("127.0.0.1", 0), close_response_before_first_byte,
                               log=None, error_log=None)

    def client(port):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("POST", URL, body=json.dumps({"input": "Fictional one."}),
                               headers={"Authorization": "Bearer " + TOKEN,
                                        "Content-Type": "application/json"})
            response = connection.getresponse()
            output.append((response.status, response.read()))
        except Exception as exc:
            errors.append(type(exc).__name__)
        finally:
            connection.close()

    try:
        service.start()
        worker = threading.Thread(target=client, args=(service.server_port,), daemon=True)
        worker.start()
        deadline = time.monotonic() + 4
        while worker.is_alive() and time.monotonic() < deadline:
            gevent.sleep(0.01)
        worker.join(timeout=0.4)
        assert not worker.is_alive(), "synthetic request exceeded bounded deadline"
        assert not errors, errors
        assert output == [(200, b"")]
        assert observed == [("claimed",)], "WSGI-prefetch invariant failed inside request greenlet"
        assert server.tts_handler.events == ["claimed", "released"]
        with server.tts_handler._generation_gate.claim():
            pass
    finally:
        service.stop(timeout=1)


@pytest.mark.parametrize("disconnect", [False, True])
def test_real_gevent_wsgi_client_disconnect_and_log_privacy(native_route, caplog, disconnect):
    """Real loopback Gevent writes, fictional credentials, bounded client close."""
    server = native_route
    server.tts_handler.start_delay = 0.08 if disconnect else 0
    marker = "FICTIONAL_PRIVATE_VOICE_AND_TEXT_8459"
    service = server.WSGIServer(("127.0.0.1", 0), server.app,
                               handler_class=server._SocketExposingHandler,
                               log=None, error_log=None)
    received = []
    errors = []

    def client(port):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            c.request("POST", URL + "?secret=" + marker,
                      body=json.dumps({"input": marker + ". Follow-up."}),
                      headers={"Authorization": "Bearer " + TOKEN,
                               "Content-Type": "application/json"})
            if disconnect:
                assert c.sock is not None, "HTTP client unexpectedly lost its live socket"
                c.sock.shutdown(socket.SHUT_RDWR)
            else:
                r = c.getresponse()
                received.append((r.status, r.read()))
        except Exception as exc:
            errors.append(type(exc).__name__)
        finally:
            c.close()

    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured), caplog.at_level(logging.DEBUG):
            service.start()
            worker = threading.Thread(target=client, args=(service.server_port,), daemon=True)
            worker.start()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                gevent.sleep(0.01)
                if not worker.is_alive() and "released" in server.tts_handler.events:
                    break
            worker.join(timeout=0.4)
        assert not worker.is_alive(), "client thread exceeded bounded deadline"
        assert not errors, errors
        assert "released" in server.tts_handler.events, server.tts_handler.events
        if disconnect:
            assert received == []
            assert "second" not in server.tts_handler.events
        else:
            assert received == [(200, b"\x01\x00" * 160 + b"\x02\x00" * 160)]
        assert marker not in captured.getvalue()
        assert marker not in caplog.text
        with server.tts_handler._generation_gate.claim():
            pass
    finally:
        service.stop(timeout=1)
