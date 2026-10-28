"""Native Gevent log-sink and fictional loopback HTTP request regressions."""
import io
from contextlib import redirect_stderr, redirect_stdout

import pytest


def test_real_gevent_server_disables_request_and_error_log_sinks():
    pytest.importorskip("gevent.pywsgi")
    from gevent.pywsgi import WSGIServer
    server = WSGIServer(("127.0.0.1", 0), lambda environ, start: [],
                        log=None, error_log=None)
    marker = "SYNTHETIC_PRIVATE_VOICE_PATH_TOKEN_8246"
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            server.log.write(marker)
            server.error_log.write(marker)
        assert marker not in captured.getvalue()
        assert type(server.log).__name__ == "_NoopLog"
        assert type(server.error_log).__name__ == "_NoopLog"
    finally:
        server.close()


def test_real_gevent_http_request_privacy(caplog):
    """Ephemeral loopback server only; test Gevent access logging, no inference."""
    gevent = pytest.importorskip("gevent")
    import http.client
    import threading
    import time
    from gevent.pywsgi import WSGIServer

    marker = "SYNTHETIC_PRIVATE_REQUEST_TOKEN_8476"

    def app(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]

    server = WSGIServer(("127.0.0.1", 0), app, log=None, error_log=None)
    captured = io.StringIO()
    responses = []
    errors = []

    def client(port):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("POST", "/v1/private/" + marker + "?key=" + marker,
                               body=marker, headers={"Authorization": "Bearer " + marker})
            response = connection.getresponse()
            responses.append((response.status, response.read()))
        except Exception as exc:
            errors.append(type(exc).__name__)
        finally:
            connection.close()

    try:
        with redirect_stdout(captured), redirect_stderr(captured), caplog.at_level("DEBUG"):
            server.start()
            worker = threading.Thread(target=client, args=(server.server_port,), daemon=True)
            worker.start()
            deadline = time.monotonic() + 4
            while worker.is_alive() and time.monotonic() < deadline:
                gevent.sleep(0.01)
            worker.join(timeout=0.5)
        assert not worker.is_alive(), "Synthetic request exceeded deadline"
        assert not errors, errors
        assert responses == [(200, b"ok")]
        assert marker not in captured.getvalue()
        assert marker not in caplog.text
    finally:
        server.stop(timeout=1)
