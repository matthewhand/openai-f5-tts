import pytest
from app.server import app, tts_handler, ENGINES
import app.utils as utils

@pytest.fixture(autouse=True)
def disable_auth(monkeypatch):
    # Disable API key requirement for testing
    monkeypatch.setattr(utils, 'REQUIRE_API_KEY', False)

@pytest.fixture
def client():
    return app.test_client()


def test_list_models(client, monkeypatch):
    # Stub available models
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: [{'name': 'Emilia'}])
    rv = client.get('/v1/models')
    assert rv.status_code == 200
    assert rv.get_json() == {'models': [{'name': 'Emilia'}]}


def test_list_voices(client, monkeypatch):
    # Stub available models
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: [{'name': 'A'}, {'name': 'B'}])
    rv = client.get('/v1/voices')
    assert rv.status_code == 200
    assert rv.get_json() == {'voices': ['A', 'B']}


def test_list_all_voices(client, monkeypatch):
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: [{'name': 'X'}, {'name': 'Y'}])
    rv = client.get('/v1/voices/all')
    assert rv.status_code == 200
    assert rv.get_json() == {'voices': ['X', 'Y']}


def test_list_engines(client, monkeypatch):
    # Replace ENGINES mapping
    from app import server
    Dummy = type('D', (), {'list_voices': lambda self: []})
    monkeypatch.setattr(server, 'ENGINES', {'f5': Dummy(), 'kokoro': Dummy()})
    rv = client.get('/v1/engines')
    assert rv.status_code == 200
    assert set(rv.get_json().get('engines')) == {'f5', 'kokoro'}


def test_engine_voices_success(client, monkeypatch):
    from app import server
    class Dummy:
        def list_voices(self):
            return ['v1', 'v2']
    monkeypatch.setattr(server, 'ENGINES', {'f5': Dummy()})
    rv = client.get('/v1/engines/f5/voices')
    assert rv.status_code == 200
    assert rv.get_json() == {'voices': ['v1', 'v2']}


def test_engine_voices_not_found(client, monkeypatch):
    from app import server
    monkeypatch.setattr(server, 'ENGINES', {})
    rv = client.get('/v1/engines/unknown/voices')
    assert rv.status_code == 404
    assert 'error' in rv.get_json()
