import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.server import app as flask_app
import app.utils

# Disable API key requirement for testing
app.utils.REQUIRE_API_KEY = False

@ pytest.fixture
def client():
    return flask_app.test_client()


def test_list_models_endpoint(monkeypatch, client):
    # Mock TTSHandler.list_available_models to return dummy data
    from app.server import tts_handler
    dummy = [{'name': 'Emilia', 'status': 'unloaded'}]
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: dummy)

    resp = client.get('/v1/models')
    assert resp.status_code == 200
    assert resp.get_json() == {'models': dummy}

def test_speech_endpoint(monkeypatch, client, tmp_path):
    # Stub generate_speech to return a fake wav file
    from app.server import tts_handler
    fake_file = tmp_path / "speech.wav"
    fake_file.write_bytes(b"FAKE")
    monkeypatch.setattr(tts_handler, 'generate_speech', lambda *args, **kwargs: str(fake_file))
    # Request with wav format
    resp = client.post('/v1/audio/speech', json={'input':'Hello','response_format':'wav'})
    assert resp.status_code == 200
    assert resp.data == b"FAKE"
    assert resp.headers['Content-Type'] == 'audio/wav'
    disp = resp.headers.get('Content-Disposition','')
    assert 'attachment' in disp and 'speech.wav' in disp

def test_healthz_endpoint(client):
    resp = client.get('/healthz')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'ok'}

def test_404_endpoint(client):
    resp = client.get('/nonexistent')
    assert resp.status_code == 404
    assert resp.get_json() == {'error': 'Endpoint not found'}

def test_list_voices_endpoint(monkeypatch, client):
    from app.server import tts_handler
    dummy = [{'name': 'E', 'language': 'en'}, {'name': 'G', 'language': 'de'}]
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: dummy)
    resp = client.get('/v1/voices')
    assert resp.status_code == 200
    assert set(resp.get_json()['voices']) == {'E', 'G'}

def test_list_voices_with_filter(monkeypatch, client):
    from app.server import tts_handler
    dummy = [{'name': 'E', 'language': 'en'}, {'name': 'G', 'language': 'de'}]
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: dummy)
    resp = client.get('/v1/voices?language=en')
    assert resp.status_code == 200
    assert resp.get_json()['voices'] == ['E']

def test_list_all_voices_endpoint(monkeypatch, client):
    from app.server import tts_handler
    dummy = [{'name': 'A'}, {'name': 'B'}]
    monkeypatch.setattr(tts_handler, 'list_available_models', lambda: dummy)
    resp = client.get('/v1/voices/all')
    assert resp.status_code == 200
    assert set(resp.get_json()['voices']) == {'A', 'B'}

def test_loaded_models_endpoint(client):
    from app.server import tts_handler
    tts_handler.loaded_models = {'X': None, 'Y': None}
    resp = client.get('/v1/loaded_models')
    assert resp.status_code == 200
    assert set(resp.get_json()['loaded_models']) == {'X', 'Y'}

def test_speech_ref_audio_override(monkeypatch, client, tmp_path):
    # Stub generate_speech to capture ref_audio override
    from app.server import tts_handler
    fake = tmp_path / "fake.mp3"
    fake.write_bytes(b"DATA")
    captured = {}
    def fake_generate(text, voice, response_format, speed, ref_audio=None):
        captured['ref_audio'] = ref_audio
        return str(fake)
    monkeypatch.setattr(tts_handler, 'generate_speech', fake_generate)
    # Send POST with ref_audio
    resp = client.post('/v1/audio/speech', json={
        'input': 'Test', 'ref_audio': 'override.wav'
    })
    assert resp.status_code == 200
    assert captured['ref_audio'] == 'override.wav'
    assert resp.data == b"DATA"
    # Verify headers reflect mp3 as default
    assert resp.headers['Content-Type'] == 'audio/mpeg'
    assert 'speech.mp3' in resp.headers.get('Content-Disposition', '')

def test_speech_engine_override(monkeypatch, client):
    """API: engine override uses plugin generate"""
    import app.server as server
    class DummyEngine:
        name = 'dummy'
        def list_voices(self):
            return ['v']
        def generate(self, text, voice, response_format, speed, ref_audio=None):
            return (16000, b'OVERRIDE')
    monkeypatch.setattr(server, 'ENGINES', {'dummy': DummyEngine()})
    resp = client.post('/v1/audio/speech', json={'input': 'Hi', 'engine': 'dummy'})
    assert resp.status_code == 200
    assert resp.data == b'OVERRIDE'
    assert resp.headers['Content-Type'] == 'audio/mpeg'

def test_speech_engine_not_found(client):
    """API: unknown engine override returns 404"""
    resp = client.post('/v1/audio/speech', json={'input': 'Hi', 'engine': 'no'})
    assert resp.status_code == 404
    assert resp.get_json() == {'error': "Engine 'no' not found"}

def test_speech_missing_input(client):
    """API: missing input field returns 400"""
    resp = client.post('/v1/audio/speech', json={})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Missing 'input' in request body"}

def test_list_engines_endpoint(monkeypatch, client):
    import app.server as server
    # Ensure ENGINES empty
    monkeypatch.setattr(server, 'ENGINES', {})
    resp = client.get('/v1/engines')
    assert resp.status_code == 200
    assert resp.get_json() == {'engines': []}

def test_engine_voices_not_found(client):
    resp = client.get('/v1/engines/nonexistent/voices')
    assert resp.status_code == 404
    assert resp.get_json() == {'error': "Engine 'nonexistent' not found"}

def test_engine_voices_endpoint(monkeypatch, client):
    import app.server as server
    class Dummy:
        def list_voices(self):
            return ['X','Y']
    # Inject dummy engine
    monkeypatch.setattr(server, 'ENGINES', {'dummy': Dummy()})
    resp = client.get('/v1/engines/dummy/voices')
    assert resp.status_code == 200
    assert resp.get_json() == {'voices': ['X','Y']}

def test_mix_api_no_files(client):
    # Missing file upload
    resp = client.post('/v1/audio/mix', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400
    assert resp.get_json() == {'error': 'No input files provided'}

def test_mix_api_invalid_mode(tmp_path, client):
    import numpy as np, soundfile as sf
    # Create one file
    arr = np.array([1.0], dtype='float32')
    f = tmp_path / 'a.wav'
    sf.write(str(f), arr, 16000, subtype='FLOAT')
    with open(str(f), 'rb') as f_obj:
        data = {
            'mode': 'invalid',
            'inputs': (f_obj, 'a.wav')
        }
        resp = client.post('/v1/audio/mix', data=data, content_type='multipart/form-data')
    assert resp.status_code == 400
    assert resp.get_json() == {'error': 'Unsupported mode: invalid'}

def test_mix_api_overlay_success(tmp_path, client):
    import numpy as np, soundfile as sf, io
    # Two files for overlay
    arr1 = np.array([1.0, 2.0], dtype='float32')
    arr2 = np.array([3.0], dtype='float32')
    f1 = tmp_path / 'a.wav'
    f2 = tmp_path / 'b.wav'
    sf.write(str(f1), arr1, 16000, subtype='FLOAT')
    sf.write(str(f2), arr2, 16000, subtype='FLOAT')
    with open(str(f1), 'rb') as f1_obj, open(str(f2), 'rb') as f2_obj:
        data = {
            'mode': 'overlay',
            'inputs': [(f1_obj, 'a.wav'), (f2_obj, 'b.wav')]
        }
        resp = client.post('/v1/audio/mix', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    # Read returned WAV bytes
    result_arr, sr = sf.read(io.BytesIO(resp.data), dtype='float32')
    assert sr == 16000
    expected = np.array([(1+3)/2, (2+0)/2], dtype=result_arr.dtype)
    assert np.allclose(result_arr, expected)

def test_mix_api_sample_rate_mismatch(tmp_path, client):
    import numpy as np, soundfile as sf
    # Two files with different SR
    arr1 = np.array([0.1], dtype='float32')
    arr2 = np.array([0.2], dtype='float32')
    f1 = tmp_path / 'a.wav'
    f2 = tmp_path / 'b.wav'
    sf.write(str(f1), arr1, 8000, subtype='FLOAT')
    sf.write(str(f2), arr2, 16000, subtype='FLOAT')
    with open(str(f1), 'rb') as f1_obj, open(str(f2), 'rb') as f2_obj:
        data = {
            'mode': 'overlay',
            'inputs': [(f1_obj, 'a.wav'), (f2_obj, 'b.wav')]
        }
        resp = client.post('/v1/audio/mix', data=data, content_type='multipart/form-data')
    assert resp.status_code == 400
    # Ensure error mentions sample rate
    assert 'Sample rate mismatch' in resp.get_json().get('error', '')

def test_mix_api_concat_success(tmp_path, client):
    import numpy as np, soundfile as sf, io
    # Two files for concat
    arr1 = np.array([1.0, 2.0], dtype='float32')
    arr2 = np.array([3.0, 4.0], dtype='float32')
    f1 = tmp_path / 'a.wav'
    f2 = tmp_path / 'b.wav'
    sf.write(str(f1), arr1, 16000, subtype='FLOAT')
    sf.write(str(f2), arr2, 16000, subtype='FLOAT')
    with open(str(f1), 'rb') as f1_obj, open(str(f2), 'rb') as f2_obj:
        data = {
            'mode': 'concat',
            'inputs': [(f1_obj, 'a.wav'), (f2_obj, 'b.wav')]
        }
        resp = client.post('/v1/audio/mix', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    # Read returned WAV bytes
    result_arr, sr = sf.read(io.BytesIO(resp.data), dtype='float32')
    assert sr == 16000
    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=result_arr.dtype)
    assert np.allclose(result_arr, expected)
