import pytest
from app.engine import BaseEngine, load_engines, mix_audio

class DummyEngine(BaseEngine):
    name = 'dummy'
    def list_voices(self):
        return ['a','b']
    def generate(self, text, voice, **opts):
        return (22050, b'rawdata')

def test_base_engine_methods():
    with pytest.raises(TypeError):
        BaseEngine()

def test_dummy_engine():
    de = DummyEngine()
    assert de.list_voices() == ['a','b']
    sr, audio = de.generate('hi', 'a')
    assert sr == 22050
    assert isinstance(audio, (bytes, bytearray))

def test_load_engines_empty():
    engines = load_engines()
    assert isinstance(engines, dict)
    # Local engine stubs should be loaded by default
    assert 'f5' in engines
    assert 'kokoro' in engines

def test_load_engines_discovery(monkeypatch):
    import pkg_resources
    # Create a dummy entry point for 'dummy' engine
    class DummyEP:
        name = 'dummy'
        def load(self):
            return DummyEngine
    # Monkeypatch iter_entry_points to return our dummy
    def dummy_iter(group):
        return [DummyEP()] if group == 'tts_engines' else []
    monkeypatch.setattr(pkg_resources, 'iter_entry_points', dummy_iter)
    engines = load_engines()
    assert 'dummy' in engines
    engine = engines['dummy']
    assert isinstance(engine, DummyEngine)
    assert engine.list_voices() == ['a','b']
    assert engines['dummy'].name == 'dummy'

def test_load_local_plugins(tmp_path):
    # Ensure app.engines package is recognized
    # We already have f5 and kokoro modules
    engines = load_engines()
    # Both local stubs should be present
    assert 'f5' in engines
    assert 'kokoro' in engines
    assert engines['f5'].name == 'f5'
    assert engines['kokoro'].name == 'kokoro'

def test_mix_audio_not_implemented():
    with pytest.raises(NotImplementedError):
        mix_audio([b'data1', b'data2'], mode='overlay')

def test_mix_audio_concat_bytes():
    # Bytes should concatenate
    assert mix_audio([b'foo', b'bar'], mode='concat') == b'foobar'

def test_mix_audio_concat_numpy():
    # Numpy arrays should concatenate
    import numpy as np
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    result = mix_audio([arr1, arr2], mode='concat')
    assert np.array_equal(result, np.array([1, 2, 3, 4]))

def test_mix_audio_overlay_bytes_not_supported():
    # Overlay mode for bytes should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        mix_audio([b'foo', b'bar'], mode='overlay')

def test_mix_audio_overlay_numpy_arrays():
    # Overlay numpy arrays should average values with padding
    import numpy as np
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5])
    result = mix_audio([arr1, arr2], mode='overlay')
    expected = np.array([(1 + 4) / 2, (2 + 5) / 2, (3 + 0) / 2])
    assert np.allclose(result, expected)

def test_mix_audio_unsupported_mode():
    # Unsupported mode should raise ValueError
    with pytest.raises(ValueError):
        mix_audio([b'a'], mode='invalid')
