import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Ensure project root import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.tts_handler import TTSHandler

@ pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    # Stub out model loading to avoid HF calls
    monkeypatch.setattr(TTSHandler, 'load_whisper_model', lambda self: (object(), object()))
    monkeypatch.setattr('app.tts_handler.load_vocoder', lambda: object())
    # Stub F5-TTS functions to avoid actual inference
    monkeypatch.setattr('app.tts_handler.load_model', lambda *args, **kwargs: object())
    # Use constant audio data
    monkeypatch.setattr(TTSHandler, 'infer', lambda self, *args, **kwargs: (16000, np.zeros(16000)))


def test_generate_speech_no_text():
    handler = TTSHandler(retain_cache=True, disable_pcm_normalization=True, default_voice='Emilia')
    with pytest.raises(ValueError):
        handler.generate_speech('', response_format='wav')


def test_generate_speech_unknown_voice():
    handler = TTSHandler(retain_cache=True, disable_pcm_normalization=True, default_voice='Emilia')
    with pytest.raises(ValueError):
        handler.generate_speech('Hello', voice='Unknown')


def test_generate_speech_success(tmp_path):
    handler = TTSHandler(retain_cache=True, disable_pcm_normalization=True, default_voice='Emilia')
    # Prepare available_models and stub load_voice_model
    handler.available_models = {'Emilia': 'dummy_path'}
    handler.load_voice_model = lambda voice: object()

    out = handler.generate_speech('Test text', voice='Emilia', response_format='wav')
    # File should exist and contain zeros
    assert Path(out).exists()
    import soundfile as sf
    data, rate = sf.read(out)
    assert rate == 16000
    assert data.shape[0] == 16000
    # Clean up
    os.remove(out)


def test_expression_speed_adjustment(monkeypatch):
    # Test that expression tags adjust speed
    handler = TTSHandler(retain_cache=True, disable_pcm_normalization=True, default_voice='Emilia')
    handler.available_models = {'Emilia': 'dummy'}
    handler.load_voice_model = lambda voice: object()
    captured = {}
    def fake_infer(self, text, voice, model, speed, ref_audio=None):
        captured['text'] = text
        captured['speed'] = speed
        return (16000, np.zeros(16000))
    monkeypatch.setattr(TTSHandler, 'infer', fake_infer)
    # Generate speech with expression tag
    out = handler.generate_speech('Hello {happy} world', voice='Emilia', response_format='wav', speed=1.0)
    # Speed should be multiplied by happy factor (1.2)
    assert captured['speed'] == pytest.approx(1.2)
    # Expression tags stripped from text
    assert '{happy}' not in captured['text']
    # Output file created
    assert Path(out).exists()
    os.remove(out)


def test_multiple_expression_speed_adjustment(monkeypatch):
    # Test that multiple expression tags adjust speed cumulatively
    handler = TTSHandler(retain_cache=True, disable_pcm_normalization=True, default_voice='Emilia')
    handler.available_models = {'Emilia': 'dummy'}
    handler.load_voice_model = lambda voice: object()
    captured = {}
    def fake_infer(self, text, voice, model, speed, ref_audio=None):
        captured['speed'] = speed
        captured['text'] = text
        return (16000, np.zeros(16000))
    monkeypatch.setattr(TTSHandler, 'infer', fake_infer)
    # happy multiplier 1.2, sad multiplier 0.8 => cumulative 0.96
    out = handler.generate_speech('Hello {happy}{sad} world', voice='Emilia', response_format='wav', speed=1.0)
    assert captured['speed'] == pytest.approx(1.2 * 0.8)
    # Expression tags stripped from text
    assert '{happy}' not in captured['text'] and '{sad}' not in captured['text']
    assert Path(out).exists()
    os.remove(out)
