"""TTSHandler.generate_speech_stream behaviour, with F5/Whisper/vocoder stubbed.

Runs on CPU (needs torch/numpy/librosa installed, i.e. inside the project image).
"""
import importlib
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
SR_IN = 24000
SR_OUT = 16000


@pytest.fixture()
def th(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(APP_DIR)
    monkeypatch.setenv("API_KEY", "k")
    for name in ("tts_handler", "utils"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("tts_handler")

    ref_wav = tmp_path / "ref.wav"
    sf.write(str(ref_wav), np.zeros(16000, dtype=np.float32), 16000)

    h = mod.TTSHandler.__new__(mod.TTSHandler)
    h.device = torch.device("cpu")
    h.available_models = {"Scarlett": "ignored"}
    h.loaded_models = OrderedDict()
    h.vocoder = object()
    h.disable_pcm_normalization = True

    calls = SimpleNamespace(load=0, process=0, whisper=0, infer=[], amp=0.5)
    h._calls = calls

    def load_voice_model(name):
        calls.load += 1
        return "MODEL"

    def process_reference_audio(name):
        calls.process += 1
        return str(ref_wav)

    class Proc:
        def __call__(self, audio, sampling_rate, return_tensors):
            return SimpleNamespace(input_features=torch.zeros(1, 1))

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["reference words"]

    class Whisper:
        def generate(self, feats):
            calls.whisper += 1
            return torch.zeros(1, 1)

    h.processor, h.whisper_model = Proc(), Whisper()
    h.load_voice_model = load_voice_model
    h.process_reference_audio = process_reference_audio

    def fake_infer_process(ref_audio, ref_text, gen_text, model, vocoder, **kw):
        calls.infer.append(SimpleNamespace(ref_text=ref_text, gen_text=gen_text, model=model, **kw))
        secs = 0.1 * len(gen_text.split())  # duration scales with word count
        t = np.arange(int(SR_IN * secs)) / SR_IN
        wave = (calls.amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return wave, SR_IN, None

    monkeypatch.setattr(mod, "infer_process", fake_infer_process)
    yield h
    for name in ("tts_handler", "utils"):
        sys.modules.pop(name, None)


def stream(h, text, **kw):
    return list(h.generate_speech_stream(text, "Scarlett", **kw))


def test_one_chunk_per_sentence(th):
    chunks = stream(th, "Hello there. How are you? I am fine!")
    assert len(chunks) == 3
    assert [c.gen_text for c in th._calls.infer] == [
        "Hello there.", "How are you?", "I am fine!"]


def test_chunks_are_int16_pcm_at_16k(th):
    (chunk,) = stream(th, "one two three four five")  # 0.5 s at 24 kHz
    assert isinstance(chunk, bytes) and len(chunk) % 2 == 0
    samples = np.frombuffer(chunk, dtype="<i2")
    assert abs(len(samples) - int(0.5 * SR_OUT)) <= 2  # resampled 24k -> 16k
    assert samples.max() > 10000, "audio should not be silent"


def test_loud_audio_is_clipped_not_wrapped(th):
    th._calls.amp = 1.8
    (chunk,) = stream(th, "one two three")
    s = np.frombuffer(chunk, dtype="<i2")
    assert s.max() == 32767 and s.min() == -32767


def test_model_ref_text_and_ref_audio_resolved_once_not_per_sentence(th):
    stream(th, "A b. C d. E f. G h.")
    assert th._calls.load == 1
    assert th._calls.whisper == 1
    assert th._calls.process <= 2  # once for transcription path, once for infer path
    assert {c.model for c in th._calls.infer} == {"MODEL"}
    assert {c.ref_text for c in th._calls.infer} == {"reference words"}


def test_reference_transcription_cached_across_requests(th):
    stream(th, "Hello there.")
    stream(th, "Another request.")
    assert th._calls.whisper == 1, "Whisper must run once per voice, not once per request"


def test_speed_is_forwarded_to_f5(th):
    stream(th, "Hello there. Again.", speed=1.3)
    assert [c.speed for c in th._calls.infer] == [1.3, 1.3]


def test_stopping_consumption_stops_inference(th):
    g = th.generate_speech_stream("One a. Two b. Three c. Four d.", "Scarlett")
    next(g)
    g.close()  # what the server does when the client disconnects
    assert len(th._calls.infer) == 1


@pytest.mark.parametrize("text,expected", [
    ("Dr. Smith arrived. He sat.", 2),
    ("It costs $3.50 today. Fine.", 2),
    ("3.14 is pi. Yes.", 2),
    ("No terminal punctuation here", 1),
    ("Line one.\n\nLine two.\nLine three.", 3),
    ("Wait... what? Really!", 2),
])
def test_sentence_splitting(th, text, expected):
    assert len(stream(th, text)) == expected


def test_empty_text_and_unknown_voice_raise_value_error(th):
    with pytest.raises(ValueError):
        next(th.generate_speech_stream("", "Scarlett"))
    with pytest.raises(ValueError):
        next(th.generate_speech_stream("Hi.", "Nobody"))


def test_sentence_splitter_needs_no_network(th, monkeypatch):
    """Streaming must work offline: no runtime data downloads (e.g. NLTK punkt)."""
    import socket

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", boom)
    assert len(stream(th, "One a. Two b.")) == 2
