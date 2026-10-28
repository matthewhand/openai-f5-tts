"""Synthetic F5 boundaries for CPU-native Torch tests; no weights or inference."""
import sys
from types import ModuleType


def stub_native_backend(monkeypatch):
    """Import real torch/librosa/gevent but replace unused upstream model startup."""
    # Native ARM64 Numba can JIT the production resampler for minutes; exercise
    # the streaming contract using SciPy's real polyphase resampler instead.
    import scipy.signal
    audio = ModuleType("librosa")
    audio.resample = lambda data, *, orig_sr, target_sr: scipy.signal.resample_poly(
        data, target_sr, orig_sr)
    monkeypatch.setitem(sys.modules, "librosa", audio)

    tf = ModuleType("transformers")
    tf.WhisperProcessor = type("SyntheticWhisperProcessor", (), {})
    tf.WhisperForConditionalGeneration = type("SyntheticWhisperModel", (), {})

    f5 = ModuleType("f5_tts")
    f5.__path__ = []
    model = ModuleType("f5_tts.model")
    model.DiT = type("SyntheticDiT", (), {})
    infer = ModuleType("f5_tts.infer")
    infer.__path__ = []
    utils = ModuleType("f5_tts.infer.utils_infer")

    def no_native_inference(*args, **kwargs):
        raise AssertionError("Native backend must never load models or infer in these tests")

    for name in ("load_vocoder", "load_model", "infer_process",
                 "preprocess_ref_audio_text", "remove_silence_for_generated_wav"):
        setattr(utils, name, no_native_inference)
    for name, module in (("transformers", tf), ("f5_tts", f5),
                         ("f5_tts.model", model), ("f5_tts.infer", infer),
                         ("f5_tts.infer.utils_infer", utils)):
        monkeypatch.setitem(sys.modules, name, module)
