"""
TTS Engine Abstraction

TODO:
- Integrate into CLI and API for multi-engine support
- Add mixing and orchestration utilities
"""
from abc import ABC, abstractmethod
import logging
import pkg_resources

class BaseEngine(ABC):
    """
    Abstract base class for TTS engines
    """
    name: str

    @abstractmethod
    def list_voices(self) -> list[str]:
        """Return a list of available voice names."""
        ...

    @abstractmethod
    def generate(self, text: str, voice: str, **opts) -> tuple[int, bytes]:
        """
        Generate speech audio for the given text and voice.
        Returns a tuple of (sample_rate, audio_bytes).
        """
        ...


def load_engines() -> dict[str, BaseEngine]:
    """
    Discover and load TTS engine plugins via entry_points 'tts_engines'.
    Return a mapping of engine name to BaseEngine instance.
    """
    engines: dict[str, BaseEngine] = {}
    # Entry-point plugins
    for ep in pkg_resources.iter_entry_points('tts_engines'):
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            engines[plugin.name] = plugin
        except Exception as e:
            logging.warning(f"Failed to load engine {ep.name}: {e}")
    # Local plugins in app.engines
    try:
        import importlib, pkgutil, app.engines as plugin_pkg
        for _, module_name, _ in pkgutil.iter_modules(plugin_pkg.__path__):
            try:
                mod = importlib.import_module(f'app.engines.{module_name}')
                cls = getattr(mod, 'Engine', None)
                if cls and issubclass(cls, BaseEngine):
                    inst = cls()
                    engines[inst.name] = inst
            except Exception as e:
                logging.warning(f"Failed to load local engine module {module_name}: {e}")
    except ImportError:
        pass
    return engines

def mix_audio(audio_arrays: list, mode: str = "overlay"):
    """
    Mix multiple audio arrays or bytes.
    Args:
        audio_arrays (list): List of audio data (numpy arrays or bytes).
        mode (str): Mixing mode ('overlay' or 'concat').
    Returns:
        Mixed audio array or bytes.
    """
    # Overlay mode
    if mode == "overlay":
        # Overlay numpy audio arrays (raw bytes not supported)
        if not audio_arrays:
            return b""
        first = audio_arrays[0]
        if isinstance(first, (bytes, bytearray)):
            raise NotImplementedError("mix_audio overlay not implemented for bytes")
        import numpy as np
        # Pad arrays to equal length
        max_len = max(arr.shape[0] for arr in audio_arrays)
        padded = [np.pad(arr, (0, max_len - arr.shape[0]), mode="constant") for arr in audio_arrays]
        mixed = sum(padded)
        # Normalize to prevent clipping
        mixed = mixed / len(audio_arrays)
        return mixed
    # Concatenate mode: simple join of sequences
    if mode == "concat":
        if not audio_arrays:
            return b""
        first = audio_arrays[0]
        # Bytes concatenation
        if isinstance(first, (bytes, bytearray)):
            return b"".join(audio_arrays)
        # Numpy arrays concatenation
        try:
            import numpy as np
            return np.concatenate(audio_arrays)
        except ImportError:
            raise RuntimeError("numpy is required for concat mode on non-byte arrays")
    # Unsupported mode
    raise ValueError(f"Unsupported mode: {mode}")
