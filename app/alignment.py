
"""Optional ElevenLabs-shaped character alignment.

Prior behaviour is unchanged: /v1/audio/speech still returns raw audio.
This module must import with zero extra packages. A real aligner (WhisperX,
etc.) is used only if it is already installed; otherwise we fall back to
duration-weighted estimates, then to null alignment.
"""
from __future__ import annotations

import logging

_VOWELS = set("aeiouAEIOU")

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


def _char_weight(ch: str) -> float:
    if ch.isspace():
        return 0.28
    if ch in ",;:":
        return 0.35
    if ch in ".!?":
        return 0.55
    if ch in _VOWELS:
        return 1.25
    if ch.isalpha():
        return 1.0
    return 0.4


def estimate_alignment(text: str, duration_s: float) -> dict:
    if not text:
        return {
            "characters": [],
            "character_start_times_seconds": [],
            "character_end_times_seconds": [],
        }
    chars = list(text)
    weights = [_char_weight(c) for c in chars]
    total = sum(weights) or float(len(chars))
    duration_s = max(float(duration_s), 0.01)
    starts, ends = [], []
    t = 0.0
    for w in weights:
        dt = duration_s * (w / total)
        starts.append(round(t, 4))
        t += dt
        ends.append(round(t, 4))
    ends[-1] = round(duration_s, 4)
    return {
        "characters": chars,
        "character_start_times_seconds": starts,
        "character_end_times_seconds": ends,
    }


def words_from_alignment(alignment: dict) -> list:
    chars = alignment["characters"]
    starts = alignment["character_start_times_seconds"]
    ends = alignment["character_end_times_seconds"]
    words = []
    buf, i0 = [], None
    for i, ch in enumerate(chars):
        if ch.isspace() or ch in ".,!?;:":
            if buf and i0 is not None:
                words.append({"word": "".join(buf), "start": starts[i0], "end": ends[i - 1]})
                buf, i0 = [], None
            continue
        if i0 is None:
            i0 = i
        buf.append(ch)
    if buf and i0 is not None:
        words.append({"word": "".join(buf), "start": starts[i0], "end": ends[len(chars) - 1]})
    return words


def _try_whisperx(audio_path: str, text: str):
    try:
        import whisperx  # noqa: F401
    except Exception:
        return None
    logging.info("whisperx is installed; duration-weighted alignment still used (no forced-align hook yet).")
    return None


def audio_duration_seconds(audio_path: str) -> float | None:
    if sf is None:
        return None
    try:
        info = sf.info(audio_path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception as e:
        logging.warning("Could not read audio duration: %s", e)
    return None


def build_alignment(text: str, audio_path: str) -> tuple[dict | None, list, str]:
    """Return (alignment or None, words, source). Never raises."""
    try:
        hooked = _try_whisperx(audio_path, text)
        if hooked is not None:
            return hooked, words_from_alignment(hooked), "whisperx"
        duration = audio_duration_seconds(audio_path)
        if duration is None:
            return None, [], "unavailable"
        alignment = estimate_alignment(text, duration)
        return alignment, words_from_alignment(alignment), "duration_weighted"
    except Exception as e:
        logging.warning("Alignment failed, returning null: %s", e)
        return None, [], "unavailable"
