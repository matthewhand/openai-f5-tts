"""Pure request validation for the local F5-TTS HTTP adapter.

This module intentionally imports no model or Flask dependencies, allowing
small offline security and compatibility regressions.
"""
import math

SUPPORTED_RESPONSE_FORMATS = frozenset(("mp3", "wav", "flac", "aac", "opus", "pcm"))


class RequestValidationError(ValueError):
    """Client-supplied request is invalid."""


def parse_speech_request(data, *, default_voice, default_format, default_speed, max_text_chars=2000):
    if not isinstance(data, dict):
        raise RequestValidationError("Expected a JSON object")
    text = data.get("input")
    if not isinstance(text, str) or not text.strip():
        raise RequestValidationError("'input' must be a nonempty string")
    if len(text) > max_text_chars:
        raise RequestValidationError("'input' exceeds the configured text limit")

    voice = data.get("voice") or default_voice
    if not isinstance(voice, str) or not voice or len(voice) > 128:
        raise RequestValidationError("'voice' must be a nonempty string of at most 128 characters")

    output_format = data.get("response_format", default_format)
    if not isinstance(output_format, str) or output_format.lower() not in SUPPORTED_RESPONSE_FORMATS:
        raise RequestValidationError("Unsupported 'response_format'")
    output_format = output_format.lower()

    speed = data.get("speed", default_speed)
    if isinstance(speed, bool):
        raise RequestValidationError("'speed' must be a finite number between 0.25 and 4")
    try:
        speed = float(speed)
    except (TypeError, ValueError, OverflowError):
        raise RequestValidationError("'speed' must be a finite number between 0.25 and 4") from None
    if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise RequestValidationError("'speed' must be a finite number between 0.25 and 4")
    return {"text": text, "voice": voice, "response_format": output_format, "speed": speed}


def parse_timestamp_request(data, *, voice, default_format, default_speed, output_format=None, max_text_chars=2000):
    """Validate the optional ElevenLabs-shaped request, returning speech parameters.

    The output format aliases deliberately map to formats the adapter can encode;
    arbitrary format names must not silently be returned as mislabeled MP3s.
    """
    if not isinstance(data, dict):
        raise RequestValidationError("Expected a JSON object")
    text = data.get("text") or data.get("input")
    settings = data.get("voice_settings") or {}
    if not isinstance(settings, dict):
        raise RequestValidationError("'voice_settings' must be a JSON object")

    raw_format = output_format or data.get("response_format") or default_format
    if not isinstance(raw_format, str):
        raise RequestValidationError("Unsupported 'output_format'")
    raw_format = raw_format.lower()
    if raw_format == "pcm" or raw_format.startswith("wav"):
        encoding = "wav"
    elif raw_format.startswith("ogg") or raw_format.startswith("opus"):
        encoding = "opus"
    elif raw_format.startswith("flac"):
        encoding = "flac"
    elif raw_format.startswith("mp3"):
        encoding = "mp3"
    else:
        raise RequestValidationError("Unsupported 'output_format'")

    return parse_speech_request(
        {
            "input": text,
            "voice": voice,
            "response_format": encoding,
            "speed": data.get("speed", settings.get("speed", default_speed)),
        },
        default_voice=voice,
        default_format=encoding,
        default_speed=default_speed,
        max_text_chars=max_text_chars,
    )
