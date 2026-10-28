"""Pure request validation for the local F5-TTS HTTP adapter.

This module intentionally imports no model or Flask dependencies, allowing
small offline security and compatibility regressions.
"""
import math

SUPPORTED_RESPONSE_FORMATS = frozenset(("mp3", "wav", "flac", "aac", "opus", "pcm"))

# Explicit opt-in encoder contracts. Historical numeric aliases remain codec-only.
# (file format, output sample rate, encoder target bitrate, channels)
STRICT_TIMESTAMP_PROFILES = {
    "mp3_44100_128_strict": ("mp3", 44100, 128000, 1),
    "ogg_48000_128_strict": ("opus", 48000, 128000, 1),
    "wav_44100_pcm16_strict": ("wav", 44100, None, 1),
}


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

    voice = data.get("voice")
    if voice is None or voice == "":
        voice = default_voice
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
    if not isinstance(text, str) or not text.strip():
        raise RequestValidationError("'text' (or 'input') must be a nonempty string")
    settings = data.get("voice_settings")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise RequestValidationError("'voice_settings' must be a JSON object")

    # Keep only verified historical aliases; arbitrary codec prefixes cannot
    # become a different encoded format through silent fallback.
    if output_format is not None:
        raw_format = output_format
    elif "response_format" in data:
        raw_format = data["response_format"]
    else:
        raw_format = default_format
    if not isinstance(raw_format, str):
        raise RequestValidationError("Unsupported 'output_format'")
    codecs = {
        "mp3": "mp3", "wav": "wav", "flac": "flac",
        "aac": "aac", "opus": "opus", "ogg": "opus", "pcm": "wav",
        "wav_44100_128": "wav", "mp3_44100_128": "mp3",
        "ogg_48000_128": "opus",
    }
    profile = raw_format.lower() if raw_format.lower() in STRICT_TIMESTAMP_PROFILES else None
    encoding = STRICT_TIMESTAMP_PROFILES[profile][0] if profile else codecs.get(raw_format.lower())
    if encoding is None:
        raise RequestValidationError("Unsupported 'output_format'")

    params = parse_speech_request(
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
    if profile is not None:
        params["encoding_profile"] = profile
    return params


def parse_stream_request(data, *, default_voice, default_speed, max_text_chars=2000):
    """Streaming always emits raw PCM; reuse the strict speech validator."""
    if isinstance(data, dict):
        if "output_format" in data:
            raise RequestValidationError("Streaming does not accept 'output_format'")
        if "response_format" in data:
            fmt = data["response_format"]
            if not isinstance(fmt, str) or fmt.lower() != "pcm":
                raise RequestValidationError("Streaming supports only 'pcm' response_format")
    return parse_speech_request(
        data, default_voice=default_voice, default_format="pcm",
        default_speed=default_speed, max_text_chars=max_text_chars,
    )
