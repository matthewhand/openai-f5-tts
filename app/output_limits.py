"""Bound HTTP audio response buffering without importing models or Flask.

A generated audio file is owned by its calling request.  This function reads
at most max_bytes + 1 bytes even when a path points to a very large file.
"""
class AudioOutputTooLarge(RuntimeError):
    """Generated audio exceeds the configured in-memory output limit."""


def read_audio_limited(path, max_bytes: int) -> bytes:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("MAX_AUDIO_OUTPUT_BYTES must be a positive integer")
    with open(path, "rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AudioOutputTooLarge("Generated audio exceeds configured output limit")
    return data
