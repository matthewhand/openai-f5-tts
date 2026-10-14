"""Fail-closed metadata checks for reference audio and predictable cache files.

Pathname checks are only advisory until an upstream pathname-only reader/writer
opens the file. Keep the entire directory tree private to trusted writers.
There is no FD handoff or guarantee against a concurrent rename/swap.
"""
import os
import stat

INPUT_EXTENSIONS = (".wav", ".mp3", ".flac", ".aac", ".ogg", ".pcm")


class UntrustedReferencePath(RuntimeError):
    """Input or cache path is unsafe, stale, or outside its intended directory."""


def validate_reference_voice(voice):
    if (not isinstance(voice, str) or not voice or voice in (".", "..")
            or "/" in voice or "\\" in voice or "\x00" in voice):
        raise UntrustedReferencePath("Invalid reference voice name")
    return voice


def validate_reference_root(root):
    try:
        normalized = os.path.abspath(os.fspath(root))
        if os.path.realpath(normalized) != normalized:
            raise UntrustedReferencePath("Reference directory has symlinked ancestors")
        if not stat.S_ISDIR(os.lstat(normalized).st_mode):
            raise UntrustedReferencePath("Reference root is not a real directory")
    except (OSError, TypeError, ValueError) as exc:
        raise UntrustedReferencePath("Reference directory unavailable") from exc
    return normalized


def _checked_file(root, voice, path, expected_name, *, allow_missing):
    root_path = validate_reference_root(root)
    validate_reference_voice(voice)
    try:
        candidate = os.fspath(path)
        expected = os.path.join(root_path, expected_name)
        if os.path.abspath(candidate) != expected:
            raise UntrustedReferencePath("Reference path escapes its directory")
        try:
            info = os.lstat(expected)
        except FileNotFoundError as exc:
            if allow_missing:
                return candidate
            raise UntrustedReferencePath("Reference file is missing") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise UntrustedReferencePath("Reference file must be a singly linked regular file")
    except (OSError, TypeError, ValueError) as exc:
        raise UntrustedReferencePath("Reference file path is invalid") from exc
    return candidate


def validate_reference_input(root, voice, path):
    validate_reference_voice(voice)
    filename = os.path.basename(os.fspath(path))
    if not any(filename == voice + ext for ext in INPUT_EXTENSIONS):
        raise UntrustedReferencePath("Unexpected reference-audio filename")
    return _checked_file(root, voice, path, filename, allow_missing=False)


def validate_reference_cache(root, voice, path, extension, *, allow_missing=False):
    validate_reference_voice(voice)
    if extension not in ("wav", "pcm"):
        raise UntrustedReferencePath("Invalid reference-cache file extension")
    return _checked_file(
        root, voice, path, voice + "_processed." + extension,
        allow_missing=allow_missing,
    )


def require_missing_cache_intermediate(root, voice, path):
    validate_reference_cache(root, voice, path, "pcm", allow_missing=True)
    if os.path.lexists(path):
        raise UntrustedReferencePath("Reference-cache intermediate already exists")
