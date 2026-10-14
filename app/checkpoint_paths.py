"""Minimal fail-closed checkpoint path checks, with no model imports.

Reject pathname escapes, symlinks (including root ancestors), directories and
multiply linked checkpoint files. All checks are metadata-only and do NOT
eliminate time-of-check/time-of-use swaps before upstream file opening, prove
checkpoint provenance, or validate safetensors contents.
"""
import os
import stat


class UntrustedCheckpointPath(RuntimeError):
    """Unsafe or stale checkpoint path; do not hand it to the model loader."""


CHECKPOINT_NAMES = ("model_1250000.safetensors", "model_1200000.safetensors")


def validate_checkpoint_root(root):
    try:
        root_path = os.path.abspath(os.fspath(root))
        # Detect symlinks in the configured root and all path ancestors.
        if os.path.realpath(root_path) != root_path:
            raise UntrustedCheckpointPath("Checkpoint root traverses a symlink")
        if not stat.S_ISDIR(os.lstat(root_path).st_mode):
            raise UntrustedCheckpointPath("Checkpoint root is not a directory")
    except (OSError, TypeError, ValueError) as exc:
        raise UntrustedCheckpointPath("Checkpoint root is unavailable or invalid") from exc
    return root_path


def validate_checkpoint_path(root, voice_name, candidate_path):
    """Return existing path only if metadata stays within a trusted root.

    The result is an ordinary pathname for an upstream path-only loader; an
    attacker who can rewrite directories concurrently could still race the
    subsequent open. The checkpoint tree must be trusted and not writable by
    less-privileged actors.
    """
    root_path = validate_checkpoint_root(root)
    if (not isinstance(voice_name, str) or not voice_name or voice_name in (".", "..")
            or "/" in voice_name or "\\" in voice_name or "\x00" in voice_name):
        raise UntrustedCheckpointPath("Invalid checkpoint voice directory name")
    try:
        candidate = os.fspath(candidate_path)
        filename = os.path.basename(candidate)
        expected = os.path.join(root_path, voice_name, filename)
        if filename not in CHECKPOINT_NAMES or os.path.abspath(candidate) != expected:
            raise UntrustedCheckpointPath("Checkpoint pathname escapes its voice directory")
        voice_path = os.path.join(root_path, voice_name)
        if not stat.S_ISDIR(os.lstat(voice_path).st_mode):
            raise UntrustedCheckpointPath("Checkpoint voice directory is not a real directory")
        file_stat = os.lstat(expected)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise UntrustedCheckpointPath("Checkpoint is not a singly linked regular file")
    except (OSError, TypeError, ValueError) as exc:
        raise UntrustedCheckpointPath("Checkpoint path is stale or invalid") from exc
    return candidate
