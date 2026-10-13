"""Fail-fast process-local admission for heavyweight speech generation.

No GPU, web-server, model, or third-party imports. This is not a distributed lock:
multiple WSGI worker processes can still synthesize independently.
"""
from contextlib import contextmanager
from threading import BoundedSemaphore


class SynthesisBusyError(RuntimeError):
    """The sole local synthesis slot is already occupied."""


class GenerationGate:
    def __init__(self):
        self._slot = BoundedSemaphore(value=1)

    @contextmanager
    def claim(self):
        if not self._slot.acquire(blocking=False):
            raise SynthesisBusyError("Speech generator busy; try again later")
        try:
            yield
        finally:
            self._slot.release()
