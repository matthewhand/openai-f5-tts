"""Pure deterministic contention regressions; no models or Flask required."""
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from generation_gate import GenerationGate, SynthesisBusyError


class GenerationGateTests(unittest.TestCase):
    def test_second_claim_fails_fast_without_releasing_first(self):
        gate = GenerationGate()
        with gate.claim():
            with self.assertRaises(SynthesisBusyError):
                with gate.claim():
                    self.fail("Busy request entered generator")
            with self.assertRaises(SynthesisBusyError):
                with gate.claim():
                    self.fail("Failed acquisition must not release existing slot")
        with gate.claim():
            pass

    def test_exceptions_always_release_slot(self):
        gate = GenerationGate()
        with self.assertRaisesRegex(ValueError, "synthetic"):
            with gate.claim():
                raise ValueError("synthetic")
        with gate.claim():
            pass

    def test_contention_from_another_thread_is_bounded(self):
        gate = GenerationGate()
        inside = threading.Event()
        release = threading.Event()
        failures = []

        def holder():
            with gate.claim():
                inside.set()
                if not release.wait(2):
                    failures.append("holder timed out")

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        try:
            self.assertTrue(inside.wait(2))
            with self.assertRaises(SynthesisBusyError):
                with gate.claim():
                    self.fail("Second concurrent job should not run")
        finally:
            release.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        with gate.claim():
            pass


if __name__ == "__main__":
    unittest.main()
