"""Model-free synthetic P1 output-buffer boundaries."""
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from output_limits import AudioOutputTooLarge, read_audio_limited


class LimitedAudioReadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="f5-output-limit-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic-audio.bin"

    def test_exact_limit_returns_complete_file(self):
        self.path.write_bytes(b"0123456789")
        self.assertEqual(read_audio_limited(self.path, 10), b"0123456789")

    def test_zero_length_is_allowed(self):
        self.path.write_bytes(b"")
        self.assertEqual(read_audio_limited(self.path, 10), b"")

    def test_oversize_file_is_rejected(self):
        self.path.write_bytes(b"0123456789X")
        with self.assertRaises(AudioOutputTooLarge):
            read_audio_limited(self.path, 10)

    def test_read_is_bounded_to_limit_plus_one(self):
        called = []
        class FakeReader(io.BytesIO):
            def read(self, count=-1):
                called.append(count)
                return super().read(count)
        data = b"x" * 1000
        with patch("builtins.open", return_value=FakeReader(data)):
            with self.assertRaises(AudioOutputTooLarge):
                read_audio_limited("synthetic", 16)
        self.assertEqual(called, [17])

    def test_nonpositive_or_noninteger_limit_fails_closed(self):
        for amount in (0, -1, 1.2, True, "16", None):
            with self.subTest(value=amount), self.assertRaises(ValueError):
                read_audio_limited("synthetic", amount)


if __name__ == "__main__":
    unittest.main()
