"""No network, model imports or GPU required: bounded request contract regressions."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from request_validation import (
    RequestValidationError,
    parse_speech_request,
    parse_timestamp_request,
)

DEFAULTS = dict(default_voice="Emilia", default_format="mp3", default_speed=1.0)


class SpeechValidationTests(unittest.TestCase):
    def parse(self, data, **overrides):
        return parse_speech_request(data, **{**DEFAULTS, **overrides})

    def test_minimal_request_defaults(self):
        self.assertEqual(self.parse({"input": "hello"}), {
            "text": "hello", "voice": "Emilia", "response_format": "mp3", "speed": 1.0
        })

    def test_valid_voice_speed_and_format(self):
        p = self.parse({"input": "one two", "voice": "MyVoice", "response_format": "WAV", "speed": 1.3})
        self.assertEqual((p["voice"], p["response_format"], p["speed"]), ("MyVoice", "wav", 1.3))

    def test_json_body_and_text_are_restricted(self):
        for invalid in (None, [], "string", 0, {}, {"input": None}, {"input": 3},
                        {"input": " "}, {"input": "a" * 2001}):
            with self.subTest(invalid=str(invalid)[:24]):
                with self.assertRaises(RequestValidationError):
                    self.parse(invalid)

    def test_text_limit_is_configurable(self):
        self.assertEqual(self.parse({"input": "abcd"}, max_text_chars=4)["text"], "abcd")
        with self.assertRaises(RequestValidationError):
            self.parse({"input": "abcde"}, max_text_chars=4)

    def test_reject_format_injection_and_voice_types(self):
        for data in ({"input": "hi", "response_format": "mp3; rm -rf /"},
                     {"input": "hi", "response_format": 12},
                     {"input": "hi", "voice": ["one"]},
                     {"input": "hi", "voice": "x" * 129}):
            with self.subTest(data=data):
                with self.assertRaises(RequestValidationError):
                    self.parse(data)

    def test_speed_rejects_nan_infinity_zero_bool_and_out_of_range(self):
        for speed in (float("nan"), float("inf"), -float("inf"), 0, 0.24, 4.01,
                      True, False, {}, "fast", None):
            with self.subTest(speed=speed):
                with self.assertRaises(RequestValidationError):
                    self.parse({"input": "hello", "speed": speed})

    def test_speed_boundary_and_supported_formats(self):
        for speed in (0.25, 4.0):
            self.assertEqual(self.parse({"input": "ok", "speed": speed})["speed"], speed)
        for fmt in ("mp3", "wav", "flac", "aac", "opus", "pcm"):
            self.assertEqual(self.parse({"input": "ok", "response_format": fmt})["response_format"], fmt)

    def test_timestamp_request_is_strict_and_maps_formats(self):
        for specified, expected in (("wav_44100_128", "wav"),
                                    ("mp3_44100_128", "mp3"),
                                    ("ogg_48000_128", "opus"),
                                    ("flac", "flac")):
            with self.subTest(fmt=specified):
                p = parse_timestamp_request({"text": "hello", "voice_settings": {"speed": 1.25}},
                         voice="Emilia", output_format=specified, default_format="mp3", default_speed=1.0)
                self.assertEqual((p["response_format"], p["speed"]), (expected, 1.25))

    def test_timestamp_bad_formats_and_settings(self):
        for data, fmt in (({"text": "hello", "voice_settings": []}, None),
                          ({"text": "hello"}, "unknown"),
                          ({"text": ""}, None)):
            with self.subTest(data=data, fmt=fmt):
                with self.assertRaises(RequestValidationError):
                    parse_timestamp_request(data, voice="Emilia", output_format=fmt,
                                            default_format="mp3", default_speed=1.0)


if __name__ == "__main__":
    unittest.main()
