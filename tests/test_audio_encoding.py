"""Execute the real audio encoder method with fake tiny audio and fake FFmpeg.

No model, FFmpeg process, network, inference or Flask imports.
"""
import ast
import json
import shutil
import wave
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ast.parse((ROOT / "app" / "tts_handler.py").read_text(encoding="utf-8"))
HANDLER = next(n for n in MODULE.body if isinstance(n, ast.ClassDef) and n.name == "TTSHandler")
ENCODE = next(n for n in HANDLER.body if isinstance(n, ast.FunctionDef) and n.name == "_write_audio_file")
PROFILES = ast.literal_eval(next(n.value for n in ast.parse(
    (ROOT / "app" / "request_validation.py").read_text()).body
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
    t.id == "STRICT_TIMESTAMP_PROFILES" for t in n.targets)))


class _FakeHandler:
    FFMPEG_TIMEOUT_SECONDS = 7

    def _get_ffmpeg_codec(self, name):
        return {"mp3": "libmp3lame", "pcm": "pcm_s16le", "wav": "pcm_s16le",
                "opus": "libopus"}.get(name, name)


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = tempfile.TemporaryDirectory(prefix="f5-synthetic-")
        self.addCleanup(self.temp_root.cleanup)
        self.calls = []
        self.write_fails = False
        self.conversion_fails = False
        self.conversion_times_out = False
        self.run_kwargs = []

        def fake_write(path, data, samplerate):
            Path(path).write_bytes(b"synthetic-wav")
            if self.write_fails:
                raise OSError("fictional write failure")

        def fake_run(args, **kwargs):
            self.calls.append(args)
            self.run_kwargs.append(kwargs)
            Path(args[-1]).write_bytes(b"synthetic-encoded")
            if self.conversion_fails:
                raise subprocess.CalledProcessError(1, args)
            if self.conversion_times_out:
                raise subprocess.TimeoutExpired(args, timeout=kwargs['timeout'])
            return None

        fake_tempfile = types.SimpleNamespace(
            NamedTemporaryFile=lambda **kw: tempfile.NamedTemporaryFile(
                dir=self.temp_root.name, **kw
            )
        )
        namespace = {
            "os": os, "logging": logging,
            "tempfile": fake_tempfile,
            "sf": types.SimpleNamespace(write=fake_write),
            "subprocess": types.SimpleNamespace(
                run=fake_run, PIPE=subprocess.PIPE
            ),
            "STRICT_TIMESTAMP_PROFILES": PROFILES,
            "AUDIO_FORMAT_MIME_TYPES": {
                "wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "application/octet-stream",
                "opus": "audio/ogg",
            }
        }
        fn = ast.fix_missing_locations(ast.FunctionDef(
            name=ENCODE.name, args=ENCODE.args, body=ENCODE.body,
            decorator_list=[], returns=ENCODE.returns, type_comment=None,
        ))
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[fn], type_ignores=[]
        )), filename="app/tts_handler.py", mode="exec"), namespace)
        self.encode = lambda fmt, profile=None: namespace["_write_audio_file"](
            _FakeHandler(), 16000, [0.0], fmt, encoding_profile=profile
        )

    def files(self):
        return list(Path(self.temp_root.name).iterdir())

    def test_wav_leaves_only_returned_file(self):
        path = self.encode("wav")
        self.assertEqual(self.files(), [Path(path)])
        self.assertEqual(Path(path).read_bytes(), b"synthetic-wav")
        self.assertEqual(self.calls, [])

    def test_mp3_cleans_intermediate_wav(self):
        path = self.encode("mp3")
        self.assertEqual(self.files(), [Path(path)])
        self.assertEqual(Path(path).read_bytes(), b"synthetic-encoded")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.run_kwargs[0]['timeout'], 7)
        self.assertIn("-nostdin", self.calls[0])
        self.assertIn("-loglevel", self.calls[0])
        self.assertIn("-nostats", self.calls[0])

    def test_pcm_uses_explicit_raw_muxer(self):
        path = self.encode("pcm")
        self.assertEqual(self.files(), [Path(path)])
        self.assertEqual(self.calls[0][-3:-1], ["-f", "s16le"])

    def test_strict_encoder_profiles_enforce_codec_rate_channels_and_bitrate(self):
        specs = (
            ("mp3", "mp3_44100_128_strict", "44100", "128000", "mp3"),
            ("opus", "ogg_48000_128_strict", "48000", "128000", "ogg"),
            ("wav", "wav_44100_pcm16_strict", "44100", None, "wav"),
        )
        for fmt, profile, rate, bitrate, muxer in specs:
            with self.subTest(profile=profile):
                self.calls.clear()
                path = self.encode(fmt, profile=profile)
                self.assertEqual(len(self.calls), 1)
                command = self.calls[-1]
                self.assertEqual(command[command.index("-ar") + 1], rate)
                self.assertEqual(command[command.index("-ac") + 1], "1")
                self.assertEqual(command[command.index("-f") + 1], muxer)
                self.assertEqual(command[command.index("-c:a") + 1],
                                 {"mp3": "libmp3lame", "opus": "libopus",
                                  "wav": "pcm_s16le"}[fmt])
                if bitrate:
                    self.assertEqual(command[command.index("-b:a") + 1], bitrate)
                else:
                    self.assertNotIn("-b:a", command)
                if fmt == "opus":
                    self.assertEqual(command[command.index("-vbr") + 1], "off")
                Path(path).unlink()
        self.assertEqual(self.files(), [])

    def test_invalid_or_mismatched_strict_profile_creates_no_files(self):
        for fmt, profile in (("mp3", "bad_profile"),
                             ("wav", "mp3_44100_128_strict")):
            with self.subTest(profile=profile):
                with self.assertRaises(ValueError):
                    self.encode(fmt, profile=profile)
                self.assertEqual(self.files(), [])

    def test_partial_wav_write_is_removed(self):
        self.write_fails = True
        with self.assertRaises(OSError):
            self.encode("wav")
        self.assertEqual(self.files(), [])

    def test_conversion_failure_removes_both_files(self):
        self.conversion_fails = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.encode("mp3")
        self.assertEqual(self.files(), [])

    def test_ffmpeg_timeout_removes_partial_output_and_wav(self):
        self.conversion_times_out = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.encode('mp3')
        self.assertEqual(self.files(), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "Offline FFmpeg and ffprobe not installed")
    def test_real_offline_ffmpeg_strict_profile_metadata(self):
        """Generate a one-second fictional silent WAV; no models or network calls."""
        def write_synthetic_wave(path, audio, rate):
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(rate)
                writer.writeframes(b"\x00\x00" * rate)
        namespace = {
            "os": os, "logging": logging,
            "tempfile": types.SimpleNamespace(NamedTemporaryFile=lambda **kw:
                tempfile.NamedTemporaryFile(dir=self.temp_root.name, **kw)),
            "sf": types.SimpleNamespace(write=write_synthetic_wave),
            "subprocess": subprocess,
            "STRICT_TIMESTAMP_PROFILES": PROFILES,
            "AUDIO_FORMAT_MIME_TYPES": {"wav": "audio/wav", "mp3": "audio/mpeg",
                                        "opus": "audio/ogg", "pcm": "application/octet-stream"},
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ENCODE], type_ignores=[])),
                     "app/tts_handler.py", "exec"), namespace)
        for fmt, profile, codec, rate in (
                ("wav", "wav_44100_pcm16_strict", "pcm_s16le", 44100),
                ("mp3", "mp3_44100_128_strict", "mp3", 44100),
                ("opus", "ogg_48000_128_strict", "opus", 48000)):
            with self.subTest(profile=profile):
                path = namespace["_write_audio_file"](
                    _FakeHandler(), 16000, None, fmt, encoding_profile=profile)
                try:
                    check = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "stream=codec_name,sample_rate,channels,bit_rate",
                         "-of", "json", path], capture_output=True, text=True,
                        check=True, timeout=7)
                    stream = json.loads(check.stdout)["streams"][0]
                    self.assertEqual(stream["codec_name"], codec)
                    self.assertEqual(int(stream["sample_rate"]), rate)
                    self.assertEqual(int(stream["channels"]), 1)
                    if fmt == "mp3":
                        self.assertEqual(int(stream["bit_rate"]), 128000)
                finally:
                    Path(path).unlink()
        self.assertEqual(self.files(), [])

    def test_unknown_format_does_not_create_files(self):
        with self.assertRaises(ValueError):
            self.encode("invalid")
        self.assertEqual(self.files(), [])


if __name__ == "__main__":
    unittest.main()
