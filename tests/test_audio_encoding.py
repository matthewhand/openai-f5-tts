"""Execute the real audio encoder method with fake tiny audio and fake FFmpeg.

No model, FFmpeg process, network, inference or Flask imports.
"""
import ast
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


class _FakeHandler:
    FFMPEG_TIMEOUT_SECONDS = 7

    def _get_ffmpeg_codec(self, name):
        return {"mp3": "libmp3lame", "pcm": "pcm_s16le", "wav": "pcm_s16le"}.get(name, name)


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
            "AUDIO_FORMAT_MIME_TYPES": {
                "wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"
            }
        }
        fn = ast.fix_missing_locations(ast.FunctionDef(
            name=ENCODE.name, args=ENCODE.args, body=ENCODE.body,
            decorator_list=[], returns=ENCODE.returns, type_comment=None,
        ))
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[fn], type_ignores=[]
        )), filename="app/tts_handler.py", mode="exec"), namespace)
        self.encode = lambda fmt: namespace["_write_audio_file"](
            _FakeHandler(), 16000, [0.0], fmt
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

    def test_pcm_uses_explicit_raw_muxer(self):
        path = self.encode("pcm")
        self.assertEqual(self.files(), [Path(path)])
        self.assertEqual(self.calls[0][-3:-1], ["-f", "s16le"])

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

    def test_unknown_format_does_not_create_files(self):
        with self.assertRaises(ValueError):
            self.encode("invalid")
        self.assertEqual(self.files(), [])


if __name__ == "__main__":
    unittest.main()
