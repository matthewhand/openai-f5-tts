"""Exercise actual reference-audio FFmpeg call sites with zero external processes."""
import ast
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from reference_paths import (
    validate_reference_voice, validate_reference_root, validate_reference_input,
    validate_reference_cache, require_missing_cache_intermediate,
)


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "app" / "tts_handler.py").read_text(encoding="utf-8"))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "TTSHandler")
METHOD = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == "process_reference_audio")


class ReferenceTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="f5-ref-ffmpeg-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ref = self.root / "Emilia.wav"
        self.ref.write_bytes(b"fictional input")
        self.cache = self.root / "Emilia_processed.wav"
        self.pcm = self.root / "Emilia_processed.pcm"
        self.calls = []
        self.fail_call = None
        self.corrupt_cache = False

        def fake_run(args, **kwargs):
            self.calls.append(kwargs)
            self.assertEqual(kwargs["timeout"], 5)
            self.assertIn("-nostdin", args)
            self.assertIn("-hide_banner", args)
            self.assertIn("-loglevel", args)
            self.assertIn("-nostats", args)
            Path(args[-1]).write_bytes(b"partial fictitious audio")
            if len(self.calls) == self.fail_call:
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])

        def fake_read(path):
            if self.corrupt_cache:
                raise OSError("fake corrupt reference")
            return ([0.0], 16000)

        module = ast.fix_missing_locations(ast.Module(
            body=[ast.FunctionDef(
                name=METHOD.name, args=METHOD.args, body=METHOD.body,
                decorator_list=[], returns=METHOD.returns, type_comment=None
            )], type_ignores=[]
        ))
        namespace = {
            "os": os, "logging": logging,
            "validate_reference_voice": validate_reference_voice,
            "validate_reference_root": validate_reference_root,
            "validate_reference_input": validate_reference_input,
            "validate_reference_cache": validate_reference_cache,
            "require_missing_cache_intermediate": require_missing_cache_intermediate,
            "subprocess": types.SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE),
            "sf": types.SimpleNamespace(read=fake_read),
            "librosa": types.SimpleNamespace(resample=lambda data, **kwargs: data),
        }
        exec(compile(module, "app/tts_handler.py", "exec"), namespace)
        self.run_method = namespace["process_reference_audio"]
        self.handler = types.SimpleNamespace(
            get_ref_audio_path=lambda name: str(self.ref),
            REF_AUDIO_CACHE_DIR=str(self.root),
            REF_AUDIO_DIR=str(self.root),
            REF_AUDIO_LIMIT_SECONDS=20,
            FFMPEG_TIMEOUT_SECONDS=5,
            disable_pcm_normalization=False,
        )

    def run_ref(self):
        return self.run_method(self.handler, "Emilia")

    def test_success_calls_all_three_ffmpeg_steps_with_timeouts(self):
        self.assertEqual(self.run_ref(), str(self.cache))
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(self.cache.exists())
        self.assertFalse(self.pcm.exists())

    def test_timeout_first_conversion_discards_partial_cache(self):
        self.fail_call = 1
        with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
            self.run_ref()
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.pcm.exists())

    def test_timeout_second_conversion_discards_partial_wav_and_pcm(self):
        self.fail_call = 2
        with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
            self.run_ref()
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.pcm.exists())

    def test_timeout_third_conversion_discards_partial_wav_and_pcm(self):
        self.fail_call = 3
        with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
            self.run_ref()
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.pcm.exists())

    def test_corrupt_existing_cache_is_invalidated_for_later_retry(self):
        self.cache.write_bytes(b"corrupt")
        self.corrupt_cache = True
        with self.assertRaisesRegex(RuntimeError, "verify"):
            self.run_ref()
        self.assertFalse(self.cache.exists())
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
