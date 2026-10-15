"""Real source-method checks with tiny fictional audio paths and fake FFmpeg.

No sound decoding, model imports, subprocess launch, network or model input.
"""
import ast
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from reference_paths import (
    UntrustedReferencePath, validate_reference_voice,
    validate_reference_root, validate_reference_input, validate_reference_cache,
    require_missing_cache_intermediate, INPUT_EXTENSIONS,
)

SOURCE = ast.parse((APP / "tts_handler.py").read_text(encoding="utf-8"))
CLASS = next(x for x in SOURCE.body if isinstance(x, ast.ClassDef) and x.name == "TTSHandler")


def extract(name, namespace):
    func = next(x for x in CLASS.body if isinstance(x, ast.FunctionDef) and x.name == name)
    func.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
                 str(APP / "tts_handler.py"), "exec"), namespace)
    return namespace[name]


class ReferencePathTrustTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fictional-reference-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref_dir = self.root / "ref"
        self.cache_dir = self.root / "cache"
        self.ref_dir.mkdir()
        self.cache_dir.mkdir()
        self.external = self.root / "external"
        self.external.mkdir()
        self.source = self.ref_dir / "Emilia.wav"
        self.source.write_bytes(b"tiny fictional sample")
        self.cached = self.cache_dir / "Emilia_processed.wav"
        self.pcm = self.cache_dir / "Emilia_processed.pcm"
        self.calls = []

        def fake_run(args, **kwargs):
            self.calls.append(tuple(args))
            self.assertEqual(kwargs["timeout"], 5)
            Path(args[-1]).write_bytes(b"fictional ffmpeg output")

        ns = {
            "os": os, "logging": logging,
            "subprocess": types.SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE),
            "sf": types.SimpleNamespace(read=lambda path: ([0.0], 16000)),
            "librosa": types.SimpleNamespace(resample=lambda a, **kw: a),
            "validate_reference_voice": validate_reference_voice,
            "validate_reference_root": validate_reference_root,
            "validate_reference_input": validate_reference_input,
            "validate_reference_cache": validate_reference_cache,
            "require_missing_cache_intermediate": require_missing_cache_intermediate,
            "INPUT_EXTENSIONS": INPUT_EXTENSIONS,
        }
        self.lookup = extract("get_ref_audio_path", ns)
        self.process = extract("process_reference_audio", ns)
        self.handler = types.SimpleNamespace(
            REF_AUDIO_DIR=str(self.ref_dir),
            REF_AUDIO_CACHE_DIR=str(self.cache_dir),
            REF_AUDIO_LIMIT_SECONDS=20,
            FFMPEG_TIMEOUT_SECONDS=5,
            disable_pcm_normalization=False,
        )
        self.handler.get_ref_audio_path = lambda voice: self.lookup(self.handler, voice)

    def test_regular_reference_works_and_fake_cache_converts(self):
        self.assertEqual(self.lookup(self.handler, "Emilia"), str(self.source))
        self.assertEqual(self.process(self.handler, "Emilia"), str(self.cached))
        self.assertEqual(len(self.calls), 3)
        self.assertFalse(self.pcm.exists())

    def test_symlinked_reference_input_rejected_before_ffmpeg(self):
        target = self.external / "confidential.wav"
        target.write_bytes(b"fictional secret")
        self.source.unlink()
        self.source.symlink_to(target)
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])
        self.assertEqual(target.read_bytes(), b"fictional secret")

    def test_symlinked_reference_input_inside_root_also_rejected(self):
        target = self.ref_dir / "genuine.wav"
        target.write_bytes(b"fictional")
        self.source.unlink()
        self.source.symlink_to(target)
        with self.assertRaises(UntrustedReferencePath):
            self.lookup(self.handler, "Emilia")

    def test_input_hardlink_rejected(self):
        target = self.external / "confidential.wav"
        target.write_bytes(b"fictional secret")
        self.source.unlink()
        try:
            os.link(target, self.source)
        except OSError:
            self.skipTest("hardlinks unsupported on disposable filesystem")
        with self.assertRaises(UntrustedReferencePath):
            self.lookup(self.handler, "Emilia")

    def test_reference_symlink_root_and_ancestor_rejected(self):
        root_link = self.root / "linked-ref"
        root_link.symlink_to(self.ref_dir, target_is_directory=True)
        self.handler.REF_AUDIO_DIR = str(root_link)
        with self.assertRaises(UntrustedReferencePath):
            self.lookup(self.handler, "Emilia")
        ancestor = self.root / "alias"
        ancestor.symlink_to(self.root, target_is_directory=True)
        self.handler.REF_AUDIO_DIR = str(ancestor / "ref")
        with self.assertRaises(UntrustedReferencePath):
            self.lookup(self.handler, "Emilia")

    def test_voice_escape_and_unexpected_format_rejected(self):
        for voice in ("../external", "Emilia/../../outside", "Emilia\\Other", ".", ""):
            with self.subTest(voice=voice), self.assertRaises(UntrustedReferencePath):
                self.lookup(self.handler, voice)
        with self.assertRaises(UntrustedReferencePath):
            validate_reference_input(str(self.ref_dir), "Emilia", str(self.external / "x.wav"))

    def test_symlinked_cache_wav_refused_before_processing_or_read(self):
        target = self.external / "victim.wav"
        target.write_bytes(b"do not overwrite")
        self.cached.symlink_to(target)
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])
        self.assertEqual(target.read_bytes(), b"do not overwrite")

    def test_dangling_cached_wav_symlink_refused(self):
        self.cached.symlink_to(self.external / "absent.wav")
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])

    def test_preexisting_pcm_cache_symlink_does_not_get_unlinked(self):
        victim = self.external / "victim.pcm"
        victim.write_bytes(b"fictional secret")
        self.pcm.symlink_to(victim)
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])
        self.assertTrue(self.pcm.is_symlink())
        self.assertEqual(victim.read_bytes(), b"fictional secret")

    def test_preexisting_pcm_regular_file_refused_as_unowned(self):
        self.pcm.write_bytes(b"not owned by this run")
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.pcm.read_bytes(), b"not owned by this run")

    def test_symlink_cache_directory_rejected(self):
        link = self.root / "linked-cache"
        link.symlink_to(self.cache_dir, target_is_directory=True)
        self.handler.REF_AUDIO_CACHE_DIR = str(link)
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])

    def test_input_swapped_after_lookup_is_rejected_before_ffmpeg(self):
        victim = self.external / "victim.wav"
        victim.write_bytes(b"fictional secret")
        def fake_lookup(_voice):
            self.source.unlink()
            self.source.symlink_to(victim)
            return str(self.source)
        self.handler.get_ref_audio_path = fake_lookup
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])

    def test_existing_valid_cache_is_read_without_ffmpeg(self):
        self.cached.write_bytes(b"fictional cached audio")
        self.assertEqual(self.process(self.handler, "Emilia"), str(self.cached))
        self.assertEqual(self.calls, [])


    def test_cache_wav_hardlink_rejected_before_ffmpeg(self):
        victim = self.external / "external-hardlink.wav"
        victim.write_bytes(b"unchanged fictional content")
        try:
            os.link(victim, self.cached)
        except OSError:
            self.skipTest("hardlinks unsupported in synthetic directory")
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])
        self.assertEqual(victim.read_bytes(), b"unchanged fictional content")

    def test_reference_input_directory_rejected(self):
        self.source.unlink()
        self.source.mkdir()
        with self.assertRaises(UntrustedReferencePath):
            self.lookup(self.handler, "Emilia")

    def test_symlinked_ancestor_of_cache_rejected(self):
        ancestor = self.root / "cache-alias"
        ancestor.symlink_to(self.root, target_is_directory=True)
        self.handler.REF_AUDIO_CACHE_DIR = str(ancestor / "cache")
        with self.assertRaises(UntrustedReferencePath):
            self.process(self.handler, "Emilia")
        self.assertEqual(self.calls, [])

    def test_whisper_revalidates_cache_after_processing(self):
        decoded = []
        scope = {
            "validate_reference_cache": validate_reference_cache,
            "sf": types.SimpleNamespace(read=lambda path: decoded.append(path)),
            "logging": logging,
        }
        transcribe = extract("transcribe_audio", scope)
        victim = self.external / "file.wav"
        victim.write_bytes(b"fictional-only")
        self.cached.symlink_to(victim)
        self.handler.process_reference_audio = lambda voice: str(self.cached)
        with self.assertRaises(UntrustedReferencePath):
            transcribe(self.handler, "Emilia")
        self.assertEqual(decoded, [])


    def test_operator_documentation_discloses_cache_and_rights_limitations(self):
        readme = (APP.parent / "README.md").read_text(encoding="utf-8")
        guidance = (APP.parent / "ref_audio" / "README.md").read_text(encoding="utf-8")
        self.assertIn("110 tests passed", readme)
        self.assertIn("concurrent pathname swap", readme)
        self.assertIn("consent", guidance)
        self.assertIn("processed.pcm", guidance)


if __name__ == "__main__":
    unittest.main()
