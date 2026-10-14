"""Synthetic filesystem trust checks for source checkpoint paths.

Only tiny fictional files and symlinks under a disposable directory. The tests
extract the actual handler methods via AST; torch and models are never loaded.
"""
import ast
from collections import OrderedDict
import gc
import logging
import os
from pathlib import Path
import tempfile
import types
import unittest

from checkpoint_paths import (
    UntrustedCheckpointPath, validate_checkpoint_path, CHECKPOINT_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / "app" / "tts_handler.py").read_text(encoding="utf-8"))
HANDLER = next(node for node in SOURCE.body if isinstance(node, ast.ClassDef)
               and node.name == "TTSHandler")


def method_from_source(name, namespace):
    node = next(n for n in HANDLER.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[node], type_ignores=[]
    )), "app/tts_handler.py", "exec"), namespace)
    return namespace[name]


class CheckpointPathTrustTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fictional-checkpoint-trust-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpts = self.root / "ckpts"
        self.ckpts.mkdir()
        self.external = self.root / "external"
        self.external.mkdir()
        self.voice = self.ckpts / "Friendly"
        self.voice.mkdir()
        self.file = self.voice / "model_1250000.safetensors"
        self.file.write_bytes(b"fictional-only")
        self.loader_calls = []
        self.handler = types.SimpleNamespace(
            CKPTS_DIR=str(self.ckpts),
            available_models={},
            loaded_models=OrderedDict(),
            MODEL_LOAD_LIMIT=1,
            F5TTS_MODEL_CONFIG={},
            device=types.SimpleNamespace(type="cpu"),
        )

        def fake_loader(_model_type, _config, path):
            self.loader_calls.append(path)
            return object()
        ns = {
            "os": os, "logging": logging, "gc": gc, "load_model": fake_loader,
            "DiT": object(), "torch": types.SimpleNamespace(
                cuda=types.SimpleNamespace(empty_cache=lambda: None)),
            "validate_checkpoint_path": validate_checkpoint_path,
            "CHECKPOINT_NAMES": CHECKPOINT_NAMES,
            "UntrustedCheckpointPath": UntrustedCheckpointPath,
            "validate_checkpoint_root": __import__(
                "checkpoint_paths"
            ).validate_checkpoint_root,
        }
        self.discover = method_from_source("discover_models", ns)
        self.load = method_from_source("load_voice_model", ns)

    def test_valid_native_file_remains_discoverable_and_loadable(self):
        self.discover(self.handler)
        self.assertEqual(self.handler.available_models["Friendly"], str(self.file))
        loaded = self.load(self.handler, "Friendly")
        self.assertIs(self.handler.loaded_models["Friendly"], loaded)
        self.assertEqual(self.loader_calls, [str(self.file)])

    def test_checkpoint_symlink_to_outside_is_not_discovered(self):
        target = self.external / "outside.safetensors"
        target.write_bytes(b"outside fictional")
        self.file.unlink()
        self.file.symlink_to(target)
        self.discover(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_symlink_voice_directory_is_not_discovered(self):
        link = self.ckpts / "Linked"
        link.symlink_to(self.external, target_is_directory=True)
        (self.external / "model_1250000.safetensors").write_bytes(b"external fictional")
        self.discover(self.handler)
        self.assertNotIn("Linked", self.handler.available_models)

    def test_symlink_checkpoint_root_fails_closed(self):
        link = self.root / "link-to-ckpts"
        link.symlink_to(self.ckpts, target_is_directory=True)
        self.handler.CKPTS_DIR = str(link)
        with self.assertRaises(UntrustedCheckpointPath):
            self.discover(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_unexpected_filename_and_escape_are_rejected(self):
        for voice, path in (
            ("Friendly", str(self.external / "outside.safetensors")),
            ("../external", str(self.file)),
            ("Friendly", str(self.file.with_suffix(".pt"))),
            ("Friendly\\Other", str(self.file)),
        ):
            with self.subTest(voice=voice, path=path), self.assertRaises(UntrustedCheckpointPath):
                validate_checkpoint_path(str(self.ckpts), voice, path)

    def test_swap_to_symlink_before_load_preserves_cached_model(self):
        self.discover(self.handler)
        existing = object()
        self.handler.loaded_models["Existing"] = existing
        target = self.external / "replacement.safetensors"
        target.write_bytes(b"fictional replacement")
        self.file.unlink()
        self.file.symlink_to(target)
        with self.assertRaises(UntrustedCheckpointPath):
            self.load(self.handler, "Friendly")
        self.assertEqual(self.loader_calls, [])
        self.assertIs(self.handler.loaded_models["Existing"], existing)

    def test_swap_to_missing_file_before_load_preserves_cache(self):
        self.discover(self.handler)
        existing = object()
        self.handler.loaded_models["Existing"] = existing
        self.file.unlink()
        with self.assertRaises(UntrustedCheckpointPath):
            self.load(self.handler, "Friendly")
        self.assertEqual(self.loader_calls, [])
        self.assertIs(self.handler.loaded_models["Existing"], existing)

    def test_hardlink_to_external_is_rejected_when_supported(self):
        source = self.external / "linked-original"
        source.write_bytes(b"fictional hardlink")
        self.file.unlink()
        try:
            os.link(source, self.file)
        except OSError:
            self.skipTest("Hard links unsupported in disposable filesystem")
        self.discover(self.handler)
        self.assertEqual(self.handler.available_models, {})


    def test_symlinked_ancestor_to_checkpoint_root_fails_closed(self):
        alias = self.root / "ancestor-link"
        alias.symlink_to(self.root, target_is_directory=True)
        self.handler.CKPTS_DIR = str(alias / "ckpts")
        with self.assertRaises(UntrustedCheckpointPath):
            self.discover(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_swap_voice_directory_to_symlink_preserves_cached_model(self):
        self.discover(self.handler)
        previous = object()
        self.handler.loaded_models["Previous"] = previous
        self.file.unlink()
        self.voice.rmdir()
        (self.external / "model_1250000.safetensors").write_bytes(b"fictional external")
        self.voice.symlink_to(self.external, target_is_directory=True)
        with self.assertRaises(UntrustedCheckpointPath):
            self.load(self.handler, "Friendly")
        self.assertEqual(self.loader_calls, [])
        self.assertIs(self.handler.loaded_models["Previous"], previous)

    def test_symlink_to_internal_checkpoint_is_still_rejected(self):
        original = self.voice / "trusted-copy.safetensors"
        original.write_bytes(b"fictional internal")
        self.file.unlink()
        self.file.symlink_to(original)
        self.discover(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_root_readme_mentions_symlink_and_swap_limitations(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("symlink", readme.lower())
        self.assertIn("replacement races", readme)
        self.assertIn("92 tests passed", readme)


if __name__ == "__main__":
    unittest.main()
