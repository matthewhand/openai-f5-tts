"""Checkpoint discovery contract without PyTorch, models or network access.

Execute the actual method extracted from the TTSHandler source AST. Only local
synthetic file names are used, never a checkpoint or model import.
"""
import ast
import logging
import os
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
HANDLER_FILE = ROOT / "app" / "tts_handler.py"


def discover_from_source(handler):
    module = ast.parse(HANDLER_FILE.read_text(encoding="utf-8"))
    cls = next(node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == "TTSHandler")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "discover_models")
    method.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[])), str(HANDLER_FILE), "exec")
    scope = {"os": os, "logging": logging}
    exec(code, scope)
    return scope["discover_models"](handler)


class CheckpointDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="f5-ckpt-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.handler = types.SimpleNamespace(CKPTS_DIR=str(self.root), available_models={})

    def add_voice(self, name, *files):
        folder = self.root / name
        folder.mkdir()
        for filename in files:
            (folder / filename).write_bytes(b"synthetic name only")
        return folder

    def test_discover_native_safetensors_priority(self):
        voice = self.add_voice("Native", "model_1200000.safetensors",
                               "model_1250000.safetensors")
        discover_from_source(self.handler)
        self.assertEqual(self.handler.available_models,
                         {"Native": str(voice / "model_1250000.safetensors")})

    def test_legacy_pt_checkpoint_not_discovered(self):
        self.add_voice("Legacy", "model_1250000.pt", "model_1200000.pt")
        discover_from_source(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_safe_fallback_not_shadowed_by_legacy_pt(self):
        voice = self.add_voice("Fallback", "model_1250000.pt",
                               "model_1200000.safetensors")
        discover_from_source(self.handler)
        self.assertEqual(self.handler.available_models,
                         {"Fallback": str(voice / "model_1200000.safetensors")})

    def test_checkpoint_candidate_must_be_regular_file(self):
        voice = self.add_voice("Directory")
        (voice / "model_1250000.safetensors").mkdir()
        discover_from_source(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_unrelated_content_not_discovered(self):
        self.add_voice("Other", "model_unknown.safetensors", "notes.txt")
        discover_from_source(self.handler)
        self.assertEqual(self.handler.available_models, {})

    def test_wrapper_has_no_pickle_checkpoint_loader(self):
        module = ast.parse(HANDLER_FILE.read_text(encoding="utf-8"))
        torch_load = [
            node for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch" and node.func.attr == "load"
        ]
        self.assertEqual(torch_load, [], "Wrapper should not deserialize .pt with torch.load")


class CheckpointDocumentationTests(unittest.TestCase):
    def test_fine_tune_docs_match_native_only_discovery(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("model_1200000.safetensors", readme)
        self.assertIn("no longer discovers `.pt` files", readme)
        self.assertNotIn("automatically be converted to `.safetensors`", readme)
        self.assertNotIn("move path_to_your_checkpoint\\model_1200000.pt", readme)

    def test_checkpoint_directory_does_not_recommend_full_clone(self):
        instructions = (ROOT / "ckpts" / "README.md").read_text(encoding="utf-8")
        self.assertIn("native safetensors", instructions)
        self.assertIn("does not discover or automatically convert legacy", instructions)
        self.assertNotIn("git clone https://huggingface.co", instructions)


if __name__ == "__main__":
    unittest.main()
