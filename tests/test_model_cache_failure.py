"""Exercise actual model-cache methods without loading torch, a checkpoint or a vocoder.

The methods are compiled from the committed handler source and run against
tiny stand-in model objects and an injected loader.
"""
import ast
from collections import OrderedDict
import gc
import logging
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / "app" / "tts_handler.py").read_text(encoding="utf-8"))
CLASS = next(n for n in SOURCE.body if isinstance(n, ast.ClassDef) and n.name == "TTSHandler")


def compiled_method(name, namespace):
    method = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method = ast.FunctionDef(
        name=method.name, args=method.args, body=method.body,
        decorator_list=[], returns=method.returns, type_comment=None,
    )
    code = compile(ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[])), "app/tts_handler.py", "exec")
    exec(code, namespace)
    return namespace[name]


class CacheFailureTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.failures = set()
        self.null_returns = set()
        self.models = {}
        def fake_load(model_type, config, path):
            self.calls.append(path)
            if path in self.failures:
                raise RuntimeError("synthetic load failure")
            if path in self.null_returns:
                return None
            return self.models.setdefault(path, object())

        namespace = {
            "load_model": fake_load, "DiT": object(),
            "OrderedDict": OrderedDict, "gc": gc, "logging": logging,
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
        self.load = compiled_method("load_voice_model", namespace)
        self.handler = types.SimpleNamespace(
            loaded_models=OrderedDict(), available_models={"A": "fixture-A", "B": "fixture-B", "C": "fixture-C"},
            MODEL_LOAD_LIMIT=1, F5TTS_MODEL_CONFIG={}, device=types.SimpleNamespace(type="cpu"),
        )

    def test_cache_hit_reuses_object(self):
        a = self.load(self.handler, "A")
        self.assertIs(self.load(self.handler, "A"), a)
        self.assertEqual(self.calls, ["fixture-A"])

    def test_successful_switch_evicts_oldest_before_loading_new(self):
        a = self.load(self.handler, "A")
        b = self.load(self.handler, "B")
        self.assertIsNot(a, b)
        self.assertEqual(list(self.handler.loaded_models), ["B"])
        self.assertEqual(self.calls, ["fixture-A", "fixture-B"])

    def test_failed_replacement_remains_uncached_and_old_voice_can_reload(self):
        self.load(self.handler, "A")
        self.failures.add("fixture-B")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.load(self.handler, "B")
        self.assertEqual(list(self.handler.loaded_models), [])
        self.assertNotIn("B", self.handler.loaded_models)
        self.assertIsNotNone(self.load(self.handler, "A"))
        self.assertEqual(list(self.handler.loaded_models), ["A"])

    def test_loader_returning_none_is_not_cached(self):
        self.null_returns.add("fixture-B")
        with self.assertRaisesRegex(RuntimeError, "returned no model"):
            self.load(self.handler, "B")
        self.assertEqual(list(self.handler.loaded_models), [])

    def test_failed_load_under_capacity_keeps_existing_voice(self):
        self.handler.MODEL_LOAD_LIMIT = 2
        a = self.load(self.handler, "A")
        self.failures.add("fixture-B")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.load(self.handler, "B")
        self.assertEqual(list(self.handler.loaded_models), ["A"])
        self.assertIs(self.handler.loaded_models["A"], a)

    def test_unknown_voice_does_not_evict_cache(self):
        self.load(self.handler, "A")
        with self.assertRaises(ValueError):
            self.load(self.handler, "missing")
        self.assertEqual(list(self.handler.loaded_models), ["A"])

    def test_invalid_limit_fails_before_loading_large_models(self):
        namespace = {
            "torch": types.SimpleNamespace(
                device=lambda name: name,
                cuda=types.SimpleNamespace(is_available=lambda: False),
            ),
            "os": types.SimpleNamespace(
                getenv=lambda name, default: "0" if name == "MODEL_LOAD_LIMIT" else default,
                makedirs=lambda *args, **kwargs: None,
            ),
            "logging": logging,
        }
        initializer = compiled_method("__init__", namespace)
        with self.assertRaisesRegex(ValueError, "MODEL_LOAD_LIMIT"):
            initializer(types.SimpleNamespace())


    def test_invalid_ffmpeg_timeout_fails_before_heavy_initialization(self):
        for value in ("0", "-3"):
            namespace = {
                "torch": types.SimpleNamespace(
                    device=lambda name: name,
                    cuda=types.SimpleNamespace(is_available=lambda: False),
                ),
                "os": types.SimpleNamespace(
                    getenv=lambda name, default: value if name == "FFMPEG_TIMEOUT_SECONDS" else default,
                    makedirs=lambda *args, **kwargs: None,
                ),
                "logging": logging,
            }
            initializer = compiled_method("__init__", namespace)
            with self.subTest(timeout=value), self.assertRaisesRegex(ValueError, "FFMPEG_TIMEOUT_SECONDS"):
                initializer(types.SimpleNamespace())


    def test_failed_cuda_replacement_attempts_cleanup_and_retries(self):
        events = []
        namespace = {
            "load_model": lambda *args: (_ for _ in ()).throw(RuntimeError("fictional loader failure")) if args[-1] == "fixture-B" and self.failures else object(),
            "DiT": object(), "OrderedDict": OrderedDict, "logging": logging,
            "gc": types.SimpleNamespace(collect=lambda: events.append("gc")),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: events.append("cuda"))),
        }
        load = compiled_method("load_voice_model", namespace)
        self.handler.device.type = "cuda"
        load(self.handler, "A")
        self.failures.add("fixture-B")
        with self.assertRaisesRegex(RuntimeError, "fictional loader failure"):
            load(self.handler, "B")
        # One GC/allocator release during LRU eviction, another after failed load.
        self.assertEqual(events, ["gc", "cuda", "gc", "cuda"])
        self.assertEqual(list(self.handler.loaded_models), [])
        self.failures.clear()
        loaded = load(self.handler, "B")
        self.assertIs(self.handler.loaded_models["B"], loaded)

    def test_failed_load_cleanup_exception_preserves_original_failure(self):
        def fail_gc():
            raise OSError("fictional GC cleanup failure")
        namespace = {
            "load_model": lambda *args: (_ for _ in ()).throw(RuntimeError("original model failure")),
            "DiT": object(), "OrderedDict": OrderedDict, "logging": logging,
            "gc": types.SimpleNamespace(collect=fail_gc),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
        load = compiled_method("load_voice_model", namespace)
        with self.assertRaisesRegex(RuntimeError, "original model failure"):
            load(self.handler, "B")
        self.assertEqual(list(self.handler.loaded_models), [])

    def test_none_checkpoint_result_also_attempts_cleanup(self):
        events = []
        namespace = {
            "load_model": lambda *args: None,
            "DiT": object(), "OrderedDict": OrderedDict, "logging": logging,
            "gc": types.SimpleNamespace(collect=lambda: events.append("gc")),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
        load = compiled_method("load_voice_model", namespace)
        with self.assertRaisesRegex(RuntimeError, "returned no model"):
            load(self.handler, "A")
        self.assertEqual(events, ["gc"])
        self.assertEqual(list(self.handler.loaded_models), [])


if __name__ == "__main__":
    unittest.main()
