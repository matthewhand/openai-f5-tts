"""Pure deterministic contention regressions; no models or Flask required."""
import sys
import ast
import logging
from types import SimpleNamespace
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from generation_gate import GenerationGate, SynthesisBusyError


class GenerationGateTests(unittest.TestCase):
    @staticmethod
    def _synthetic_stream(gate, *, fail_model=False, fail_second=False, voice="Emilia", text="One. Two."):
        # Compile the *real* handler method without importing any native models.
        import numpy as np
        path = Path(__file__).resolve().parents[1] / "app" / "tts_handler.py"
        tree = ast.parse(path.read_text())
        klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "TTSHandler")
        method = next(x for x in klass.body
                      if isinstance(x, ast.FunctionDef) and x.name == "generate_speech_stream")
        calls = []
        def infer(*args, **kwargs):
            calls.append("infer")
            if fail_second and calls.count("infer") == 2:
                raise RuntimeError("SYNTHETIC_BACKEND_FAILURE")
            return np.array([0.1, 0.5], dtype=np.float32), 16000, None
        def load(voice):
            calls.append("load")
            if fail_model:
                raise RuntimeError("SYNTHETIC_MODEL_FAILURE")
            return object()
        namespace = {
            "logging": logging, "np": np,
            "torch": SimpleNamespace(Tensor=type("NeverTorch", (), {})),
            "librosa": SimpleNamespace(), "split_sentences": lambda text: text.split("."),
            "infer_process": infer,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(path), "exec"), namespace)
        handler = SimpleNamespace(
            _generation_gate=gate, available_models={voice: object()},
            load_voice_model=load, transcribe_audio=lambda name: "reference",
            process_reference_audio=lambda name: "synthetic.wav", vocoder=object(),
        )
        return namespace["generate_speech_stream"](handler, text, voice), calls

    def test_real_streaming_method_claims_before_loading_and_holds_until_close(self):
        gate = GenerationGate()
        with gate.claim():
            stream, calls = self._synthetic_stream(gate)
            with self.assertRaises(SynthesisBusyError):
                next(stream)
            self.assertEqual(calls, [], "Busy stream must not load/swap any voice")
        stream, calls = self._synthetic_stream(gate)
        self.assertTrue(next(stream))
        self.assertEqual(calls[:2], ["load", "infer"])
        with self.assertRaises(SynthesisBusyError):
            with gate.claim():
                self.fail("Ordinary/timestamp generation overlapped stream")
        stream.close()
        with gate.claim():
            pass

    def test_real_streaming_method_releases_on_finish_and_failures(self):
        gate = GenerationGate()
        for configuration in ({}, {"fail_model": True}, {"fail_second": True}):
            with self.subTest(configuration=configuration):
                stream, _ = self._synthetic_stream(gate, **configuration)
                if configuration:
                    with self.assertRaises(RuntimeError):
                        list(stream)
                else:
                    self.assertEqual(len(list(stream)), 2)
                with gate.claim():
                    pass

    def test_missing_reference_audio_never_logs_raw_exception_path(self):
        """Execute the actual reference-audio error branch without model imports."""
        sentinel = "PRIVATE_AUDIO_PATH_SENTINEL_/hidden/source.wav"
        path = Path(__file__).resolve().parents[1] / "app" / "tts_handler.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "TTSHandler")
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef)
                      and n.name == "process_reference_audio")
        namespace = {"logging": logging, "validate_reference_voice": lambda name: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(path), "exec"), namespace)
        error = FileNotFoundError(sentinel)

        def missing(voice):
            raise error

        handler = SimpleNamespace(get_ref_audio_path=missing)
        with self.assertLogs(level="ERROR") as logs:
            with self.assertRaises(FileNotFoundError) as outcome:
                namespace["process_reference_audio"](handler, "synthetic voice")
        self.assertIs(outcome.exception, error)
        self.assertNotIn(sentinel, " ".join(logs.output))


    def test_streaming_debug_logs_exclude_voice_reference_path_and_text(self):
        gate = GenerationGate()
        voice = "PRIVATE_VOICE_SENTINEL_93"
        text = "PRIVATE_REQUEST_TEXT_SENTINEL_72. Next sentence."
        stream, calls = self._synthetic_stream(gate, voice=voice, text=text)
        with self.assertLogs(level="DEBUG") as logs:
            self.assertEqual(len(list(stream)), 2)
        combined = " ".join(logs.output)
        self.assertNotIn(voice, combined)
        self.assertNotIn(text, combined)
        self.assertIn("Streaming", combined)
        self.assertEqual(calls.count("infer"), 2)
        with gate.claim():
            pass

    def test_reference_audio_lookup_log_does_not_include_path_or_voice(self):
        path = Path(__file__).resolve().parents[1] / "app" / "tts_handler.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TTSHandler")
        method = next(n for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == "get_ref_audio_path")
        root = "PRIVATE_REFERENCE_ROOT_SENTINEL"
        voice = "PRIVATE_VOICE_SENTINEL_83"
        fake_os = SimpleNamespace(path=SimpleNamespace(
            join=lambda *segments: "/".join(segments), lexists=lambda _: True))
        namespace = {
            "os": fake_os, "logging": logging, "INPUT_EXTENSIONS": (".wav",),
            "validate_reference_voice": lambda name: None,
            "validate_reference_root": lambda path: None,
            "validate_reference_input": lambda *args: None,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(path), "exec"), namespace)
        handler = SimpleNamespace(REF_AUDIO_DIR=root)
        with self.assertLogs(level="DEBUG") as logs:
            returned = namespace["get_ref_audio_path"](handler, voice)
        self.assertIn(voice, returned)
        self.assertIn(root, returned)
        self.assertNotIn(voice, " ".join(logs.output))
        self.assertNotIn(root, " ".join(logs.output))

    def test_application_logs_never_interpolate_voice_or_reference_paths(self):
        root = Path(__file__).resolve().parents[1] / "app"
        protected = {
            "voice", "voice_name", "default_voice", "folder", "lru_model_name",
            "ref_audio_path", "cached_audio_path", "processed_audio_path",
        }
        for filename in ("tts_handler.py", "server.py"):
            tree = ast.parse((root / filename).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                if not isinstance(node.func.value, ast.Name) or node.func.value.id != "logging":
                    continue
                if node.func.attr not in {"debug", "info", "warning", "error", "critical", "exception"}:
                    continue
                for arg in node.args:
                    for expr in ast.walk(arg):
                        if isinstance(expr, ast.Name):
                            self.assertNotIn(expr.id, protected,
                                             f"{filename}:{node.lineno} logs private identifier {expr.id}")
                        if isinstance(expr, ast.Attribute):
                            self.assertNotIn(expr.attr, {"REF_AUDIO_CACHE_DIR", "REF_AUDIO_DIR"},
                                             f"{filename}:{node.lineno} logs private directory")
                            if filename == "server.py" and isinstance(expr.value, ast.Name):
                                self.assertFalse(expr.value.id == "request" and expr.attr == "path",
                                                 "Do not log dynamic request paths")
                for keyword in node.keywords:
                    if keyword.arg in {"exc_info", "stack_info"}:
                        self.assertFalse(isinstance(keyword.value, ast.Constant) and
                                         keyword.value.value, f"{filename}:{node.lineno} logs traceback details")

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
