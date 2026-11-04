"""Offline source guarantees: a fail-closed single-channel PyTorch build."""
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TorchBuildContract(unittest.TestCase):
    def test_base_is_pinned_by_digest_and_python_version(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertRegex(
            dockerfile,
            r"FROM python:3\.10-slim-bookworm@sha256:[0-9a-f]{64} AS base",
        )
        self.assertIn('test "$TARGETARCH" = amd64', dockerfile)

    def test_single_explicit_pytorch_channel_with_no_floating_reinstall(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("ARG TORCH_CHANNEL=cpu", dockerfile)
        self.assertIn("cpu|cu124", dockerfile)
        self.assertIn("torch==2.5.1+$TORCH_CHANNEL", dockerfile)
        self.assertIn("torchaudio==2.5.1+$TORCH_CHANNEL", dockerfile)
        self.assertIn('https://download.pytorch.org/whl/$TORCH_CHANNEL', dockerfile)
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("-c constraints/py310-torch25.txt", dockerfile)
        self.assertIn("python3 -m pip check", dockerfile)
        self.assertNotIn("--extra-index-url", dockerfile)
        self.assertNotRegex(dockerfile, r"pip install[^\n]*\btorch\s+torchaudio\b")

    def test_constraints_pin_torch_audio_codec_and_transformers(self):
        constraints = (ROOT / "constraints/py310-torch25.txt").read_text().splitlines()
        constraints = [x.strip() for x in constraints if x.strip() and not x.lstrip().startswith("#")]
        self.assertEqual(len(constraints), len(set(constraints)))
        for exact in (
            "torch==2.5.1",
            "torchaudio==2.5.1",
            "torchcodec==0.1.1",
            "transformers==4.46.2",
            "tokenizers==0.20.3",
        ):
            self.assertIn(exact, constraints)
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", x)
                            for x in constraints))

    def test_constraints_in_git_checkout(self):
        # A local ignored file can make source tests pass but break a clean Docker build.
        if not (ROOT / ".git").exists() or shutil.which("git") is None:
            self.skipTest("Git index unavailable in source archive")
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", "constraints/py310-torch25.txt"],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr or "constraints not tracked")

    def test_compose_keeps_nvidia_default_channel(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("${TORCH_CHANNEL:-cu124}", compose)
        self.assertIn("runtime: nvidia", compose)
        self.assertIn("127.0.0.1:9090:9090", compose)

    def test_sensitive_inputs_never_copied_into_image(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
        for path in ("ckpts", "ref_audio", ".env"):
            self.assertIn(path, dockerignore)
            self.assertNotRegex(dockerfile, rf"(?m)^COPY\s+{path}(?:/|\s)")
        self.assertIn("COPY app/", dockerfile)
        for pattern in ("**/.env", "**/ckpts/**", "**/ref_audio/**",
                        "**/*.safetensors", "**/*.pt", "**/*.wav", "**/*.pem"):
            self.assertIn(pattern, dockerignore, f"Missing nested context exclusion: {pattern}")


if __name__ == "__main__":
    unittest.main()
