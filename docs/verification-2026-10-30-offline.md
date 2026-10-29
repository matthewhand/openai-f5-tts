# F5-TTS offline native acceptance — 30 October 2026

This record covers the **source tree at `76630dfff2576387fee8950c17411b2d7009f341`** plus only the synthetic Gevent lifecycle test added for issue #20. It is **not** a replacement for the separately dated 20 October 2026 model-backed Docker verification report.

## Environment and provenance

- Environment: isolated task-owned Python virtual environment on Linux ARM64 (`aarch64`), Python 3.10.12, Ubuntu 22.04 userland and Linux 6.8 kernel. No Docker build was run.
- Dependencies: Torch 2.5.1 **CPU** (`torch.cuda.is_available() == False`), Gevent 26.9.0, greenlet 3.5.6, Flask 2.3.2, pytest 9.1.1, NumPy 2.2.6, SciPy 1.15.3, SoundFile 0.14.0, librosa 0.11.0, python-dotenv 1.0.0. Installed from already-cached distributions without fetching model weights.
- F5-TTS/Whisper/vocoder model startup was replaced with **fictional fixtures**; the real application stream route and real Gevent WSGI server were exercised only on the ephemeral IPv4 loopback interface. No model-backed synthesis, NVIDIA device, deployment, hosted CI, third-party egress, or real reference-audio asset was used.
- Audio and timestamp regressions remained synthetic or local FFmpeg-based, not GPU inference. Prior historical checks in `docs/verification-2026-10-20.md` retain their original date and scope.

## Bounded commands and results

Environment exports: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. In the isolated venv, run `timeout 100s python -B -m pytest -q -rs --tb=short -p no:cacheprovider tests` from the repository root.

| Test environment | Result | Native coverage |
|---|---|---|
| Isolated ARM64 environment with real Gevent 26.9.0 and CPU Torch 2.5.1 | **235 passed; 210 subtests passed; zero skips**, 12.16 seconds on the independently reviewed remediation candidate | Real Gevent localhost streaming, early/uniterated WSGI close, repeated close, fictional client disconnect and access-log redaction; real CPU Torch with synthetic upstream model boundaries |
| Minimal host Python without Gevent or Torch | **208 passed; 210 subtests passed; five expected skips**, 4.06 seconds | Source, Flask test-client, FFmpeg and synthetic contracts only; no native acceptance claimed |

Four new test functions (eight cases across production and debug log configurations) are in `tests/test_gevent_wsgi_lifecycle.py`: early close of the Flask WSGI iterable, early close via a real Gevent server, successful streamed fictional PCM, and disconnect while asserting the retained generation gate and private-log suppression. Request-greenlet events are asserted from the controlling test thread, and the fake backend is explicitly cleaned up between tests. The earlier `tests/test_gevent_native_privacy.py` additionally tests a real Gevent fictional localhost HTTP request.

## Remaining boundaries

- **No live model inference**, CUDA/GPU memory exhaustion/recovery, multi-process admission or model-backed concurrent request testing was performed. The synthesis gate remains process-local, not a cluster lock.
- A closed HTTP connection is not a guarantee of instant cancellation while an uninterruptible native inference call is active. The tests use bounded fictional chunks and cannot prove network/proxy behavior beyond local Gevent WSGI.
- HTTP logging checks cover the configured Gevent access/error sinks and sanitized application messages, **not** native file-descriptor writes, unrelated third-party loggers, load balancers or reverse proxies.
- These results neither verify the current unpinned upstream F5-TTS dependency cold install nor establish a reproducible Docker build (tracked separately by issue #21).

Production promotion still requires independent model-backed, privacy, licensing and deployment assessment. Do not extrapolate these offline results to arbitrary GPUs, OS versions or external networks.
