# Issue #21: immutable upstream-source handoff (5 November 2026)

**Status: source-only checkpoint, NOT a complete reproducible build.** This file is intentionally preserved for review and migration before a scheduled development-host outage. No model weights, inference, service changes, or Docker builds were used.

## Verified upstream provenance

- Git remote: `https://github.com/SWivid/F5-TTS.git`
- Resolved upstream `HEAD` and `main` at inspection: `600e5fdfbf319f84735557aca80a7aaf60d487e6` (queried via `git ls-remote` on 5 November 2026, Sydney time).
- Upstream `pyproject.toml` at that exact revision reports `f5-tts` version `1.1.22`, with minimum requirements including `torch>=2.0.0`, `torchaudio>=2.0.0` and unrestricted `transformers`. These **are not** an independently resolved or pinned compatibility matrix.
- `requirements.txt` now names the immutable 40-character Git commit, rather than implicitly installing the moving upstream default branch. The duplicate weaker `transformers>=4.46.2` requirement was removed; the existing bounded `transformers>=4.46.2,<5.0.0` remains.

## How to update this source pin

1. Record the intended upstream revision from an authorized source and review its commit history, release provenance, license and dependency changes before promotion.
2. Replace only the 40-character Git SHA in `requirements.txt`; include the upstream commit URL and change rationale in the PR.
3. Run a clean dependency resolution for each intended Python version and CPU/CUDA image architecture, recording exact package versions, index provenance and hashes; verify whether upstream transitive dependencies override constraints. Include a lock/constraints file if compatible and review-tested.
4. In a separate isolated Docker daemon/build root with sufficient reserve space, build from a pinned base-image digest, record the resulting image **digest** and package inventory, and run bounded synthetic acceptance before promoting. Use no real voice checkpoints or reference audio.

## Unmet acceptance gates — mandatory for issue #21

- Python and transitive dependencies are **not yet fully locked**. `constraints/py310-torch25.txt` is a small compatibility candidate, not a hash-locked complete resolution; dynamic transitive dependencies and OS packages remain.
- The Dockerfile now selects one explicit pinned PyTorch/Torchaudio wheel channel, and NVIDIA Compose opts into `cu124`, but **neither flavor has been verified in a clean image build**. Upstream dependency compatibility is also unverified.
- The Python Bookworm base manifest digest has now been pinned (remote manifest inspected only). There is **no measured cold build, final image digest or complete dependency inventory**.
- The full tests, GPU/runtime compatibility and independent security/privacy complete-diff acceptance for a final buildable candidate remain outstanding.
- Operator-provided `/app/ckpts` and `/app/ref_audio` must remain external runtime-mounted data and must not be embedded in any build context or image.

**Do not close issue #21 or squash merge this source-only checkpoint as if all acceptance gates passed.** The hosted CI system is not a required gate, but locally observed clean builds and genuine independent review are.

## 5 November source-only Torch build-contract update

The Dockerfile now pins a Docker Hub Python 3.10 Bookworm manifest-list digest, requires `linux/amd64`, installs `torch==2.5.1+cpu` and `torchaudio==2.5.1+cpu` from the PyTorch CPU wheel index by default, and switches **both** to `+cu124` only when explicitly requested. The NVIDIA Compose file retains its existing NVIDIA runtime and chooses `cu124` as its build argument; it is **not** a CPU-only deployment configuration. A single compatible constraints file applies to the upstream installation, and an installed-version check rejects accidental channel replacement. TorchCodec 0.1.x is constrained based on upstream PyTorch/TorchCodec compatibility guidance.

The exact observed Python image manifest index digest on 5 November was `sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370`; the observed `linux/amd64` child manifest was `sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8`. Public metadata inspection established candidate wheel availability for Python 3.10, but **did not download wheels or run the full resolver**. The chosen TorchCodec/Transformers combination is still a candidate until a real isolated cold build and `pip check` pass. The pinned Python image digest is not the digest of a built F5-TTS image. No GPU inference, NVIDIA runtime or full dependency lock was tested.

`docker build --build-arg TORCH_CHANNEL=cpu` and `docker build --build-arg TORCH_CHANNEL=cu124` are the intended **future** isolated build checks, not commands run for this checkpoint. Do not execute either against the busy shared Ubuntu-Max Docker daemon; preserve a safe storage reserve, an isolated build root, and all operator-provided voice assets.
