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

- Python and transitive dependencies are **not yet locked**. The Dockerfile still upgrades pip and installs packages from dynamic indexes.
- Existing `Dockerfile`/`docker-compose.yml` use NVIDIA/CUDA-oriented settings and a second unpinned Torch/Torchaudio install from an extra index, which can replace the resolver's earlier selection. CPU and CUDA compatibility is **unverified**.
- The Docker base tag is not pinned to an image digest; no authenticated, clean cold build, output image digest or dependency inventory has been measured.
- The full tests, GPU/runtime compatibility and independent security/privacy complete-diff acceptance for a final buildable candidate remain outstanding.
- Operator-provided `/app/ckpts` and `/app/ref_audio` must remain external runtime-mounted data and must not be embedded in any build context or image.

**Do not close issue #21 or squash merge this source-only checkpoint as if all acceptance gates passed.** The hosted CI system is not a required gate, but locally observed clean builds and genuine independent review are.
