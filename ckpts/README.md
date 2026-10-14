# Checkpoint directory (experimental)

Put a trusted, licensed, **native safetensors** model in `ckpts/<voice>/`:

- Preferred filename: `model_1250000.safetensors`
- Fallback filename: `model_1200000.safetensors`

The API intentionally does not discover or automatically convert legacy `.pt`
checkpoints, which may use Python pickle. Obtain an actual safetensors export
from your trusted offline training environment; renaming a `.pt` file is not
conversion. This repository does not validate arbitrary checkpoint provenance.

For a pinned reference checkpoint URL, license and integrity hash, see the
root [`README.md`](../README.md). Do not clone the entire multi-gigabyte model
repository by default, and do not place untrusted files here.

Only store trusted checkpoints in a non-symlinked directory tree. The API
rejects symlinked roots, voice folders, checkpoint files and multiply linked
checkpoint files. It rechecks discovered paths before replacing cached models,
but cannot eliminate concurrent replacement races in the upstream pathname-only
loader. Do not give untrusted users write access to the checkpoint directory.
