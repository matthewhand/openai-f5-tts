"""DiT hyper-parameters per checkpoint.

F5-TTS v0 checkpoints (model_1200000.*) were trained with text_mask_padding=False and
pe_attn_head=1; v1 checkpoints (model_1250000.*) use the v1 defaults. Loading a v0
checkpoint with v1 settings produces plausible-sounding babble, not an error.

Override per voice by placing a `model_config.json` next to the checkpoint; its keys
are merged over the v1 base config (and win over the filename convention).
"""
import json
import os

V1_CONFIG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
V0_OVERRIDES = dict(text_mask_padding=False, pe_attn_head=1)
SIDECAR_NAME = "model_config.json"


def model_config_for(model_path):
    """Return the DiT kwargs to use when loading `model_path`."""
    sidecar = os.path.join(os.path.dirname(model_path), SIDECAR_NAME)
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                overrides = json.load(fh)
            if not isinstance(overrides, dict):
                raise ValueError("top level must be an object")
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid {SIDECAR_NAME} at {sidecar}: {e}") from e
        return {**V1_CONFIG, **overrides}
    if os.path.basename(model_path).startswith("model_1200000"):
        return {**V1_CONFIG, **V0_OVERRIDES}
    return dict(V1_CONFIG)
