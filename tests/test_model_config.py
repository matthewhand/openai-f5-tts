"""Per-checkpoint DiT config: F5-TTS v0 checkpoints (model_1200000) need different
flags than v1 (model_1250000). Loading a v0 checkpoint with v1 defaults yields babble
(verified with Whisper round-trip on real weights)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from model_config import model_config_for  # noqa: E402

V1_BASE = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


def test_v1_checkpoint_uses_v1_defaults(tmp_path):
    assert model_config_for(str(tmp_path / "model_1250000.safetensors")) == V1_BASE


@pytest.mark.parametrize("name", ["model_1200000.safetensors", "model_1200000.pt"])
def test_v0_checkpoint_gets_v0_flags(tmp_path, name):
    cfg = model_config_for(str(tmp_path / name))
    assert cfg == {**V1_BASE, "text_mask_padding": False, "pe_attn_head": 1}


def test_unknown_name_defaults_to_v1(tmp_path):
    assert model_config_for(str(tmp_path / "my_finetune.safetensors")) == V1_BASE


def test_sidecar_json_overrides_naming_convention(tmp_path):
    (tmp_path / "model_config.json").write_text(json.dumps({"text_mask_padding": False, "pe_attn_head": 1}))
    cfg = model_config_for(str(tmp_path / "my_finetune.safetensors"))
    assert cfg == {**V1_BASE, "text_mask_padding": False, "pe_attn_head": 1}
    # sidecar wins even over the filename convention
    (tmp_path / "model_config.json").write_text(json.dumps({"ff_mult": 2}))
    cfg = model_config_for(str(tmp_path / "model_1200000.safetensors"))
    assert cfg == V1_BASE
    # the v0 overrides must be suppressed, not merely equal by accident
    assert "text_mask_padding" not in cfg and "pe_attn_head" not in cfg


def test_bad_sidecar_is_a_clear_error(tmp_path):
    (tmp_path / "model_config.json").write_text("{not json")
    with pytest.raises(ValueError, match="model_config.json"):
        model_config_for(str(tmp_path / "model_1250000.safetensors"))


def test_handler_passes_per_voice_config_to_f5(monkeypatch, tmp_path):
    """load_voice_model must use the checkpoint-specific config, not one global config."""
    import importlib
    from collections import OrderedDict

    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "app"))
    monkeypatch.setenv("API_KEY", "k")
    for n in ("tts_handler", "utils"):
        sys.modules.pop(n, None)
    mod = importlib.import_module("tts_handler")
    root = tmp_path / "ckpts"
    names = {"Emilia": "model_1250000.safetensors", "V0": "model_1200000.safetensors"}
    available = {}
    for voice, name in names.items():
        voice_dir = root / voice
        voice_dir.mkdir(parents=True)
        path = voice_dir / name
        path.write_bytes(b"x")
        available[voice] = str(path)
    seen = {}
    monkeypatch.setattr(mod, "load_model", lambda cls, cfg, path: seen.update({path: cfg}) or "M")
    h = mod.TTSHandler.__new__(mod.TTSHandler)
    h.F5TTS_MODEL_CONFIG = dict(V1_BASE)
    h.CKPTS_DIR = str(root)
    h.MODEL_LOAD_LIMIT = 2
    h.loaded_models = OrderedDict()
    h.available_models = available
    h.load_voice_model("Emilia")
    h.load_voice_model("V0")
    assert seen[available["Emilia"]] == V1_BASE
    assert seen[available["V0"]]["text_mask_padding"] is False
    assert seen[available["V0"]]["pe_attn_head"] == 1
    for n in ("tts_handler", "utils"):
        sys.modules.pop(n, None)
