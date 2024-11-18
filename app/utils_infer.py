import torch
import logging
from safetensors.torch import load_file

def load_model(model_class, config, model_path):
    """
    Load a model from the given checkpoint path with specified configurations.
    Handles both `.pt` and `.safetensors`.

    Args:
        model_class: The model class to instantiate (e.g., DiT).
        config (dict): Configuration dictionary to initialize the model.
        model_path (str): Path to the model checkpoint file.

    Returns:
        model: The loaded model with weights from the checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model from {model_path} on {device}...")

    # Initialize model with the given configuration
    model = model_class(**config)

    try:
        if model_path.endswith(".safetensors"):
            # Load `.safetensors` format
            checkpoint = load_file(model_path)
            logging.debug(f"Loaded .safetensors file: {model_path}")
        else:
            # Load `.pt` format
            checkpoint = torch.load(model_path, map_location=device)
            logging.debug(f"Loaded .pt file: {model_path}")

        # Check if the checkpoint contains `state_dict`
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            logging.debug("Loaded 'state_dict' into the model.")
        else:
            model.load_state_dict(checkpoint)
            logging.debug("Loaded checkpoint directly into the model.")

    except Exception as e:
        if model_path.endswith(".safetensors"):
            logging.error(f"Error loading `.safetensors` file from {model_path}: {e}")
        else:
            logging.error(f"Error loading `.pt` file from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from checkpoint at {model_path}")

    # Move the model to the appropriate device
    model.to(device)
    logging.info(f"Model loaded and moved to {device}.")

    return model

def load_vocoder(vocoder_name="vocos", is_local=True, local_path=None):
    if vocoder_name == "vocos":
        if is_local:
            # Load local vocoder
            logging.info(f"Loading VOCOS vocoder from local path: {local_path}")
            return VocoderClass.from_pretrained(local_path)
        else:
            # Load vocoder from external resource
            return VocoderClass.from_pretrained("charactr/vocos-mel-24khz")
    elif vocoder_name == "bigvgan":
        if is_local:
            logging.info(f"Loading BigVGAN vocoder from local path: {local_path}")
            return VocoderClass.from_pretrained(local_path)
        else:
            return VocoderClass.from_pretrained("charactr/bigvgan_v2_24khz_100band_256x")
    else:
        raise ValueError(f"Unsupported vocoder name: {vocoder_name}")
