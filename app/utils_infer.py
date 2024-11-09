import torch
import logging

def load_model(model_class, config, model_path):
    """
    Load a model from the given checkpoint path with specified configurations.
    Handles both CPU and GPU environments by using map_location.
    
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
        # Load checkpoint and map it to the appropriate device
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if the loaded checkpoint has a 'state_dict'
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logging.debug(f"Model state dict loaded from checkpoint at {model_path}.")
        else:
            model.load_state_dict(checkpoint)  # Load weights directly if no 'state_dict' key
            logging.debug(f"Checkpoint loaded directly for model at {model_path}.")

    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from checkpoint at {model_path}")

    # Move the model to the appropriate device
    model.to(device)
    logging.info(f"Model loaded and moved to {device}.")
    
    return model

def load_vocoder():
    """
    Dummy function to load the vocoder. Replace with actual vocoder loading code.
    
    Returns:
        vocoder: The loaded vocoder object.
    """
    # Placeholder - Replace with actual vocoder loading logic
    logging.info("Loading vocoder (placeholder function)...")
    vocoder = None  # Implement actual vocoder loading
    return vocoder
