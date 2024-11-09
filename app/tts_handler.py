import os
import re
import tempfile
import librosa
import logging
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav
)

# Enable debug mode based on environment variable (default: INFO level)
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load the Whisper model and processor for transcription
def load_whisper_model():
    """Loads the Whisper model and processor."""
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to(device)
    return processor, model

processor, whisper_model = load_whisper_model()

# Load the F5-TTS vocoder for audio generation
vocoder = load_vocoder()
logging.info("Vocoder loaded successfully.")

# F5-TTS model configuration and directory setup
F5TTS_MODEL_CONFIG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
CKPTS_DIR = "ckpts/"
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)

def get_model_checkpoint(voice_dir, checkpoint_file="model_1200000.safetensors"):
    """Loads a specified model checkpoint directly, skipping primary checkpoint if unreliable."""
    model_path = os.path.join(voice_dir, checkpoint_file)
    if not os.path.exists(model_path):
        logging.error(f"No valid model checkpoint found at {model_path}")
        raise FileNotFoundError(f"Checkpoint file {model_path} does not exist.")
    logging.debug(f"Selected model checkpoint path: {model_path}")
    return model_path

def load_voice_models():
    """Load F5-TTS models for available voices."""
    voice_mapping = {}
    for folder in os.listdir(CKPTS_DIR):
        voice_dir = os.path.join(CKPTS_DIR, folder)
        if os.path.isdir(voice_dir):
            try:
                # Directly target "model_1200000.safetensors"
                model_path = get_model_checkpoint(voice_dir)
                voice_mapping[folder] = load_model(DiT, F5TTS_MODEL_CONFIG, model_path)
                logging.info(f"Loaded {folder} model from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load model for {folder}: {e}")
    return voice_mapping

voice_mapping = load_voice_models()

# Load default 'Emilia' model if not available
if 'Emilia' not in voice_mapping:
    try:
        emilia_model_path = get_model_checkpoint(os.path.join(CKPTS_DIR, "F5-TTS", "F5TTS_Base"))
        voice_mapping['Emilia'] = load_model(DiT, F5TTS_MODEL_CONFIG, emilia_model_path)
        logging.info(f"Loaded default Emilia model from {emilia_model_path}")
    except FileNotFoundError as e:
        logging.error(f"Default Emilia model could not be loaded: {e}")

def transcribe_audio(ref_audio_path):
    """Transcribes the audio file at the given path using Whisper."""
    logging.info("Transcribing reference audio...")
    audio_input, sample_rate = sf.read(ref_audio_path)
    if sample_rate != 16000:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)

    # Process audio as input features for Whisper
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    with torch.no_grad():
        generated_ids = whisper_model.generate(inputs)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logging.debug(f"Transcription result: {transcription}")
    return transcription

def infer(ref_audio=None, ref_text=None, gen_text=None, model=None):
    """Generates speech using F5-TTS based on provided text or transcribed reference audio."""
    ref_audio = ref_audio or REF_AUDIO_PATH
    if not ref_text and ref_audio:
        logging.info("No reference text provided, transcribing reference audio...")
        ref_text = transcribe_audio(ref_audio)

    try:
        logging.debug(f"ref_audio: {ref_audio}, gen_text: {gen_text}")
        if model is None:
            raise ValueError("Model is None.")
        if vocoder is None:
            raise ValueError("Vocoder is None.")
        if not gen_text or not isinstance(gen_text, str):
            raise ValueError("gen_text must be a non-empty string.")

        if ref_audio and os.path.exists(ref_audio):
            logging.info(f"Using ref_audio: {ref_audio}")
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio, ref_text, gen_text, model, vocoder, cross_fade_duration=0.15, speed=1.0
            )
        else:
            logging.info("Proceeding without reference audio.")
            final_wave, final_sample_rate, _ = infer_process(
                None, ref_text, gen_text, model, vocoder, cross_fade_duration=0.15, speed=1.0
            )
        logging.debug(f"Generated waveform shape: {final_wave.shape}")
        return final_sample_rate, final_wave.squeeze().cpu().numpy() if isinstance(final_wave, torch.Tensor) else final_wave.squeeze()
    except Exception as e:
        logging.error(f"Error during inference process: {e}")
        raise RuntimeError("Failed to generate speech.")

def generate_speech(text, voice='Emilia', response_format='mp3', speed=1.0, ref_audio=None):
    """Generates and saves speech audio from text or reference audio."""
    if not text and ref_audio:
        text = transcribe_audio(ref_audio)
    if not text:
        raise ValueError("No text available for TTS generation.")
    model = voice_mapping.get(voice, voice_mapping['Emilia'])
    sample_rate, audio_data = infer(ref_audio, "", text, model)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
    try:
        sf.write(temp_file.name, audio_data, sample_rate)
        logging.info(f"Generated speech saved to {temp_file.name}")
    except Exception as e:
        logging.error(f"Error saving audio file: {e}")
        raise RuntimeError("Failed to save generated audio.")
    return temp_file.name

def get_models():
    """Returns available F5-TTS models."""
    return [{"id": f"f5-tts-{voice.lower()}", "name": f"F5-TTS {voice}"} for voice in voice_mapping]

def get_voices(language=None):
    """Returns available voices, optionally filtered by language."""
    return [{"name": voice} for voice in voice_mapping.keys()]
