import os
import shutil
import tempfile
import librosa
import logging
import numpy as np
import soundfile as sf
import torch
import subprocess
import gc  # Import garbage collection
from collections import OrderedDict
from safetensors.torch import save_file, load_file
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav
)
from utils import AUDIO_FORMAT_MIME_TYPES

# Initialize logging
logging.basicConfig(level=logging.INFO)

class TTSHandler:
    def __init__(self, retain_cache=False, disable_pcm_normalization=False, default_voice='Emilia'):
        """
        Initialize the TTS Handler with cache and normalization settings.

        Args:
            retain_cache (bool): Whether to retain the reference audio cache on startup.
            disable_pcm_normalization (bool): Whether to disable PCM normalization during audio processing.
            default_voice (str): The default voice to load first.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.F5TTS_MODEL_CONFIG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        self.CKPTS_DIR = "ckpts/"
        self.REF_AUDIO_DIR = "ref_audio/"  # Directory containing reference audio files

        self.MODEL_LOAD_LIMIT = int(os.getenv("MODEL_LOAD_LIMIT", "1"))
        self.REF_AUDIO_LIMIT_SECONDS = int(os.getenv("REF_AUDIO_LIMIT_SECONDS", "20"))
        self.REF_AUDIO_CACHE_DIR = "ref_audio_cache"
        os.makedirs(self.REF_AUDIO_CACHE_DIR, exist_ok=True)

        self.retain_cache = retain_cache
        self.disable_pcm_normalization = disable_pcm_normalization
        self.default_voice = default_voice

        self.cleanup_cache()
        self.processor, self.whisper_model = self.load_whisper_model()
        self.vocoder = load_vocoder()
        if self.vocoder:
            logging.info("Vocoder loaded successfully.")
        else:
            logging.error("Vocoder failed to load. Ensure that load_vocoder is properly implemented.")
            raise RuntimeError("Failed to load vocoder.")

        self.available_models = {}
        self.loaded_models = OrderedDict()
        self.discover_models()
        self.load_default_model()

    def cleanup_cache(self):
        """
        Deletes the ref_audio_cache/ directory unless retain_cache is specified.
        """
        if not self.retain_cache:
            if os.path.exists(self.REF_AUDIO_CACHE_DIR):
                logging.info(f"Deleting cache directory: {self.REF_AUDIO_CACHE_DIR}")
                shutil.rmtree(self.REF_AUDIO_CACHE_DIR)
            os.makedirs(self.REF_AUDIO_CACHE_DIR, exist_ok=True)
            logging.info(f"Cache directory recreated: {self.REF_AUDIO_CACHE_DIR}")
        else:
            logging.info(f"Cache directory retained: {self.REF_AUDIO_CACHE_DIR}")

    def load_whisper_model(self):
        """Loads the Whisper model and processor."""
        logging.info("Loading Whisper model and processor...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to(self.device)
        return processor, model

    def convert_pt_to_safetensors(self, pt_path, safetensors_path):
        """Converts a .pt file to .safetensors format, handling `ema_model_state_dict`."""
        logging.info(f"Converting .pt file to .safetensors: {pt_path} -> {safetensors_path}")
        checkpoint = torch.load(pt_path, map_location="cpu")

        # Extract the `ema_model_state_dict` if it exists, otherwise use `state_dict`
        if "ema_model_state_dict" in checkpoint:
            state_dict = checkpoint["ema_model_state_dict"]
            logging.info("Found 'ema_model_state_dict' in checkpoint.")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logging.info("Found 'state_dict' in checkpoint.")
        else:
            raise KeyError(f"The checkpoint does not contain 'ema_model_state_dict' or 'state_dict'. Keys: {checkpoint.keys()}")

        # Ensure all values in state_dict are tensors
        validated_state_dict = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                logging.warning(f"Skipping key {key}: not a tensor (found {type(value)}).")
                continue
            validated_state_dict[key] = value

        # Save the validated tensors to .safetensors
        save_file(validated_state_dict, safetensors_path)
        logging.info(f".safetensors file successfully saved at {safetensors_path}")


    def discover_models(self):
        """
        Discover all available models in the ckpts/ directory.
        Populates the `available_models` dictionary.
        """
        logging.info("Discovering available models...")
        self.available_models = {}

        for folder in os.listdir(self.CKPTS_DIR):
            voice_dir = os.path.join(self.CKPTS_DIR, folder)
            if os.path.isdir(voice_dir):
                model_path = os.path.join(voice_dir, "model_1250000.safetensors")
                if not os.path.exists(model_path):
                    model_path = os.path.join(voice_dir, "model_1250000.pt")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(voice_dir, "model_1200000.safetensors")
                        if not os.path.exists(model_path):
                            model_path = os.path.join(voice_dir, "model_1200000.pt")
                
                logging.debug(f"Checking for model file for voice {folder}: {model_path}, exists: {os.path.exists(model_path)}")
                if os.path.exists(model_path):
                    self.available_models[folder] = model_path
                    logging.info(f"Discovered model for voice: {folder}")
                else:
                    available_files = os.listdir(voice_dir)
                    logging.warning(f"No valid checkpoint found for voice: {folder}. Files present: {available_files}")

    def load_voice_model(self, voice_name):
        """
        Load a voice model into memory, unloading another model if necessary.

        Args:
            voice_name (str): The name of the voice to load.

        Returns:
            model: The loaded model object.
        """
        if voice_name in self.loaded_models:
            # Move to end to mark as recently used
            self.loaded_models.move_to_end(voice_name)
            logging.info(f"Model for voice '{voice_name}' is already loaded.")
            return self.loaded_models[voice_name]

        if voice_name not in self.available_models:
            raise ValueError(f"Voice '{voice_name}' is not available.")

        # Enforce the model load limit
        if len(self.loaded_models) >= self.MODEL_LOAD_LIMIT:
            # Unload the least recently used (LRU) model
            lru_model_name, _ = self.loaded_models.popitem(last=False)
            logging.info(f"Unloading model: {lru_model_name}")
            self.unload_voice_model(lru_model_name)

        # Load the requested model
        model_path = self.available_models[voice_name]
        model = load_model(DiT, self.F5TTS_MODEL_CONFIG, model_path)
        self.loaded_models[voice_name] = model  # Add to loaded models (LRU order)
        logging.info(f"Loaded model for voice: {voice_name}")
        return model

    def unload_voice_model(self, voice_name):
        """
        Unload a specific voice model from memory.

        Args:
            voice_name (str): The name of the voice to unload.
        """
        if voice_name in self.loaded_models:
            del self.loaded_models[voice_name]
            logging.info(f"Unloaded model for voice: {voice_name}")
            gc.collect()
            logging.info(f"Garbage collection triggered after unloading '{voice_name}'.")
        else:
            logging.info(f"Model for voice '{voice_name}' is not currently loaded.")

    def list_available_models(self):
        """
        List all discovered voice models, indicating whether each is loaded.

        Returns:
            list: A list of dictionaries with model information.
        """
        return [
            {
                "name": voice_name,
                "status": "loaded" if voice_name in self.loaded_models else "unloaded"
            }
            for voice_name in self.available_models
        ]

    def load_default_model(self):
        """
        Load the default voice model if not loaded and within the load limit.
        """
        default_voice = self.default_voice
        if default_voice not in self.available_models:
            logging.error(f"Default voice '{default_voice}' not found in available models.")
            return

        if default_voice not in self.loaded_models and len(self.loaded_models) < self.MODEL_LOAD_LIMIT:
            try:
                self.load_voice_model(default_voice)
                logging.info(f"Loaded default '{default_voice}' model.")
            except Exception as e:
                logging.error(f"Failed to load default '{default_voice}' model: {e}")

    def get_ref_audio_path(self, voice_name):
        """
        Automatically find the reference audio file for a given voice.

        Args:
            voice_name (str): The name of the voice.

        Returns:
            str: Path to the reference audio file.

        Raises:
            FileNotFoundError: If no matching reference audio file is found.
        """
        supported_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.pcm']
        for ext in supported_extensions:
            ref_audio_path = os.path.join(self.REF_AUDIO_DIR, f"{voice_name}{ext}")
            if os.path.exists(ref_audio_path):
                logging.info(f"Found reference audio for '{voice_name}': {ref_audio_path}")
                return ref_audio_path
        raise FileNotFoundError(f"No reference audio file found for voice '{voice_name}' in {self.REF_AUDIO_DIR}")

    def process_reference_audio(self, voice_name):
        """
        Process the reference audio: trim, resample to 16kHz, convert to mono, and optionally perform PCM normalization.

        Args:
            voice_name (str): The name of the voice.

        Returns:
            str: Path to the processed audio file in the cache.
        """
        try:
            ref_audio_path = self.get_ref_audio_path(voice_name)
        except FileNotFoundError as e:
            logging.error(e)
            raise

        logging.info(f"Processing reference audio for voice '{voice_name}': {ref_audio_path}")

        base_filename = os.path.basename(ref_audio_path)
        cached_audio_path = os.path.join(self.REF_AUDIO_CACHE_DIR, f"{voice_name}_processed.wav")

        if not os.path.exists(cached_audio_path):
            logging.info(f"Processing and caching reference audio to: {cached_audio_path}")
            try:
                # Use FFmpeg to trim, resample to 16kHz, convert to mono
                subprocess.run([
                    "ffmpeg",
                    "-i", ref_audio_path,
                    "-t", str(self.REF_AUDIO_LIMIT_SECONDS),  # Trim to REF_AUDIO_LIMIT_SECONDS
                    "-ar", "16000",  # Resample to 16kHz
                    "-ac", "1",  # Convert to mono
                    "-y",  # Overwrite output files without asking
                    cached_audio_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info("Audio trimming and resampling completed and cached.")

                if not self.disable_pcm_normalization:
                    # Convert to raw PCM and back to WAV to normalize
                    temp_pcm_path = os.path.join(self.REF_AUDIO_CACHE_DIR, f"{voice_name}_processed.pcm")
                    subprocess.run([
                        "ffmpeg",
                        "-y",
                        "-i", cached_audio_path,
                        "-f", "s16le",
                        "-acodec", "pcm_s16le",
                        temp_pcm_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    logging.info("Audio converted to PCM.")

                    subprocess.run([
                        "ffmpeg",
                        "-y",
                        "-f", "s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        "-i", temp_pcm_path,
                        cached_audio_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    logging.info("Audio converted back from PCM.")

                    # Clean up PCM file
                    os.remove(temp_pcm_path)
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error: {e.stderr.decode()}")
                raise RuntimeError("Failed to process reference audio with FFmpeg.")

        # Verify the processed audio
        try:
            audio_input, sample_rate = sf.read(cached_audio_path)
            logging.debug(f"Processed audio loaded: {cached_audio_path} with sample rate {sample_rate}")

            # Ensure sample rate matches 16kHz
            if sample_rate != 16000:
                audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
                sf.write(cached_audio_path, audio_input, 16000)
                logging.info("Audio resampled to 16000 Hz.")
        except Exception as e:
            logging.error(f"Error verifying or resampling processed audio: {e}")
            raise RuntimeError("Failed to verify or resample processed audio.")

        return cached_audio_path

    def transcribe_audio(self, voice_name):
        """
        Transcribes the reference audio file for a given voice using Whisper.

        Args:
            voice_name (str): The name of the voice.

        Returns:
            str: Transcription text.
        """
        logging.info(f"Transcribing reference audio for voice '{voice_name}'...")
        processed_audio_path = self.process_reference_audio(voice_name)

        # Read the processed audio
        audio_input, sample_rate = sf.read(processed_audio_path)
        logging.debug(f"Processed audio loaded: {processed_audio_path} with sample rate {sample_rate}")

        # Process audio as input features for Whisper
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

        with torch.no_grad():
            generated_ids = self.whisper_model.generate(inputs)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.debug(f"Transcription result: {transcription}")
        return transcription

    def infer(self, gen_text, voice_name, model):
        """
        Generates speech using F5-TTS based on provided text and voice.

        Args:
            gen_text (str): Text to generate speech from.
            voice_name (str): Voice name.
            model: Loaded model object.

        Returns:
            tuple: (sample_rate, audio_data)
        """
        try:
            logging.debug(f"Generating speech for voice '{voice_name}': {gen_text}")
            ref_text = self.transcribe_audio(voice_name)

            ref_audio_path = self.process_reference_audio(voice_name)

            if ref_audio_path and os.path.exists(ref_audio_path):
                logging.info(f"Using reference audio for voice '{voice_name}': {ref_audio_path}")
                final_wave, final_sample_rate, _ = infer_process(
                    ref_audio_path, ref_text, gen_text, model, self.vocoder, cross_fade_duration=0.15, speed=1.0
                )
            else:
                logging.info(f"No reference audio found for voice '{voice_name}'. Proceeding without it.")
                final_wave, final_sample_rate, _ = infer_process(
                    None, ref_text, gen_text, model, self.vocoder, cross_fade_duration=0.15, speed=1.0
                )
            logging.debug(f"Generated waveform shape: {final_wave.shape}")
            return final_sample_rate, final_wave.squeeze().cpu().numpy() if isinstance(final_wave, torch.Tensor) else final_wave.squeeze()
        except Exception as e:
            logging.error(f"Error during inference process: {e}")
            raise RuntimeError("Failed to generate speech.")

    def generate_speech(self, text, voice='Emilia', response_format='mp3', speed=1.0):
        """
        Generates and saves speech audio from text for a specific voice.

        Args:
            text (str): Text to generate speech from.
            voice (str): Voice name.
            response_format (str): Audio format (e.g., 'mp3').
            speed (float): Speed adjustment factor.

        Returns:
            str: Path to the generated audio file.
        """
        logging.debug(f"generate_speech called with: text={text}, voice={voice}, response_format={response_format}, speed={speed}")

        if not text:
            logging.error("No text available for TTS generation.")
            raise ValueError("No text available for TTS generation.")

        if voice not in self.available_models:
            logging.error(f"Requested voice '{voice}' not found in available voices: {list(self.available_models.keys())}")
            raise ValueError(f"Voice '{voice}' is not available.")

        try:
            # Load the requested voice model (prioritize this model)
            model = self.load_voice_model(voice)
            logging.info(f"Using model for voice: {voice}")

            # Perform inference
            sample_rate, audio_data = self.infer(text, voice, model)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
            sf.write(temp_file.name, audio_data, sample_rate)
            logging.info(f"Generated speech saved to {temp_file.name}")
            return temp_file.name
        except Exception as e:
            logging.error(f"Error in generate_speech: {e}")
            raise RuntimeError("Failed to generate speech.")
