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
from gevent.lock import Semaphore
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
from generation_gate import GenerationGate
from reference_paths import (
    INPUT_EXTENSIONS, UntrustedReferencePath, validate_reference_voice,
    validate_reference_root, validate_reference_input, validate_reference_cache,
    require_missing_cache_intermediate,
)
from checkpoint_paths import (
    UntrustedCheckpointPath, validate_checkpoint_root, validate_checkpoint_path,
    CHECKPOINT_NAMES,
)
from sentences import split_sentences
from model_config import model_config_for

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
        if self.MODEL_LOAD_LIMIT < 1:
            raise ValueError('MODEL_LOAD_LIMIT must be at least 1')
        self.REF_AUDIO_LIMIT_SECONDS = int(os.getenv("REF_AUDIO_LIMIT_SECONDS", "20"))
        self.FFMPEG_TIMEOUT_SECONDS = int(os.getenv("FFMPEG_TIMEOUT_SECONDS", "120"))
        if self.FFMPEG_TIMEOUT_SECONDS < 1:
            raise ValueError('FFMPEG_TIMEOUT_SECONDS must be at least 1')
        self.REF_AUDIO_CACHE_DIR = "ref_audio_cache"
        os.makedirs(self.REF_AUDIO_CACHE_DIR, exist_ok=True)

        self.retain_cache = retain_cache
        self.disable_pcm_normalization = disable_pcm_normalization
        self.default_voice = default_voice
        # Gevent serves cooperative request handlers sharing this mutable model cache.
        # Serialize model swaps and inference; overlapping GPU jobs are unsafe here.
        self._inference_lock = Semaphore(1)
        self._generation_gate = GenerationGate()
        self._ref_text_cache: dict = {}

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

    def discover_models(self):
        """
        Discover all available models in the ckpts/ directory.
        Populates the `available_models` dictionary.
        """
        logging.info("Discovering available models...")
        self.available_models = {}

        # Fail closed before walking a symlinked checkpoint root.
        validate_checkpoint_root(self.CKPTS_DIR)
        for folder in os.listdir(self.CKPTS_DIR):
            if os.path.isfile(os.path.join(self.CKPTS_DIR, folder)):
                continue  # README.md, .gitkeep etc. are not voices
            model_path = None
            for name in CHECKPOINT_NAMES:
                path = os.path.join(self.CKPTS_DIR, folder, name)
                try:
                    validate_checkpoint_path(self.CKPTS_DIR, folder, path)
                except UntrustedCheckpointPath:
                    continue
                model_path = path
                break
            if model_path is not None:
                self.available_models[folder] = model_path
                logging.info("Discovered model for voice: %s", folder)
            else:
                logging.warning("No eligible native safetensors checkpoint for voice %r", folder)

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

        # Revalidate the remembered path before evicting any healthy model.
        # Pathname checks cannot pin the inode during upstream load_model().
        model_path = validate_checkpoint_path(
            self.CKPTS_DIR, voice_name, self.available_models[voice_name]
        )

        # Enforce the model load limit
        if len(self.loaded_models) >= self.MODEL_LOAD_LIMIT:
            # Unload the least recently used (LRU) model
            lru_model_name, lru_model = self.loaded_models.popitem(last=False)
            logging.info("Evicting model: %s", lru_model_name)
            # The model was already removed from the LRU dictionary. Drop the
            # last local reference *before* allocating the replacement model.
            del lru_model
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Load the already checked pathname. The checkpoint tree must remain
        # immutable under an untrusted actor while the upstream loader runs.
        # Never register an incomplete model after a failed load. Because the
        # old model was evicted before allocation to cap peak memory, it stays
        # unloaded; later requests may reload that voice from disk.
        try:
            model = load_model(DiT, model_config_for(model_path), model_path)
            if model is None:
                raise RuntimeError('Voice checkpoint loader returned no model')
        except Exception:
            # A failed allocation may leave collectable Python/GPU objects.
            # Do not mask the original checkpoint failure if cleanup also fails.
            try:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                logging.warning("Post-failure model allocator cleanup failed", exc_info=True)
            raise
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
        validate_reference_voice(voice_name)
        validate_reference_root(self.REF_AUDIO_DIR)
        for ext in INPUT_EXTENSIONS:
            ref_audio_path = os.path.join(self.REF_AUDIO_DIR, f"{voice_name}{ext}")
            if os.path.lexists(ref_audio_path):
                validate_reference_input(self.REF_AUDIO_DIR, voice_name, ref_audio_path)
                logging.info("Found reference audio for voice %r", voice_name)
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
        validate_reference_voice(voice_name)
        try:
            ref_audio_path = self.get_ref_audio_path(voice_name)
        except FileNotFoundError as e:
            logging.error(e)
            raise
        # Recheck even if the first lookup happened before an external rename.
        validate_reference_input(self.REF_AUDIO_DIR, voice_name, ref_audio_path)
        validate_reference_root(self.REF_AUDIO_CACHE_DIR)
        logging.info("Processing reference audio for voice %r", voice_name)

        cached_audio_path = os.path.join(self.REF_AUDIO_CACHE_DIR, f"{voice_name}_processed.wav")
        temp_pcm_path = os.path.join(self.REF_AUDIO_CACHE_DIR, f"{voice_name}_processed.pcm")
        validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name, cached_audio_path,
                                 "wav", allow_missing=True)

        if not os.path.lexists(cached_audio_path):
            # Never clobber or remove a leftover/foreign intermediate PCM file.
            require_missing_cache_intermediate(
                self.REF_AUDIO_CACHE_DIR, voice_name, temp_pcm_path
            )
            logging.info("Processing reference audio for voice %r into cache", voice_name)
            try:
                validate_reference_input(self.REF_AUDIO_DIR, voice_name, ref_audio_path)
                validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                         cached_audio_path, "wav", allow_missing=True)
                # Use FFmpeg to trim, resample to 16kHz, convert to mono
                subprocess.run([
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats",
                    "-i", ref_audio_path,
                    "-t", str(self.REF_AUDIO_LIMIT_SECONDS),  # Trim to REF_AUDIO_LIMIT_SECONDS
                    "-ar", "16000",  # Resample to 16kHz
                    "-ac", "1",  # Convert to mono
                    "-y",  # Overwrite output files without asking
                    cached_audio_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.FFMPEG_TIMEOUT_SECONDS)
                validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                         cached_audio_path, "wav")
                logging.info("Audio trimming and resampling completed and cached.")

                if not self.disable_pcm_normalization:
                    # Reject unexpected intermediate substitution before writing.
                    require_missing_cache_intermediate(
                        self.REF_AUDIO_CACHE_DIR, voice_name, temp_pcm_path
                    )
                    validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                             cached_audio_path, "wav")
                    # Convert to raw PCM and back to WAV to normalize
                    subprocess.run([
                        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats",
                        "-y",
                        "-i", cached_audio_path,
                        "-f", "s16le",
                        "-acodec", "pcm_s16le",
                        temp_pcm_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.FFMPEG_TIMEOUT_SECONDS)
                    validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                             temp_pcm_path, "pcm")
                    validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                             cached_audio_path, "wav")
                    logging.info("Audio converted to PCM.")

                    subprocess.run([
                        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats",
                        "-y",
                        "-f", "s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        "-i", temp_pcm_path,
                        cached_audio_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.FFMPEG_TIMEOUT_SECONDS)
                    validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                             cached_audio_path, "wav")
                    logging.info("Audio converted back from PCM.")

            except Exception as exc:
                # Never reuse a partial or timed-out reference-audio conversion.
                logging.warning("Reference audio conversion failed; discarding partial cache")
                try:
                    os.remove(cached_audio_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logging.warning("Could not remove incomplete reference cache")
                raise RuntimeError("Failed to process reference audio with FFmpeg") from exc
            finally:
                try:
                    os.remove(temp_pcm_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logging.warning("Could not remove intermediate reference PCM")

        # Reject a substituted cache file before decoding or resampling.
        validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                 cached_audio_path, "wav")
        try:
            audio_input, sample_rate = sf.read(cached_audio_path)
            logging.debug(f"Processed audio loaded: {cached_audio_path} with sample rate {sample_rate}")

            # Ensure sample rate matches 16kHz
            if sample_rate != 16000:
                audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
                validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                         cached_audio_path, "wav")
                sf.write(cached_audio_path, audio_input, 16000)
                logging.info("Audio resampled to 16000 Hz.")
        except Exception as e:
            # A failed cache validation must trigger a fresh conversion next time.
            try:
                os.remove(cached_audio_path)
            except OSError:
                pass
            logging.error("Failed to verify processed reference audio")
            raise RuntimeError("Failed to verify or resample processed audio") from e

        return cached_audio_path

    def transcribe_audio(self, voice_name):
        """
        Transcribes the reference audio file for a given voice using Whisper.
        Result is cached in memory so Whisper only runs once per voice per server lifetime.

        Args:
            voice_name (str): The name of the voice.

        Returns:
            str: Transcription text.
        """
        cached = getattr(self, "_ref_text_cache", None)
        if cached is not None and voice_name in cached:
            logging.debug(f"Using cached transcription for voice '{voice_name}'.")
            return cached[voice_name]

        logging.info(f"Transcribing reference audio for voice '{voice_name}'...")
        processed_audio_path = self.process_reference_audio(voice_name)

        # Revalidate after processing, before Whisper reads the pathname.
        validate_reference_cache(self.REF_AUDIO_CACHE_DIR, voice_name,
                                 processed_audio_path, "wav")
        audio_input, sample_rate = sf.read(processed_audio_path)
        logging.debug(f"Processed audio loaded: {processed_audio_path} with sample rate {sample_rate}")

        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

        with torch.no_grad():
            generated_ids = self.whisper_model.generate(inputs)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.debug("Reference transcription produced %d characters", len(transcription))
        if getattr(self, "_ref_text_cache", None) is None:
            self._ref_text_cache = {}
        self._ref_text_cache[voice_name] = transcription
        return transcription

    def infer(self, gen_text, voice_name, model, speed=1.0):
        """
        Generates speech using F5-TTS based on provided text and voice.

        Args:
            gen_text (str): Text to generate speech from.
            voice_name (str): Voice name.
            model: Loaded model object.
            speed (float): Speed adjustment factor.

        Returns:
            tuple: (sample_rate, audio_data)
        """
        try:
            logging.debug("Generating speech for voice %r (%d text characters)", voice_name, len(gen_text))
            ref_text = self.transcribe_audio(voice_name)

            ref_audio_path = self.process_reference_audio(voice_name)

            if ref_audio_path and os.path.exists(ref_audio_path):
                logging.info(f"Using reference audio for voice '{voice_name}': {ref_audio_path}")
                final_wave, final_sample_rate, _ = infer_process(
                    ref_audio_path, ref_text, gen_text, model, self.vocoder, cross_fade_duration=0.15, speed=speed
                )
            else:
                logging.info(f"No reference audio found for voice '{voice_name}'. Proceeding without it.")
                final_wave, final_sample_rate, _ = infer_process(
                    None, ref_text, gen_text, model, self.vocoder, cross_fade_duration=0.15, speed=speed
                )
            logging.debug(f"Generated waveform shape: {final_wave.shape}")
            return final_sample_rate, final_wave.squeeze().cpu().numpy() if isinstance(final_wave, torch.Tensor) else final_wave.squeeze()
        except Exception as e:
            logging.error(f"Error during inference process: {e}")
            raise RuntimeError("Failed to generate speech.")

    def generate_speech(self, text, voice='Emilia', response_format='mp3', speed=1.0):
        """Fail fast if a process-local synthesis request is already active."""
        with self._generation_gate.claim():
            return self._generate_speech_claimed(
                text, voice=voice, response_format=response_format, speed=speed
            )

    def _generate_speech_claimed(self, text, voice='Emilia', response_format='mp3', speed=1.0):
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
        logging.debug("generate_speech request: voice=%r format=%r speed=%s characters=%d", voice, response_format, speed, len(text) if isinstance(text, str) else 0)

        if not text:
            logging.error("No text available for TTS generation.")
            raise ValueError("No text available for TTS generation.")

        if voice not in self.available_models:
            logging.error(f"Requested voice '{voice}' not found in available voices: {list(self.available_models.keys())}")
            raise ValueError(f"Voice '{voice}' is not available.")

        try:
            # Load the requested voice model (prioritize this model)
            with self._inference_lock:
                model = self.load_voice_model(voice)
                logging.info("Using model for voice: %s", voice)
                # The shared cache is not safe across concurrent inference/model swaps.
                sample_rate, audio_data = self.infer(text, voice, model, speed=speed)
            
            output_path = self._write_audio_file(sample_rate, audio_data, response_format)
            logging.info("Generated speech audio for voice %s", voice)
            return output_path
        except Exception as e:
            logging.error(f"Error in generate_speech: {e}")
            raise RuntimeError("Failed to generate speech.")
    
    def _write_audio_file(self, sample_rate, audio_data, response_format):
        """Encode one request, removing partial WAV and converted files on any error.

        Each temporary file is closed before another library opens it, including
        on Windows. Only the successfully returned output remains for the HTTP
        request handler to consume and delete.
        """
        fmt = response_format.lower()
        if fmt not in AUDIO_FORMAT_MIME_TYPES:
            raise ValueError("Unsupported response format")
        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = wav_tmp.name
        wav_tmp.close()
        converted_path = None
        retained_path = None
        try:
            sf.write(wav_path, audio_data, sample_rate)
            if fmt == "wav":
                retained_path = wav_path
            else:
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + fmt)
                converted_path = out_tmp.name
                out_tmp.close()
                command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats", "-y", "-i", wav_path, "-c:a", self._get_ffmpeg_codec(fmt)]
                # Raw PCM has no container header, so ffmpeg needs an explicit muxer.
                if fmt == "pcm":
                    command.extend(["-f", "s16le"])
                command.append(converted_path)
                subprocess.run(
                    command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.FFMPEG_TIMEOUT_SECONDS
                )
                retained_path = converted_path
            return retained_path
        finally:
            for path in (wav_path, converted_path):
                if path and path != retained_path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logging.warning("Could not delete incomplete generated audio")

    def _get_ffmpeg_codec(self, format_name):
        """Map response format to ffmpeg codec."""
        codecs = {
            'mp3': 'libmp3lame',
            'opus': 'libopus',
            'aac': 'aac',
            'flac': 'flac',
            'wav': 'pcm_s16le',
            'pcm': 'pcm_s16le',
        }
        return codecs.get(format_name.lower(), 'libmp3lame')



    def generate_speech_stream(self, text, voice='Emilia', speed=1.0):
        """
        Generator that splits text into sentences and yields int16 PCM bytes at 16 kHz
        as each sentence is synthesised. Loads the model and reference data once before
        the loop so neither is reloaded per sentence.

        Args:
            text (str): Full text to synthesise.
            voice (str): Voice name.
            speed (float): Speed adjustment factor.

        Yields:
            bytes: Raw int16 little-endian PCM audio at 16 kHz, one chunk per sentence.
        """
        if not text:
            raise ValueError("No text provided for TTS generation.")
        if voice not in self.available_models:
            raise ValueError(f"Voice '{voice}' is not available.")

        model = self.load_voice_model(voice)
        ref_text = self.transcribe_audio(voice)
        ref_audio_path = self.process_reference_audio(voice)

        sentences = split_sentences(text)
        logging.info(f"Streaming {len(sentences)} sentence(s) for voice '{voice}'.")

        try:
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                logging.debug(f"Synthesising sentence: {sentence}")
                final_wave, final_sample_rate, _ = infer_process(
                    ref_audio_path, ref_text, sentence, model, self.vocoder,
                    cross_fade_duration=0.15, speed=speed
                )
                audio = final_wave.squeeze().cpu().numpy() if isinstance(final_wave, torch.Tensor) else final_wave.squeeze()

                if final_sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=final_sample_rate, target_sr=16000)

                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                yield pcm.tobytes()
        except GeneratorExit:
            logging.info(f"Client disconnected — stopping stream for voice '{voice}'.")
            raise
