import os
import io
import base64
import logging
from flask import Flask, request, send_file, jsonify
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
from argparse import ArgumentParser

from tts_handler import TTSHandler
from utils import require_api_key, AUDIO_FORMAT_MIME_TYPES, REQUIRE_API_KEY
from auth_bind_policy import validate_auth_bind
from alignment import build_alignment
from request_validation import parse_speech_request, parse_timestamp_request, RequestValidationError
from generation_gate import SynthesisBusyError
from output_limits import read_audio_limited, AudioOutputTooLarge

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()
# Inbound payloads are bounded before Flask parses JSON. One request must not
# enqueue arbitrarily large text into the expensive synthesis pipeline.
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_REQUEST_BYTES', '65536'))
MAX_TEXT_CHARS = int(os.getenv('MAX_TEXT_CHARS', '2000'))
MAX_AUDIO_OUTPUT_BYTES = int(os.getenv('MAX_AUDIO_OUTPUT_BYTES', '16777216'))
if min(app.config['MAX_CONTENT_LENGTH'], MAX_TEXT_CHARS, MAX_AUDIO_OUTPUT_BYTES) <= 0:
    raise ValueError('MAX_REQUEST_BYTES, MAX_TEXT_CHARS and MAX_AUDIO_OUTPUT_BYTES must be positive')

# Parse command-line arguments
parser = ArgumentParser(description="F5-TTS Server")
parser.add_argument(
    "--retain-cache",
    action="store_true",
    help="Retain the ref_audio_cache/ directory across instance restarts."
)
parser.add_argument(
    "--disable-pcm-normalization",
    action="store_true",
    help="Disable PCM normalization during audio processing."
)
parser.add_argument(
    "--port",
    type=int,
    default=int(os.getenv('PORT', 9090)),
    help="Port to run the server on."
)
parser.add_argument(
    "--host",
    default=os.getenv("HOST", "0.0.0.0"),
    help="Host interface to bind (default: 0.0.0.0 for Docker)."
)
args = parser.parse_args()
# Reject unsafe unauthenticated binds *before* expensive model initialization.
# The container's loopback-published host port is not sufficient when its own
# process listens on 0.0.0.0 and REQUIRE_API_KEY is explicitly disabled.
validate_auth_bind(args.host, REQUIRE_API_KEY)

# Load configuration from environment variables with defaults
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'Emilia')
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', 'mp3')
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', 1.0))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() in ('true', '1', 'yes')
PORT = args.port

# Set up logging configuration
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Debug mode is enabled.")
else:
    logging.basicConfig(level=logging.INFO)
logging.info(f"Debug mode is {'enabled' if DEBUG_MODE else 'disabled'}.")

# Middleware for logging HTTP requests in debug mode
@app.before_request
def log_request_info():
    if DEBUG_MODE:
        # Never log headers, URL query parameters or synthesis text: credentials
        # can arrive in either Authorization or xi-api-key, and text is private.
        logging.debug('HTTP Request: %s %s', request.method, request.path)

# Initialize TTSHandler with CLI arguments and DEFAULT_VOICE
tts_handler = TTSHandler(
    retain_cache=args.retain_cache,
    disable_pcm_normalization=args.disable_pcm_normalization,
    default_voice=DEFAULT_VOICE
)

def _delete_request_file(path):
    """Delete only a file owned by the calling request; never touch peer jobs."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logging.warning("Could not delete generated audio file")


def _send_and_delete(path, mime_type, download_name):
    """Buffer audio and delete this request's file before creating the response."""
    try:
        payload = read_audio_limited(path, MAX_AUDIO_OUTPUT_BYTES)
    finally:
        _delete_request_file(path)
    return send_file(
        io.BytesIO(payload),
        mimetype=mime_type,
        as_attachment=True,
        download_name=download_name,
    )


@app.route('/v1/audio/speech', methods=['POST'])
@require_api_key
def text_to_speech():
    """
    Handle POST requests to generate speech from text input.

    Expects a JSON body with the following fields:
      - input: The text to convert to speech.
      - voice: (Optional) The speaker's name. Defaults to the DEFAULT_VOICE.
      - response_format: (Optional) Desired audio format. Defaults to 'mp3'.
      - speed: (Optional) Speed adjustment factor. Defaults to 1.0.

    Returns:
        Audio file in the requested format.
    """
    try:
        params = parse_speech_request(
            request.get_json(silent=True),
            default_voice=DEFAULT_VOICE,
            default_format=DEFAULT_RESPONSE_FORMAT,
            default_speed=DEFAULT_SPEED,
            max_text_chars=MAX_TEXT_CHARS,
        )
    except RequestValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    text, voice = params["text"], params["voice"]
    response_format, speed = params["response_format"], params["speed"]

    # Determine MIME type based on response format
    mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format.lower(), "audio/mpeg")

    try:
        # Generate speech using TTSHandler and return the audio file
        output_file_path = tts_handler.generate_speech(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed
        )
        return _send_and_delete(
            output_file_path, mime_type, f"speech.{response_format}"
        )
    except ValueError as e:
        logging.error(f"ValueError during TTS generation: {e}")
        return jsonify({"error": str(e)}), 400
    except AudioOutputTooLarge:
        return jsonify({"error": "Generated audio exceeds configured output limit"}), 503
    except SynthesisBusyError:
        return jsonify({"error": "Speech generator busy; try again later"}), 503
    except RuntimeError as e:
        logging.error(f"RuntimeError during TTS generation: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Unhandled exception during TTS generation: {e}")
        return jsonify({"error": "Failed to generate speech"}), 500




@app.route("/v1/text-to-speech/<voice_id>/with-timestamps", methods=["POST"])
@require_api_key
def text_to_speech_with_timestamps(voice_id):
    """ElevenLabs-shaped timestamps. Missing aligner software does not fail TTS."""
    try:
        params = parse_timestamp_request(
            request.get_json(silent=True),
            voice=voice_id or DEFAULT_VOICE,
            default_format=DEFAULT_RESPONSE_FORMAT,
            default_speed=DEFAULT_SPEED,
            output_format=request.args.get("output_format"),
            max_text_chars=MAX_TEXT_CHARS,
        )
    except RequestValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    text, voice = params["text"], params["voice"]
    response_format, speed = params["response_format"], params["speed"]
    mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format.lower(), "audio/mpeg")

    try:
        output_file_path = tts_handler.generate_speech(
            text=text, voice=voice, response_format=response_format, speed=speed
        )
        try:
            alignment, words, source = build_alignment(text, output_file_path)
            audio_b64 = base64.b64encode(
                read_audio_limited(output_file_path, MAX_AUDIO_OUTPUT_BYTES)
            ).decode("ascii")
        finally:
            _delete_request_file(output_file_path)
        return jsonify({
            "audio_base64": audio_b64,
            "alignment": alignment,
            "normalized_alignment": alignment,
            "words": words,
            "alignment_source": source,
        })
    except AudioOutputTooLarge:
        return jsonify({"error": "Generated audio exceeds configured output limit"}), 503
    except SynthesisBusyError:
        return jsonify({"error": "Speech generator busy; try again later"}), 503
    except ValueError as e:
        logging.error(f"ValueError during timestamped TTS: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Unhandled exception during timestamped TTS: {e}")
        return jsonify({"error": "Failed to generate speech"}), 500


@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """
    List available TTS models.

    Returns:
        JSON response with available models.
    """
    models = tts_handler.list_available_models()
    return jsonify({"models": models})

@app.route('/v1/voices', methods=['GET'])
@require_api_key
def list_voices():
    """
    List available voices, with optional language filtering.

    Returns:
        JSON response with available voices.
    """
    specific_language = None
    data = request.args if request.method == 'GET' else request.json

    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    models = tts_handler.list_available_models()
    if specific_language:
        # Assuming you have language metadata for each model, which isn't currently implemented.
        # This is a placeholder for actual language filtering logic.
        filtered_models = [model for model in models if model.get('language') == specific_language]
    else:
        filtered_models = models

    voices = [model['name'] for model in filtered_models]
    return jsonify({"voices": voices})

@app.route('/v1/voices/all', methods=['GET'])
@require_api_key
def list_all_voices():
    """
    List all supported voices.

    Returns:
        JSON response with all supported voices.
    """
    models = tts_handler.list_available_models()
    voices = [model['name'] for model in models]
    return jsonify({"voices": voices})

@app.route('/v1/loaded_models', methods=['GET'])
@require_api_key
def get_loaded_models():
    """
    List currently loaded TTS models.

    Returns:
        JSON response with loaded models.
    """
    loaded = list(tts_handler.loaded_models.keys())
    return jsonify({"loaded_models": loaded})

if __name__ == '__main__':
    logging.info(f"F5-TTS API running on http://localhost:{PORT}")
    # Start the server using Gevent WSGI server for better concurrency support
    http_server = WSGIServer((args.host, PORT), app)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server.")
