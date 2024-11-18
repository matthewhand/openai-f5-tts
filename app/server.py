import os
import logging
from flask import Flask, request, send_file, jsonify
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
from argparse import ArgumentParser

from tts_handler import TTSHandler
from utils import require_api_key, AUDIO_FORMAT_MIME_TYPES

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

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
    default=int(os.getenv('PORT', 5060)),
    help="Port to run the server on."
)
args = parser.parse_args()

# Load configuration from environment variables with defaults
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'Emilia')
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', 'mp3')
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', 1.0))
PORT = args.port

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Initialize TTSHandler with CLI arguments
tts_handler = TTSHandler(
    retain_cache=args.retain_cache,
    disable_pcm_normalization=args.disable_pcm_normalization
)

@app.route('/v1/audio/speech', methods=['POST'])
@require_api_key
def text_to_speech():
    """
    Handle POST requests to generate speech from text input.

    Expects a JSON body with the following fields:
      - input: The text to convert to speech.
      - voice: (Optional) The speaker's name. Defaults to 'Emilia'.
      - response_format: (Optional) Desired audio format. Defaults to 'mp3'.
      - speed: (Optional) Speed adjustment factor. Defaults to 1.0.

    Returns:
        Audio file in the requested format.
    """
    data = request.json

    # Validate request body
    if not data or 'input' not in data:
        return jsonify({"error": "Missing 'input' in request body"}), 400

    # Extract parameters from request body with defaults
    text = data.get('input')
    voice = data.get('voice', DEFAULT_VOICE)
    response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
    speed = float(data.get('speed', DEFAULT_SPEED))

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
        return send_file(output_file_path, mimetype=mime_type,
                         as_attachment=True,
                         download_name=f"speech.{response_format}")
    except ValueError as e:
        logging.error(f"ValueError during TTS generation: {e}")
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        logging.error(f"RuntimeError during TTS generation: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Unhandled exception during TTS generation: {e}")
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

if __name__ == '__main__':
    logging.info(f"F5-TTS API running on http://localhost:{PORT}")
    # Start the server using Gevent WSGI server for better concurrency support
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server.")
