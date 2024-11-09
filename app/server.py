import os
import logging
from flask import Flask, request, send_file, jsonify
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
from tts_handler import generate_speech, get_models, get_voices
from utils import require_api_key, AUDIO_FORMAT_MIME_TYPES

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

# Load configuration from environment variables with defaults
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
PORT = int(os.getenv('PORT', 5060))
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'Emilia')
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', 'mp3')
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', 1.0))

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

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
      - ref_audio: (Optional) Reference audio for transcription.
    
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
    ref_audio = data.get('ref_audio', None)

    # Determine MIME type based on response format
    mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format.lower(), "audio/mpeg")

    try:
        # Generate speech using F5-TTS and return the audio file
        output_file_path = generate_speech(text, voice, response_format, speed, ref_audio=ref_audio)
        return send_file(output_file_path, mimetype=mime_type,
                         as_attachment=True,
                         download_name=f"speech.{response_format}")

    except Exception as e:
        logging.error(f"Error during TTS generation: {e}")
        return jsonify({"error": "Failed to generate speech"}), 500

@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """
    List available TTS models.

    Returns:
        JSON response with available models.
    """
    return jsonify({"data": get_models()})

@app.route('/v1/voices', methods=['GET'])
@require_api_key
def list_voices():
    """
    List available voices.

    Returns:
        JSON response with available voices.
    """
    specific_language = None
    data = request.args if request.method == 'GET' else request.json

    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    return jsonify({"voices": get_voices(specific_language)})

@app.route('/v1/voices/all', methods=['GET'])
@require_api_key
def list_all_voices():
    """
    List all supported voices.

    Returns:
        JSON response with all supported voices.
    """
    return jsonify({"voices": get_voices('all')})

if __name__ == '__main__':
    print(f"F5-TTS API running on http://localhost:{PORT}")
    
    # Start the server using Gevent WSGI server for better concurrency support
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")
