import os
import logging
from flask import Flask, request, send_file, jsonify, g, has_request_context
from dotenv import load_dotenv
from .tts_handler import TTSHandler
from .utils import require_api_key, AUDIO_FORMAT_MIME_TYPES
from .engine import load_engines, mix_audio
import uuid
import tempfile

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

# Configuration from environment variables
RETAIN_CACHE = os.getenv("RETAIN_CACHE", "false").lower() in ("true","1","yes")
DISABLE_PCM_NORMALIZATION = os.getenv("DISABLE_PCM_NORMALIZATION", "false").lower() in ("true","1","yes")
PORT = int(os.getenv("PORT", 9090))

# Load configuration from environment variables with defaults
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'Emilia')
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', 'mp3')
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', 1.0))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() in ('true', '1', 'yes')

# Add filter to include request_id in logs
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if has_request_context():
            record.request_id = getattr(g, 'request_id', '-')
        else:
            record.request_id = '-'
        return True
logging.getLogger().addFilter(RequestIDFilter())
# Set up structured logging configuration
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(request_id)s] %(message)s')
    logging.debug("Debug mode is enabled.")
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(request_id)s] %(message)s')
logging.info(f"Debug mode is {'enabled' if DEBUG_MODE else 'disabled' }.")

# Middleware for logging HTTP requests in debug mode
@app.before_request
def log_request_info():
    # Assign unique request ID for traceability
    g.request_id = str(uuid.uuid4())
    if DEBUG_MODE:
        # Redact Authorization header
        redacted_headers = {k: ('***' if 'authorization' in k.lower() else v)
                            for k, v in request.headers.items()}
        logging.debug(f"HTTP Request: {request.method} {request.url}")
        logging.debug(f"Headers: {redacted_headers}")
        if request.is_json:
            logging.debug(f"Payload: {request.json}")
        else:
            logging.debug("Payload: Non-JSON or empty.")

# Initialize TTSHandler with environment-configured cache and normalization settings
tts_handler = TTSHandler(
    retain_cache=RETAIN_CACHE,
    disable_pcm_normalization=DISABLE_PCM_NORMALIZATION,
    default_voice=DEFAULT_VOICE
)

# Load and register TTS engine plugins
ENGINES = load_engines()

@app.route('/v1/audio/speech', methods=['POST'])
@require_api_key
def text_to_speech():
    """
    Generate speech from text input.
    ---
    tags:
      - TTS
    consumes:
      - application/json
    produces:
      - audio/mpeg
      - audio/wav
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            input:
              type: string
            voice:
              type: string
            response_format:
              type: string
            speed:
              type: number
            ref_audio:
              type: string
            engine:
              type: string
    responses:
      200:
        description: Audio file
      400:
        description: Bad request
    """
    data = request.json

    # Validate request body
    if not data or 'input' not in data:
        return jsonify({"error": "Missing 'input' in request body"}), 400

    # Extract parameters from request body with defaults
    text = data.get('input')
    voice = data.get('voice') or DEFAULT_VOICE  # Use DEFAULT_VOICE if not provided or empty
    response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
    speed = float(data.get('speed', DEFAULT_SPEED))
    # Optional reference audio path override
    ref_audio = data.get('ref_audio')
    # Plugin engine override
    engine_name = data.get('engine')
    if engine_name:
        engine = ENGINES.get(engine_name)
        if not engine:
            return jsonify({"error": f"Engine '{engine_name}' not found"}), 404
        # Generate via plugin
        sr, audio_bytes = engine.generate(text, voice, response_format=response_format, speed=speed, ref_audio=ref_audio)
        # Write bytes to temp file
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
        tf.write(audio_bytes)
        tf.close()
        return send_file(tf.name, mimetype=AUDIO_FORMAT_MIME_TYPES.get(response_format.lower(), "audio/mpeg"), as_attachment=True, download_name=f"speech.{response_format}")
    else:
        # Determine MIME type based on response format
        mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format.lower(), "audio/mpeg")

        try:
            # Generate speech using TTSHandler and return the audio file
            output_file_path = tts_handler.generate_speech(
                text=text,
                voice=voice,
                response_format=response_format,
                speed=speed,
                ref_audio=ref_audio
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

@app.route('/v1/audio/mix', methods=['POST'])
@require_api_key
def mix_audio_api():
    """
    Mix multiple audio files via API.
    Consumes multipart/form-data with file inputs.
    """
    import soundfile as sf
    if not request.files:
        return jsonify({"error": "No input files provided"}), 400
    files = request.files.getlist('inputs')
    if not files:
        return jsonify({"error": "Missing 'inputs' files"}), 400
    mode = request.form.get('mode', 'overlay')
    sr_val = request.form.get('sr')
    sr = int(sr_val) if sr_val else None
    arrays = []
    for f in files:
        try:
            data, s = sf.read(f)
        except Exception as e:
            return jsonify({"error": f"Failed to read file: {e}"}), 400
        if sr is None:
            sr = s
        elif sr != s:
            return jsonify({"error": f"Sample rate mismatch: {s}Hz, expected {sr}Hz"}), 400
        arrays.append(data)
    try:
        mixed = mix_audio(arrays, mode=mode)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tf.name, mixed, sr, subtype='FLOAT')
    tf.close()
    return send_file(tf.name, mimetype=AUDIO_FORMAT_MIME_TYPES.get('wav', 'audio/wav'), as_attachment=True, download_name='mixed.wav')

@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """
    Lists available TTS models.
    ---
    tags:
      - Models
    responses:
      200:
        description: A list of available models
    """
    models = tts_handler.list_available_models()
    return jsonify({"models": models})

@app.route('/v1/voices', methods=['GET'])
@require_api_key
def list_voices():
    """
    Lists available voices.
    ---
    tags:
      - Voices
    parameters:
      - name: language
        in: query
        type: string
    responses:
      200:
        description: List of voices
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
    Lists all supported voices.
    ---
    tags:
      - Voices
    responses:
      200:
        description: List of all voices
    """
    models = tts_handler.list_available_models()
    voices = [model['name'] for model in models]
    return jsonify({"voices": voices})

@app.route('/v1/loaded_models', methods=['GET'])
@require_api_key
def get_loaded_models():
    """
    Lists currently loaded TTS models in memory.
    ---
    tags:
      - Models
    responses:
      200:
        description: List of loaded models
    """
    loaded = list(tts_handler.loaded_models.keys())
    return jsonify({"loaded_models": loaded})

@app.route('/v1/engines', methods=['GET'])
@require_api_key
def list_engines():
    """List available TTS engine plugins."""
    return jsonify({"engines": list(ENGINES.keys())})

@app.route('/v1/engines/<string:engine_name>/voices', methods=['GET'])
@require_api_key
def engine_voices(engine_name):
    """List voices for a specific TTS engine."""
    engine = ENGINES.get(engine_name)
    if not engine:
        return jsonify({"error": f"Engine '{engine_name}' not found"}), 404
    return jsonify({"voices": engine.list_voices()})

# Health check endpoint
@app.route('/healthz', methods=['GET'])
def healthz():
    """
    Health check endpoint returning service status.
    ---
    tags:
      - Health
    responses:
      200:
        description: Service status OK
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
    """
    return jsonify({'status': 'ok'}), 200

# Documentation endpoints (stub)
@app.route('/apidocs/swagger.json')
def swagger_spec():
    """Serve Swagger/OpenAPI JSON spec stub."""
    spec = {'openapi': '3.0.0', 'info': {'title': 'openai-f5-tts', 'version': '0.1.0'}, 'paths': {}}
    return jsonify(spec)

@app.route('/apidocs/')
def swagger_ui():
    """Serve Swagger UI HTML stub."""
    html = "<!doctype html><html><head><title>Swagger UI</title></head><body><h1>Swagger UI</h1></body></html>"
    return html, 200, {'Content-Type': 'text/html'}

# Global JSON error handlers
@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def handle_500(e):
    logging.exception(f"Unhandled exception: {e}")
    return jsonify({'error': 'Internal server error'}), 500

try:
    from gevent.pywsgi import WSGIServer
except ImportError:
    WSGIServer = None  # Gevent not installed

if __name__ == '__main__':
    logging.info(f"F5-TTS API running on http://localhost:{PORT}")
    # Start the server using Gevent WSGI server for better concurrency support
    if WSGIServer is not None:
        http_server = WSGIServer(('0.0.0.0', PORT), app)
        try:
            http_server.serve_forever()
        except KeyboardInterrupt:
            logging.info("Shutting down server.")
