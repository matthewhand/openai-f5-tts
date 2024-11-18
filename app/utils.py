from flask import request, jsonify
from functools import wraps
import os
import logging
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

def getenv_bool(name: str, default: bool = False) -> bool:
    """
    Get a boolean value from an environment variable.

    Args:
        name (str): The name of the environment variable.
        default (bool): The default value if the variable is not set.

    Returns:
        bool: The boolean value of the environment variable.
    """
    return os.getenv(name, str(default)).lower() in ("yes", "y", "true", "1", "t")

# Load API key and configuration for requiring API key from environment variables
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
REQUIRE_API_KEY = getenv_bool('REQUIRE_API_KEY', True)

if not API_KEY or API_KEY == 'your_api_key_here':
    raise ValueError("API_KEY must be set in the environment or .env file.")

def require_api_key(f):
    """
    Decorator to enforce API key authentication on routes.

    Args:
        f (function): The Flask route handler function to decorate.

    Returns:
        function: The decorated function that checks for a valid API key.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # If API key requirement is disabled, proceed without checking
        if not REQUIRE_API_KEY:
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.warning("Authorization header is missing.")
            return jsonify({"error": "Authorization header is missing"}), 401

        if not auth_header.startswith('Bearer '):
            logging.warning("Authorization header is malformed.")
            return jsonify({"error": "Authorization header must start with 'Bearer'"}), 401

        token = auth_header.split('Bearer ')[1]

        # Validate the token against the stored API key
        if token != API_KEY:
            logging.warning("Invalid API key provided.")
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)

    return decorated_function

# Mapping of audio formats to their corresponding MIME types
AUDIO_FORMAT_MIME_TYPES = json.loads(os.getenv("AUDIO_FORMAT_MIME_TYPES", """
{
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16"
}
"""))
