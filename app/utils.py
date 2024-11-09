# utils.py

from flask import request, jsonify
from functools import wraps
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

        # Check for Authorization header with Bearer token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid API key"}), 401

        token = auth_header.split('Bearer ')[1]
        
        # Validate the token against the stored API key
        if token != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

# Mapping of audio formats to their corresponding MIME types
AUDIO_FORMAT_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16"
}
