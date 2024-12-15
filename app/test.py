import os
from dotenv import load_dotenv
from tts_handler import TTSHandler

# Load environment variables from .env file
load_dotenv()

# Fetch the REF_AUDIO_PATH from the environment
# REF_AUDIO_PATH = os.getenv('REF_AUDIO_PATH')

# Simple test parameters
text = "Hello, this is a test."
voice = "Emilia"
response_format = "mp3"
speed = 1.0

try:
    # Verify reference audio path
    # if not REF_AUDIO_PATH or not os.path.exists(REF_AUDIO_PATH):
        # raise FileNotFoundError(f"Reference audio file specified in REF_AUDIO_PATH not found: {REF_AUDIO_PATH}")

    # Initialize TTSHandler
    tts_handler = TTSHandler(default_voice=voice)

    # Test generate_speech with reference audio
    output_file_path = tts_handler.generate_speech(
        text=text,
        voice=voice,
        response_format=response_format,
        speed=speed
    )
    print(f"Generated speech saved to: {output_file_path}")

except Exception as e:
    print(f"Error generating speech: {e}")
