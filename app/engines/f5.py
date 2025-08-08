"""
F5 TTS engine plugin stub
"""
from app.engine import BaseEngine

class Engine(BaseEngine):
    """F5 TTS engine plugin using TTSHandler."""
    name = 'f5'

    def list_voices(self) -> list[str]:
        from app.server import tts_handler
        return [m['name'] for m in tts_handler.list_available_models()]

    def generate(self, text: str, voice: str, **opts) -> tuple[int, bytes]:
        """Generate speech via TTSHandler and return audio bytes."""
        from app.server import tts_handler
        out_path = tts_handler.generate_speech(
            text=text, voice=voice,
            response_format=opts.get('response_format'),
            speed=opts.get('speed'),
            ref_audio=opts.get('ref_audio')
        )
        import soundfile as sf
        info = sf.info(out_path)
        sr = info.samplerate
        with open(out_path, 'rb') as f:
            data = f.read()
        return (sr, data)
