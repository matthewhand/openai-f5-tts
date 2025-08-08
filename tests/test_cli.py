import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch
from app.cli import main, PORT, app

def test_cli_parser():
    """Test that the CLI parser works correctly."""
    with patch('sys.argv', ['openai-f5-tts', 'serve', '--port', '8080']):
        with patch('app.cli.app.run') as mock_run:
            with patch('app.cli.WSGIServer') as mock_wsgi:
                # This will raise SystemExit when it tries to run the server
                try:
                    main()
                except SystemExit:
                    pass
                
                # In debug mode, it should call app.run
                mock_run.assert_not_called()
                
                # In production mode, it should use WSGIServer
                mock_wsgi.assert_called_once()
                args, kwargs = mock_wsgi.call_args
                assert args[0][1] == 8080  # Check that port was set correctly

def test_cli_debug_mode():
    """Test that debug flag invokes app.run and skips WSGIServer."""
    with patch('sys.argv', ['openai-f5-tts', 'serve', '--debug', '--host', '127.0.0.1', '--port', '8001']):
        with patch('app.cli.app.run') as mock_run:
            with patch('app.cli.WSGIServer') as mock_wsgi:
                main()
                mock_run.assert_called_once_with(host='127.0.0.1', port=8001, debug=True)
                mock_wsgi.assert_not_called()

def test_cli_defaults():
    """Test that default host/port are used when no args provided."""
    with patch('sys.argv', ['openai-f5-tts', 'serve']):
        with patch('app.cli.app.run') as mock_run:
            with patch('app.cli.WSGIServer') as mock_wsgi:
                main()
                mock_run.assert_not_called()
                # WSGIServer is called with default host and PORT
                call_args = mock_wsgi.call_args[0]
                assert call_args[0] == ('0.0.0.0', PORT)
                # Ensure the Flask app is passed as second argument
                assert call_args[1] == app

def test_cli_list_engines(monkeypatch, capsys):
    """CLI: list-engines prints engine names."""
    import app.cli as cli
    monkeypatch.setattr(cli, 'ENGINES', {'a': None, 'b': None})
    with patch('sys.argv', ['openai-f5-tts', 'list-engines']):
        cli.main()
    out = capsys.readouterr().out.strip().splitlines()
    assert sorted(out) == ['a', 'b']

def test_cli_list_voices(monkeypatch, capsys):
    """CLI: list-voices prints voices for engine."""
    import app.cli as cli
    class Dummy:
        def list_voices(self):
            return ['x', 'y']
    monkeypatch.setattr(cli, 'ENGINES', {'dummy': Dummy()})
    with patch('sys.argv', ['openai-f5-tts', 'list-voices', '--engine', 'dummy']):
        cli.main()
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ['x', 'y']

def test_cli_speak(monkeypatch, tmp_path, capsys):
    """CLI: speak generates audio file and prints confirmation."""
    import app.cli as cli
    class Dummy:
        def generate(self, text, voice, **opts):
            return (22050, b'data')
    monkeypatch.setattr(cli, 'ENGINES', {'dummy': Dummy()})
    output = tmp_path / 'out.raw'
    with patch('sys.argv', ['openai-f5-tts', 'speak', '--engine', 'dummy', '--text', 'hello', '--voice', 'v', '--output', str(output)]):
        cli.main()
    assert output.read_bytes() == b'data'
    assert 'Generated speech saved to' in capsys.readouterr().out

# CLI mix command tests
def test_cli_mix_overlay(tmp_path):
    import numpy as np, soundfile as sf
    from app.cli import main
    from unittest.mock import patch
    # Create sample WAV files
    arr1 = np.array([1.0, 2.0], dtype='float32')
    arr2 = np.array([3.0], dtype='float32')
    f1 = tmp_path / 'a.wav'
    f2 = tmp_path / 'b.wav'
    sf.write(str(f1), arr1, 16000, subtype='FLOAT')
    sf.write(str(f2), arr2, 16000, subtype='FLOAT')
    out = tmp_path / 'out.wav'
    with patch('sys.argv', ['openai-f5-tts', 'mix', '--inputs', str(f1), str(f2), '--mode', 'overlay', '--output', str(out)]):
        main()
    data, sr = sf.read(str(out))
    assert sr == 16000
    assert np.allclose(data, np.array([(1 + 3) / 2, (2 + 0) / 2], dtype=data.dtype))

def test_cli_mix_concat(tmp_path):
    import numpy as np, soundfile as sf
    from app.cli import main
    from unittest.mock import patch
    # Create sample WAV files
    arr1 = np.array([0.1, 0.2], dtype='float32')
    arr2 = np.array([0.3], dtype='float32')
    f1 = tmp_path / 'c.wav'
    f2 = tmp_path / 'd.wav'
    sf.write(str(f1), arr1, 44100, subtype='FLOAT')
    sf.write(str(f2), arr2, 44100, subtype='FLOAT')
    out = tmp_path / 'out2.wav'
    with patch('sys.argv', ['openai-f5-tts', 'mix', '--inputs', str(f1), str(f2), '--mode', 'concat', '--output', str(out)]):
        main()
    data, sr = sf.read(str(out))
    assert sr == 44100
    assert np.allclose(data, np.concatenate([arr1, arr2]))