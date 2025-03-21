import pytest
from unittest.mock import patch
from app.cli import main

def test_cli_parser():
    """Test that the CLI parser works correctly."""
    with patch('sys.argv', ['openai-f5-tts', '--port', '8080']):
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