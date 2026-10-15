"""No import of torch, F5-TTS, Flask or model inference."""
import os
import sys
import unittest
from unittest.mock import patch

from f5_tts_app.cli import main


class CliTests(unittest.TestCase):
    def test_defaults_are_loopback(self):
        with patch("os.execv") as execute:
            main([])
        argv = execute.call_args.args[1]
        self.assertEqual(argv[-4:], ["--host", "127.0.0.1", "--port", "9090"])
        self.assertTrue(argv[1].endswith("app/server.py"))
        self.assertEqual(execute.call_args.args[0], sys.executable)

    def test_custom_host_and_port(self):
        with patch("os.execv") as execute:
            main(["--host", "127.0.0.2", "--port", "9099"])
        self.assertEqual(execute.call_args.args[1][-4:],
                         ["--host", "127.0.0.2", "--port", "9099"])

    def test_bad_ports_rejected_without_startup(self):
        for port in ("0", "-2", "65536", "notanumber"):
            with self.subTest(port=port), patch("os.execv") as execute:
                with self.assertRaises(SystemExit):
                    main(["--port", port])
                execute.assert_not_called()

    def test_debug_flag_is_forwarded_through_environment(self):
        with patch.dict(os.environ, {"DEBUG_MODE": "false"}):
            with patch("os.execv"):
                main(["--debug"])
            self.assertEqual(os.environ["DEBUG_MODE"], "true")


if __name__ == "__main__":
    unittest.main()
