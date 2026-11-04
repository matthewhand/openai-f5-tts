"""Offline regression for the immutable F5-TTS upstream source reference."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DependencyPinContracts(unittest.TestCase):
    def test_upstream_is_single_immutable_git_revision(self):
        lines = (ROOT / "requirements.txt").read_text().splitlines()
        pins = [line for line in lines if line.startswith("git+https://github.com/SWivid/F5-TTS.git")]
        self.assertEqual(pins, [
            "git+https://github.com/SWivid/F5-TTS.git@600e5fdfbf319f84735557aca80a7aaf60d487e6"
        ])
        self.assertRegex(pins[0], r"@[0-9a-f]{40}$")
        self.assertEqual(lines.count("transformers>=4.46.2,<5.0.0"), 1)
        self.assertNotIn("transformers>=4.46.2", lines)

    def test_source_pin_does_not_claim_complete_build(self):
        note = (ROOT / "docs" / "issue21-upstream-pin-handoff.md").read_text()
        self.assertIn("NOT a complete reproducible build", note)
        self.assertIn("Unmet acceptance gates", note)
        self.assertIn("not yet fully locked", note)


if __name__ == "__main__":
    unittest.main()
