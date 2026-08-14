
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from alignment import estimate_alignment, words_from_alignment, build_alignment


def test_estimate_alignment_covers_text_and_duration():
    text = "Hello from F5."
    al = estimate_alignment(text, 2.0)
    assert "".join(al["characters"]) == text
    assert len(al["character_start_times_seconds"]) == len(text)
    assert al["character_start_times_seconds"][0] == 0.0
    assert al["character_end_times_seconds"][-1] == 2.0
    words = words_from_alignment(al)
    assert [w["word"] for w in words] == ["Hello", "from", "F5"]


def test_build_alignment_null_when_file_missing():
    al, words, source = build_alignment("Hello", "Z:/definitely-missing.wav")
    assert al is None
    assert words == []
    assert source == "unavailable"
