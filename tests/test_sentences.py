import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from sentences import split_sentences  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("Hello. World!", ["Hello.", "World!"]),
    ("", []),
    ("   \n\n  ", []),
    ("No terminal punctuation", ["No terminal punctuation"]),
    ("Dr. Smith arrived. He sat.", ["Dr. Smith arrived.", "He sat."]),
    ("Use e.g. this one. Done.", ["Use e.g. this one.", "Done."]),
    ("It costs $3.50 today. Fine.", ["It costs $3.50 today.", "Fine."]),
    ("3.14 is pi. Yes.", ["3.14 is pi.", "Yes."]),
    ("Wait... what? Really!", ["Wait... what?", "Really!"]),
    ('She said "stop." Then left.', ['She said "stop."', "Then left."]),
    ("Line one.\n\nLine two.\nLine three", ["Line one.", "Line two.", "Line three"]),
    ("你好。今天天气很好！你呢？", ["你好。", "今天天气很好！", "你呢？"]),
    ("Mixed 你好。Then English. Ok", ["Mixed 你好。", "Then English.", "Ok"]),
])
def test_split(text, expected):
    assert split_sentences(text) == expected


def test_never_returns_empty_chunks_and_preserves_all_words():
    text = "A b.  C d?!  \n E f... g h."
    parts = split_sentences(text)
    assert all(p.strip() for p in parts)
    assert " ".join(parts).split() == text.split()
