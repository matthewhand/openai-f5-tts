"""Dependency-free sentence splitter for the streaming TTS endpoint.

Why not NLTK: sent_tokenize needs the `punkt_tab` data downloaded at runtime,
which fails offline / in locked-down containers and then breaks every request.
For TTS chunking we only need a good-enough, predictable split.
"""
import re

# Words that end in '.' but do not end a sentence.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
    "inc", "ltd", "co", "e.g", "i.e", "approx", "dept", "fig",
})

# Latin terminators need trailing whitespace; CJK terminators do not.
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+|[。！？]+[\"')\]」』）]*\s*")


def _is_abbreviation(prefix: str) -> bool:
    m = re.search(r"([A-Za-z.]+?)\.$", prefix)
    return bool(m) and m.group(1).lower().rstrip(".") in _ABBREVIATIONS


def split_sentences(text: str) -> list:
    """Split text into sentence-sized chunks, preserving order. Never returns empties."""
    out = []
    for line in re.split(r"[\r\n]+", text or ""):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            end = m.end()
            terminator = m.group(0).strip()
            latin = terminator[0] in ".!?"
            if latin:
                nxt = line[end:end + 1]
                if nxt and nxt.islower():  # "Wait... what?" / "e.g. this"
                    continue
                if terminator.startswith(".") and not terminator.startswith("..") \
                        and _is_abbreviation(line[start:m.start() + 1]):
                    continue
            piece = line[start:end].strip()
            if piece:
                out.append(piece)
            start = end
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out
