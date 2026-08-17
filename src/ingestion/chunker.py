"""Semantic chunking by cosine distance (§3.3).

The blueprint's rule: split where the cosine distance between adjacent sentence embeddings
exceeds a *calculated* threshold, so a chunk holds one coherent legal thought rather than an
arbitrary window of characters.

Three things in this corpus stop that from being a one-liner:

* **Legal prose is full of false sentence boundaries.** ``Rule 8.3.1(1)(d)`` and ``e.g.`` both
  end in a period followed by a space. Splitting there fragments a clause mid-citation.
* **Tables are not prose.** 121 ObliQA passages carry ``/Table Start`` regions whose rows are
  tab-separated; the largest is a 152k-character glossary. Cosine distance between adjacent
  glossary entries is meaningless -- those split by row, with the header repeated.
* **A percentile threshold needs a population.** On a three-sentence passage the 95th
  percentile is just the largest of two numbers, so short text is left whole.

Everything here is a pure function: text in, text out. I/O and metadata live in ``loader.py``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

import numpy as np

# A chunk shorter than this is usually a stub that retrieves badly on its own; longer than
# this and the reranker has to carry too much irrelevant text into the prompt.
MIN_CHARS = 200
MAX_CHARS = 2000
DEFAULT_PERCENTILE = 95.0

# Below this many sentences a percentile is not a statistic, it is noise.
MIN_SENTENCES_FOR_PERCENTILE = 6

Encoder = Callable[[Sequence[str]], np.ndarray]


# --- normalization -------------------------------------------------------------------

# ObliQA text carries 4,467 U+200E marks and 971 U+F0FC (a Private Use Area codepoint --
# a Wingdings bullet that survived the publisher's PDF extraction). Both are invisible,
# both perturb embeddings, and neither means anything.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Co", "Cs"})

_PROVENANCE_HEADER = re.compile(r"\A(?:#[^\n]*\n)+\s*")
_TABLE_REGION = re.compile(r"/Table Start\n(.*?)\n/Table End", re.DOTALL)


def strip_invisibles(text: str) -> str:
    """Drop format/private-use codepoints that carry no meaning but shift embeddings."""
    return "".join(ch for ch in text if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES)


def normalize(text: str) -> str:
    """Canonicalise prose. Not for table regions -- tabs there are column separators."""
    text = unicodedata.normalize("NFKC", strip_invisibles(text))
    text = text.replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def strip_provenance_header(text: str) -> str:
    """Remove the ``#``-prefixed banner ``download.py`` writes onto scraped FINRA rules."""
    return _PROVENANCE_HEADER.sub("", text)


# --- sentence splitting --------------------------------------------------------------

# Tokens that end in a period without ending a sentence. Kept lowercase for comparison.
_ABBREVIATIONS = frozenset(
    """
    e.g i.e etc no nos art arts reg regs sch para paras cf vs approx incl
    mr mrs ms dr prof inc ltd plc llc co corp dept est fig vol ch ss
    """.split()
)

# A list marker opening a line -- "(a)", "(iii)", "3." -- starts a new unit even without
# terminal punctuation, because sub-paragraphs are what legal drafting splits on.
_LIST_OPENER = re.compile(r"^\s*(?:\(\w{1,4}\)|\d{1,2}\.)\s")
_CANDIDATE_END = re.compile(r"[.!?]+[\"')\]]*\s+")


def _is_real_boundary(text: str, start: int, end: int) -> bool:
    """Decide whether the punctuation at ``start`` genuinely ends a sentence."""
    before = text[:start]

    # "8.3.1(1)(d)" and "31 U.S.C. 5318" -- a digit either side of the period means it is
    # part of a reference, not a full stop.
    if before[-1:].isdigit() and text[end : end + 1].isdigit():
        return False

    last_token = re.split(r"[\s(\[]", before)[-1].rstrip(".").lower()
    if last_token in _ABBREVIATIONS:
        return False
    # A single trailing initial ("A." in "Schedule A. The") is ambiguous; treat one bare
    # letter as an abbreviation rather than risk cutting a clause in half.
    if len(last_token) == 1 and last_token.isalpha():
        return False

    nxt = text[end : end + 1]
    return nxt.isupper() or nxt in "(‘“\"'" or nxt == ""


def split_sentences(text: str) -> list[str]:
    """Split legal prose into sentence-ish units, preserving citations and list items."""
    units: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        # Lines opening with a list marker are their own unit even when the previous line
        # ran on without punctuation.
        if _LIST_OPENER.match(line) or not units:
            units.append(line.strip())
        else:
            units.append(line.strip())

    sentences: list[str] = []
    for unit in units:
        start = 0
        for match in _CANDIDATE_END.finditer(unit):
            if not _is_real_boundary(unit, match.start(), match.end()):
                continue
            piece = unit[start : match.end()].strip()
            if piece:
                sentences.append(piece)
            start = match.end()
        tail = unit[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


# --- tables --------------------------------------------------------------------------


def has_table(text: str) -> bool:
    return _TABLE_REGION.search(text) is not None


def split_table(text: str, *, max_chars: int = MAX_CHARS) -> list[str]:
    """Split a table region by rows, repeating the header so each chunk stands alone.

    The GLO glossary is a single 152k-character passage of ``term<TAB>definition`` rows.
    Semantic distance between "1P" and "1U" tells you nothing; row grouping does.
    """
    match = _TABLE_REGION.search(text)
    if match is None:
        return [normalize(text)]

    preamble = normalize(text[: match.start()])
    rows = [row for row in match.group(1).split("\n") if row.strip()]
    if not rows:
        return [preamble] if preamble else []

    header, body = rows[0], rows[1:]
    header_text = normalize(header.replace("\t", " | "))

    chunks: list[str] = []
    current: list[str] = []
    size = len(header_text)
    for row in body:
        rendered = normalize(row.replace("\t", " | "))
        # A single row can exceed the budget on its own (a glossary definition running to a
        # paragraph). Split it rather than emitting an oversized chunk.
        for part in _hard_split(rendered, max_chars - len(header_text) - 1):
            if current and size + len(part) + 1 > max_chars:
                chunks.append("\n".join([header_text, *current]))
                current, size = [], len(header_text)
            current.append(part)
            size += len(part) + 1
    if current:
        chunks.append("\n".join([header_text, *current]))

    if preamble:
        chunks[:0] = _hard_split(preamble, max_chars) if len(preamble) > max_chars else [preamble]
    return chunks


# --- cosine boundaries ---------------------------------------------------------------


def adjacent_distances(vectors: np.ndarray) -> np.ndarray:
    """Cosine distance between each consecutive pair of (L2-normalized) vectors."""
    return 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)


def boundary_indices(distances: np.ndarray, percentile: float) -> set[int]:
    """Indices where a new chunk starts, i.e. where similarity dropped far enough.

    The threshold is the document's *own* distance distribution rather than a constant:
    a dense rulebook and a discursive advisory have different baseline similarity, and a
    fixed 0.3 would over-split one and under-split the other.
    """
    if distances.size == 0:
        return set()
    threshold = float(np.percentile(distances, percentile))
    return {i + 1 for i, distance in enumerate(distances) if distance > threshold}


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Last resort for a single sentence longer than the budget: split on whitespace."""
    words, out, current = sentence.split(" "), [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > max_chars:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def assemble(
    sentences: Sequence[str],
    boundaries: set[int],
    *,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[str]:
    """Glue sentences into chunks, honouring boundaries but enforcing the size budget."""
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for index, sentence in enumerate(sentences):
        pending = len(" ".join([*current, sentence]))
        if index in boundaries and len(" ".join(current)) >= min_chars:
            flush()
        elif current and pending > max_chars:
            flush()
        if len(sentence) > max_chars:
            flush()
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        current.append(sentence)
    flush()

    # A trailing stub retrieves poorly alone; fold it back into its predecessor when that
    # does not blow the budget.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        merged = f"{chunks[-2]} {chunks[-1]}"
        if len(merged) <= max_chars:
            chunks[-2:] = [merged]
    return chunks


def chunk_semantic(
    text: str,
    encode: Encoder,
    *,
    percentile: float = DEFAULT_PERCENTILE,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[str]:
    """Split ``text`` at points where adjacent sentences stop being about the same thing."""
    if has_table(text):
        return split_table(text, max_chars=max_chars)

    text = normalize(text)
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return _hard_split(text, max_chars)

    if len(sentences) < MIN_SENTENCES_FOR_PERCENTILE:
        boundaries: set[int] = set()
    else:
        boundaries = boundary_indices(adjacent_distances(encode(sentences)), percentile)

    return assemble(sentences, boundaries, min_chars=min_chars, max_chars=max_chars)
