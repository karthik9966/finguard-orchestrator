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

from src.config import get_config

# Every size budget lives in config.yaml (LLD §8) and is read *at call time*, through a `None`
# default rather than a module constant. The distinction is not stylistic: a module-level
# `X = get_config().chunking.min_chars` freezes at import, so editing config.yaml and resetting the
# cache would change nothing and the file would look live while being dead. A chunk shorter than
# `min_chars` is usually a stub that retrieves badly on its own; longer than `max_chars` and the
# reranker has to carry too much irrelevant text into the prompt.

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


def split_table(text: str, *, max_chars: int | None = None) -> list[str]:
    """Split a table region by rows, repeating the header so each chunk stands alone.

    The GLO glossary is a single 152k-character passage of ``term<TAB>definition`` rows.
    Semantic distance between "1P" and "1U" tells you nothing; row grouping does.
    """
    max_chars = get_config().chunking.max_chars if max_chars is None else max_chars
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
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Glue sentences into chunks, honouring boundaries but enforcing the size budget."""
    chunking = get_config().chunking
    min_chars = chunking.min_chars if min_chars is None else min_chars
    max_chars = chunking.max_chars if max_chars is None else max_chars
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
    percentile: float | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Split ``text`` at points where adjacent sentences stop being about the same thing."""
    chunking = get_config().chunking
    percentile = chunking.percentile if percentile is None else percentile
    min_chars = chunking.min_chars if min_chars is None else min_chars
    max_chars = chunking.max_chars if max_chars is None else max_chars
    if has_table(text):
        return split_table(text, max_chars=max_chars)

    text = normalize(text)
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return _hard_split(text, max_chars)

    # Below this many sentences a percentile is not a statistic, it is noise.
    if len(sentences) < chunking.min_sentences_for_percentile:
        boundaries: set[int] = set()
    else:
        boundaries = boundary_indices(adjacent_distances(encode(sentences)), percentile)

    return assemble(sentences, boundaries, min_chars=min_chars, max_chars=max_chars)

# --- tier-aware splitting (LLD §2.1) --------------------------------------------------
#
# Law and guidance are chunked by *structure*, not by cosine distance. A statute already tells
# you where one obligation ends -- the paragraph lettering is the author's own boundary, and it
# is more reliable than any embedding. Guidance splits per red-flag bullet because that is what
# makes "which indicator matched" answerable at all: an indicator that shares a chunk with nine
# others cannot be cited on its own.

_DESIGNATOR = re.compile(r"^\((?P<label>[A-Za-z0-9]{1,4})\)\s*")
_ROMAN = frozenset({"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"})


def _kind(label: str) -> str:
    """Which lettering series a designator belongs to.

    ``(i)`` is genuinely ambiguous -- the ninth lowercase letter and the first roman numeral --
    and 31 CFR 1020.320 runs to ``(g)``, so a section one subsection longer would reach it. The
    caller disambiguates with the stack; this returns the reading that is right far more often,
    because CFR nests ``(a)(1)(i)`` and a ninth lettered subsection is rare.
    """
    if label.isdigit():
        return "digit"
    if label.isupper():
        return "upper"
    return "roman" if label.lower() in _ROMAN else "lower"


def _successor(kind: str, previous: str, current: str) -> bool:
    """Is ``current`` the next label after ``previous`` in its series?"""
    if kind == "digit":
        return current.isdigit() and previous.isdigit() and int(current) == int(previous) + 1
    if kind in {"lower", "upper"}:
        return len(previous) == len(current) == 1 and ord(current) == ord(previous) + 1
    order = sorted(_ROMAN, key=lambda r: (len(r), r))
    ordered = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
    return (
        previous in ordered
        and current in ordered
        and ordered.index(current) == ordered.index(previous) + 1
    )


def paragraph_ref(stack: list[tuple[str, str]], label: str) -> str:
    """Advance ``stack`` by one designator and return the full path, e.g. ``(a)(2)(i)``.

    CFR nests ``(a) (1) (i) (A) (1) (i)``, so a designator's *type* does not determine its depth:
    31 CFR 1020.320(e) contains ``(1)`` at depth two and again at depth five. The rule that
    resolves it on real text is **deepest successor wins** -- look for the deepest level of the
    same series whose current label this one immediately follows, and continue there; otherwise
    this designator opens a new, deeper level.

    Mutates ``stack`` so a caller can walk paragraphs in one pass.
    """
    kind = _kind(label)
    for depth in range(len(stack) - 1, -1, -1):
        seen_kind, seen_label = stack[depth]
        if seen_kind == kind and _successor(kind, seen_label, label):
            del stack[depth:]
            stack.append((kind, label))
            return "".join(f"({value})" for _, value in stack)

    stack.append((kind, label))
    return "".join(f"({value})" for _, value in stack)


def split_by_section(text: str, *, section: str) -> list[tuple[str, str]]:
    """Statute or regulation → ``(section_ref, text)`` per lettered paragraph.

    ``section`` is the citation stem (``§ 1020.320``); each paragraph's designator path is
    appended to it. Paragraphs with no designator attach to the one before them, because an
    unlettered continuation is part of its parent obligation, not a rule of its own.
    """
    chunks: list[tuple[str, str]] = []
    stack: list[tuple[str, str]] = []

    for paragraph in (block.strip() for block in text.split("\n\n")):
        if not paragraph:
            continue
        match = _DESIGNATOR.match(paragraph)
        if match is None:
            if chunks:
                ref, body = chunks[-1]
                chunks[-1] = (ref, f"{body}\n{paragraph}")
            else:
                # A section with no lettering at all. 31 CFR 1010.311 -- the CTR obligation and
                # the source of the $10,000 threshold -- is one unlettered paragraph, and
                # dropping undesignated text for want of a parent silently removed the entire
                # section from the index.
                chunks.append((section, paragraph))
            continue
        ref = paragraph_ref(stack, match.group("label"))
        chunks.append((f"{section}{ref}", paragraph))

    return chunks


def split_cfr_xml(xml: str) -> list[tuple[str, str]]:
    """eCFR XML → ``(section_ref, text)``.

    The acquisition layer stores eCFR XML rather than flattened text precisely so this is
    possible: ``DIV8/HEAD/P`` carries the section number and paragraph order that a real
    ``section_ref`` needs. ``test_cfr_sections_are_stored_as_structured_xml`` pins that shape.
    """
    number = re.search(r'<DIV8[^>]*\sN="([^"]+)"', xml)
    section = f"§ {number.group(1)}" if number else "§"
    paragraphs = [
        normalize(re.sub(r"<[^>]+>", "", raw)).strip()
        for raw in re.findall(r"<P>(.*?)</P>", xml, re.DOTALL)
    ]
    return split_by_section("\n\n".join(p for p in paragraphs if p), section=section)


_BULLET = re.compile(r"^\s*[\u2022\u25cf\u25aa\u00b7\-\*]\s+")
_HEADING_MAX_WORDS = 10


def _is_heading(line: str) -> bool:
    """A short, unpunctuated line that introduces the bullets under it.

    The comma test is doing real work. PDF extraction wraps prose at the page width, so a
    sentence's first line arrives looking exactly like a heading -- short, capitalised, no
    terminal period. "In May 2009, the Basel Committee on Banking Supervision is" collected
    thirteen indicators under it before this rejected it. Section headings do not contain commas.
    """
    stripped = line.strip()
    if not stripped or _BULLET.match(stripped) or len(stripped.split()) > _HEADING_MAX_WORDS:
        return False
    if "," in stripped or stripped.endswith((".", ";", ":")):
        return False
    return stripped[0].isupper()


# A running header or footer repeats on every page, so two occurrences is the floor -- several of
# these documents are only two pages long, and a threshold of three left their footers in. The
# alternative, pattern-matching "Manual F-8 2/27/2015.V2", only works until the next publisher.
_CHROME_REPEATS = 2
_DIGITS = re.compile(r"\d+")


def _page_chrome(lines: list[str]) -> frozenset[str]:
    """Lines that repeat often enough to be page furniture rather than content.

    Without this, FFIEC's footer is read as a heading and 14 of Appendix F's indicators are filed
    under "FFIEC BSA/AML Examination Manual F-8 2/27/2015.V2" -- a section_ref no reviewer can
    look up, which is the one thing section_ref exists to be.
    """
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and not _BULLET.match(stripped):
            # Digits are blanked before counting because the page number is the one part of a
            # footer that changes: "Manual F-8 2/27/2015.V2" and "Manual F-9 ..." are the same
            # furniture, and counting them literally finds no repeat at all.
            counts[_DIGITS.sub("#", stripped)] = counts.get(_DIGITS.sub("#", stripped), 0) + 1
    return frozenset(
        line.strip()
        for line in lines
        if line.strip() and counts.get(_DIGITS.sub("#", line.strip()), 0) >= _CHROME_REPEATS
    )


def partition_guidance(
    text: str, *, fallback_heading: str = "Guidance"
) -> tuple[list[tuple[str, str]], str]:
    """Split guidance into ``(indicator chunks, the prose around them)``.

    FFIEC guidance is not a bullet list *or* narrative -- it is narrative with red-flag lists
    embedded in it. Taking only the bullets discards the SAR filing requirements; chunking the
    whole document semantically buries each indicator among nine others and makes "which
    indicator matched" unanswerable. So both come out, and the caller chunks the prose with the
    encoder it already has.
    """
    lines = text.splitlines()
    chrome = _page_chrome(lines)

    chunks: list[tuple[str, str]] = []
    prose: list[str] = []
    heading, ordinal = fallback_heading, 0
    current: list[str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            body = normalize(" ".join(current)).strip()
            if body:
                chunks.append((f"{heading} ¶ {ordinal}", body))
        current = None

    for line in lines:
        stripped = line.strip()
        if stripped in chrome:
            continue
        if _BULLET.match(line):
            flush()
            ordinal += 1
            current = [_BULLET.sub("", line).strip()]
        elif current is not None and stripped and not _is_heading(line):
            # Only a *wrapped* line continues a bullet. Unbounded, the rule lets the first
            # bullet of a narrative document swallow every line until the next heading:
            # ffiec-manual-sar collapsed 50k characters into two chunks that way, while
            # reporting 86% bullet coverage so nothing downstream noticed.
            if " ".join(current).rstrip().endswith((".", ";", "!", "?")):
                flush()
                prose.append(stripped)
            else:
                current.append(stripped)
        elif _is_heading(line):
            flush()
            heading, ordinal = stripped, 0
            prose.append(stripped)
        elif stripped:
            prose.append(stripped)

    flush()
    return chunks, "\n".join(prose)


def split_red_flag_bullets(text: str, *, fallback_heading: str = "Guidance") -> list[tuple[str, str]]:
    """The indicator half of :func:`partition_guidance`, for callers that only want red flags."""
    chunks, _ = partition_guidance(text, fallback_heading=fallback_heading)
    return chunks
