"""Turn the regulatory corpus into chunk records ready for ChromaDB (§3.3 -> §3.4).

Two paths, because the two corpora are structurally different:

**Path A -- ObliQA.** The 13,732 passages are *already* split, one legal clause each, and each
carries a ``PassageID`` (``14.2.3.Guidance.1.``) that is exactly the ``section_clause`` §3.4
mandates and §6.4's citations drawer displays. Re-chunking them wholesale would merge clauses
and degrade every citation to "somewhere in 14.2.3". So this path filters and normalizes, and
only reaches for the cosine chunker on the 74 passages that are genuinely oversized.

**Path B -- the four PDFs and two scraped FINRA rules.** Continuous prose with no section IDs.
This is the "unstructured document feed" the blueprint's cosine-distance rule was written for.

Usage::

    uv run python -m src.ingestion.loader                      # minilm
    uv run python -m src.ingestion.loader --backend openai
    uv run python -m src.ingestion.loader --backend both
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader

from src.ingestion.chunker import (
    DEFAULT_PERCENTILE,
    MAX_CHARS,
    MIN_CHARS,
    chunk_semantic,
    normalize,
    strip_invisibles,
    strip_provenance_header,
)
from src.ingestion.embeddings import BACKENDS, MissingCredentials, get_backend
from src.ingestion.obliqa_map import load_document_map

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REGULATIONS = DATA_DIR / "raw" / "regulations"
OBLIQA_DOCS = REGULATIONS / "obliqa" / "StructuredRegulatoryDocuments"
CHUNK_DIR = DATA_DIR / "processed" / "chunks"

# Passages this short are headings or numbering artefacts, not retrievable content.
MIN_PASSAGE_CHARS = 40
# Below this a chunk is a fragment like "(c) enquire into the background..."; prefixing the
# document and clause makes it self-describing without changing its citation.
CONTEXT_PREFIX_BELOW = 200

# Tiers from the §3.2 corpus analysis: only 2.9% of ObliQA passages are AML-bearing and 62% of
# those sit in Document 1. Tier 3 is kept indexed as the distractor set that makes §8.1 Context
# Precision measurable -- it is filtered at query time, not at index time.
TIER_1 = frozenset({1, 7, 8, 10, 17, 27})
TIER_2 = frozenset({3, 15, 16, 19, 22, 23, 34, 40})

# Publication dates for the four PDFs. Taken from the documents themselves (FINRA 19-18 prints
# "May 6, 2019" in its footer) or the publisher's URL path, not guessed.
PDF_SOURCES: dict[str, dict] = {
    "regulatory-notice-19-18.pdf": {
        "title": "FINRA Regulatory Notice 19-18",
        "corpus": "finra",
        "date": "2019-05-06",
        "tier": 1,
    },
    "fin-2022-alert002-russian-elites.pdf": {
        "title": "FinCEN Alert FIN-2022-Alert002 (Russian elites, high-value assets)",
        "corpus": "fincen",
        "date": "2022-03-01",
        "tier": 1,
    },
    "fin-2022-alert003-fincen-bis-joint.pdf": {
        "title": "FinCEN/BIS Joint Alert FIN-2022-Alert003 (export control evasion)",
        "corpus": "fincen",
        "date": "2022-06-01",
        "tier": 1,
    },
    "fin-2023-alert002-commercial-real-estate.pdf": {
        "title": "FinCEN Alert FIN-2023-Alert002 (commercial real estate)",
        "corpus": "fincen",
        "date": "2023-01-25",
        "tier": 1,
    },
}

FINRA_RULE_TITLES = {
    "3310": "FINRA Rule 3310. Anti-Money Laundering Compliance Program",
    "3110": "FINRA Rule 3110. Supervision",
}

# --- dates ---------------------------------------------------------------------------

_MONTHS = (
    "january february march april may june july august september october november december"
).split()

_VERSION_STAMP = re.compile(r"VER\d+\.(\d{2})(\d{2})(\d{2})")
_TRAILING_STAMP = re.compile(r"_(\d{2})(\d{2})(\d{2})\b")
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
_MONTH_YEAR = re.compile(r"\b([A-Za-z]+)\s+(\d{4})\b")


_BARE_YEAR = re.compile(r"\b(19|20)(\d{2})\b")


def document_date(source_file: str) -> str | None:
    """Recover a publication date from an ObliQA filename, at the best precision available.

    28 of the 40 carry a ``VERxx.DDMMYY`` stamp; the rest date themselves in the title
    (``FSMR (Consolidated_December 2023)``). Returns ``YYYY-MM-DD``, ``YYYY-MM`` or ``YYYY``
    depending on what the source actually states -- padding a bare year to January 1st would
    invent a precision the document does not have. Four documents carry no date at all and
    return ``None``; §3.4's ``last_updated_date`` is legitimately nullable.
    """
    if match := _VERSION_STAMP.search(source_file):
        day, month, year = match.groups()
        return f"20{year}-{month}-{day}"
    if match := _TRAILING_STAMP.search(source_file):
        day, month, year = match.groups()
        return f"20{year}-{month}-{day}"

    # Underscores and hyphens are word characters, so "Consolidated_December" hides the month
    # from a \b-anchored pattern. Flatten separators before matching prose dates.
    flat = re.sub(r"[_\-]+", " ", source_file)

    # Scan every match, not just the first: "CRS Regulations 2017 (Consolidated October 2023)"
    # offers "Regulations 2017" before the real month, and the consolidation date is the one
    # that means "last updated".
    for match in _DAY_MONTH_YEAR.finditer(flat):
        day, month_name, year = match.groups()
        if month_name.lower() in _MONTHS:
            return f"{year}-{_MONTHS.index(month_name.lower()) + 1:02d}-{int(day):02d}"
    for match in _MONTH_YEAR.finditer(flat):
        month_name, year = match.groups()
        if month_name.lower() in _MONTHS:
            return f"{year}-{_MONTHS.index(month_name.lower()) + 1:02d}"
    if match := _BARE_YEAR.search(flat):
        return match.group(0)
    return None


def tier_for(document_id: int) -> int:
    if document_id in TIER_1:
        return 1
    return 2 if document_id in TIER_2 else 3


# --- Path A: ObliQA ------------------------------------------------------------------


def is_bare_heading(text: str) -> bool:
    """True for numbering stubs and all-caps section titles carrying no obligation."""
    stripped = text.strip()
    return len(stripped) < MIN_PASSAGE_CHARS


def context_prefix(document_title: str, section_clause: str) -> str:
    return f"{document_title} - {section_clause}: "


def obliqa_chunks(encode, *, percentile: float) -> Iterator[dict]:
    document_map = load_document_map()
    for path in sorted(OBLIQA_DOCS.glob("*.json"), key=lambda p: p.name):
        for passage in json.loads(path.read_text()):
            document_id = passage["DocumentID"]
            raw = passage["Passage"] or ""
            if is_bare_heading(raw):
                continue

            meta = document_map[document_id]
            title = meta["title"]
            source_file = meta["source_file"]
            # One PassageID carries a U+200E mark. The gold set's copies carry none, so
            # stripping it both cleans the citation and fixes a would-be scoring miss.
            clause = strip_invisibles(passage["PassageID"]).strip()

            pieces = chunk_semantic(
                raw, encode, percentile=percentile, min_chars=MIN_CHARS, max_chars=MAX_CHARS
            )
            multi = len(pieces) > 1
            for index, text in enumerate(pieces, start=1):
                if not text.strip():
                    continue
                if len(text) < CONTEXT_PREFIX_BELOW:
                    text = context_prefix(title, clause) + text
                suffix = f"#{index}" if multi else ""
                # (DocumentID, PassageID) is NOT unique: 17 keys collide across 44 passages
                # in documents 7, 11, 13 and 17, with different text each time. The per-passage
                # UUID is ObliQA's real primary key, so it disambiguates the chunk_id while
                # section_clause keeps carrying the citation a human reads.
                yield {
                    "chunk_id": f"obliqa:{document_id}:{clause}:{passage['ID'][:8]}{suffix}",
                    "passage_uuid": passage["ID"],
                    "text": text,
                    "source_file": source_file,
                    "section_clause": clause,
                    "last_updated_date": document_date(source_file),
                    "corpus": "obliqa",
                    "document_id": document_id,
                    "document_title": title,
                    "relevance_tier": tier_for(document_id),
                    "jurisdiction": "ADGM",
                    "part": f"{index} of {len(pieces)}" if multi else None,
                }


# --- Path B: PDFs and scraped rules ---------------------------------------------------

_BULLET_ARTEFACT = re.compile(r"(?m)^\s*00\s+")


def strip_page_furniture(pages: list[str], *, min_pages: int = 3) -> str:
    """Drop lines that repeat across pages -- running headers, footers, page numbers.

    FINRA 19-18 prints ``May 6, 201919-18`` on six of its twelve pages. Embedded verbatim it
    would appear in several chunks and pull unrelated queries toward whichever one it landed in.
    """
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.split("\n") if line.strip()})
    furniture = {line for line, count in counts.items() if count >= min_pages}

    kept: list[str] = []
    for page in pages:
        for line in page.split("\n"):
            if line.strip() and line.strip() not in furniture:
                kept.append(line)
    return "\n".join(kept)


def pdf_chunks(encode, *, percentile: float) -> Iterator[dict]:
    for path in sorted(REGULATIONS.rglob("*.pdf")):
        source = PDF_SOURCES.get(path.name)
        if source is None:
            raise RuntimeError(f"no metadata registered for {path.name}")

        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        text = normalize(_BULLET_ARTEFACT.sub("", strip_page_furniture(pages)))
        pieces = chunk_semantic(
            text, encode, percentile=percentile, min_chars=MIN_CHARS, max_chars=MAX_CHARS
        )
        for index, piece in enumerate(pieces, start=1):
            yield {
                "chunk_id": f"{source['corpus']}:{path.stem}:{index}",
                "text": piece,
                "source_file": path.name,
                "section_clause": f"part {index} of {len(pieces)}",
                "last_updated_date": source["date"],
                "corpus": source["corpus"],
                "document_id": None,
                "document_title": source["title"],
                "relevance_tier": source["tier"],
                "jurisdiction": "US",
                "part": f"{index} of {len(pieces)}",
            }


def finra_rule_chunks(encode, *, percentile: float) -> Iterator[dict]:
    for path in sorted((REGULATIONS / "finra").glob("finra-rule-*.txt")):
        number = path.stem.rsplit("-", 1)[-1]
        retrieved = next(
            (
                line.split(":", 1)[1].strip()
                for line in path.read_text().splitlines()
                if line.startswith("# Retrieved:")
            ),
            None,
        )
        text = normalize(strip_provenance_header(path.read_text()))
        pieces = chunk_semantic(
            text, encode, percentile=percentile, min_chars=MIN_CHARS, max_chars=MAX_CHARS
        )
        for index, piece in enumerate(pieces, start=1):
            yield {
                "chunk_id": f"finra:rule-{number}:{index}",
                "text": piece,
                "source_file": path.name,
                "section_clause": f"Rule {number} part {index} of {len(pieces)}",
                "last_updated_date": retrieved,
                "corpus": "finra",
                "document_id": None,
                "document_title": FINRA_RULE_TITLES[number],
                "relevance_tier": 1,
                "jurisdiction": "US",
                "part": f"{index} of {len(pieces)}",
            }


# --- entry point ---------------------------------------------------------------------


def build(backend_name: str, *, percentile: float = DEFAULT_PERCENTILE) -> Path:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHUNK_DIR / f"{backend_name}.jsonl"

    with get_backend(backend_name) as backend:
        records = [
            *obliqa_chunks(backend.encode, percentile=percentile),
            *finra_rule_chunks(backend.encode, percentile=percentile),
            *pdf_chunks(backend.encode, percentile=percentile),
        ]

    seen = Counter(record["chunk_id"] for record in records)
    if duplicates := [cid for cid, count in seen.items() if count > 1]:
        raise RuntimeError(f"duplicate chunk_ids: {duplicates[:5]}")

    with dest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_corpus = Counter(record["corpus"] for record in records)
    lengths = sorted(len(record["text"]) for record in records)
    undated = sum(1 for record in records if record["last_updated_date"] is None)
    print(f"  {backend_name}: {len(records):,} chunks -> {dest.relative_to(PROJECT_ROOT)}")
    print(f"    by corpus   {dict(by_corpus)}")
    print(
        f"    chars       median {lengths[len(lengths) // 2]}  "
        f"p95 {lengths[int(len(lengths) * 0.95)]}  max {lengths[-1]}"
    )
    print(f"    undated     {undated:,} chunks have no last_updated_date")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="minilm", choices=[*BACKENDS, "both"])
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    args = parser.parse_args()

    names = list(BACKENDS) if args.backend == "both" else [args.backend]
    failures = 0
    for name in names:
        try:
            build(name, percentile=args.percentile)
        except MissingCredentials as error:
            print(f"  {name}: SKIPPED -- {error}")
            failures += 1
    return 1 if failures and len(names) == 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
