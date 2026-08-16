"""Resolve ObliQA ``DocumentID`` values to the ADGM documents they came from.

The structured corpus stores only ``{ID, DocumentID, PassageID, Passage}`` -- there is no
document title anywhere in it, and the upstream repository's 40 standardized ``.txt`` files
are *not* in DocumentID order, so the mapping cannot be inferred from filename ordering.

Without this, every citation the agent produces reads "Document 1, section 1.1.1". §3.4
wants a ``source_file`` metadata field and §6.1 wants a human-readable inventory of loaded
regulations, so we recover the mapping by matching text.

Method: take each structured document's longest passages and test them as verbatim
substrings of each standardized ``.txt``. Long passages are effectively unique, so the
document that contains them is the source. The result is asserted to be injective -- 40
documents onto 40 distinct files -- which makes a silent mis-mapping impossible.

Usage::

    uv run python -m src.ingestion.obliqa_map           # build data/obliqa_document_map.json
    uv run python -m src.ingestion.obliqa_map --force   # rebuild
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import httpx

from src.ingestion.download import DATA_DIR, OBLIQA_DOCS, TIMEOUT, USER_AGENT

TXT_LISTING_URL = (
    "https://api.github.com/repos/RegNLP/ObliQADataset/contents/"
    "scripts/StandartizedRegulatoryDocumentsTXT"
)
MAP_PATH = DATA_DIR / "obliqa_document_map.json"
EXPECTED_DOCUMENTS = 40

# Titles taken verbatim from each document's own opening text. Everything else falls back
# to its cleaned upstream filename -- a slightly clumsy label beats an invented one.
CONFIRMED_TITLES = {
    "AML": "AML Rulebook",
    "CMC": "Code of Market Conduct",
    "CONF": "Confidentiality Policy",
    "FP": "Fund Passporting Rules",
    "GPM": "Guidance and Policies Manual",
    "IFR": "Islamic Finance Rules",
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_title(filename: str) -> str:
    """Turn ``AML_VER09.211223.txt`` into ``AML``, ``GPM VER03.120623.txt`` into ``GPM``."""
    stem = Path(filename).stem
    stem = re.sub(r"_?VER\s*\d+[._]\d+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\s*\)", "", stem)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -–")
    return CONFIRMED_TITLES.get(stem, stem)


def fetch_standardized_texts(client: httpx.Client) -> dict[str, str]:
    listing = client.get(TXT_LISTING_URL).raise_for_status().json()
    texts = {}
    for item in listing:
        if not item["name"].endswith(".txt"):
            continue
        body = client.get(item["download_url"]).raise_for_status().text
        texts[item["name"]] = normalise(body)
    if len(texts) != EXPECTED_DOCUMENTS:
        raise RuntimeError(f"expected {EXPECTED_DOCUMENTS} standardized texts, got {len(texts)}")
    return texts


def probes_for(passages: list[dict], count: int = 5) -> list[str]:
    """The longest passages, which are the least likely to appear in more than one document."""
    ordered = sorted((p["Passage"] or "" for p in passages), key=len, reverse=True)
    long_enough = [normalise(p) for p in ordered if len(p) > 200]
    return (long_enough or [normalise(p) for p in ordered])[:count]


def build_map(docs_dir: Path, texts: dict[str, str]) -> dict:
    documents: dict[str, dict] = {}
    claimed: dict[str, int] = {}

    for path in sorted(docs_dir.glob("*.json"), key=lambda p: int(p.stem)):
        passages = json.loads(path.read_text())
        doc_id = passages[0]["DocumentID"]
        probes = probes_for(passages)

        hits = Counter({name: sum(pr in body for pr in probes) for name, body in texts.items()})
        best, score = hits.most_common(1)[0]
        winners = [name for name, n in hits.items() if n == score and n > 0]

        if score == 0:
            raise RuntimeError(f"DocumentID {doc_id}: no standardized text contains its passages")
        if len(winners) > 1:
            raise RuntimeError(f"DocumentID {doc_id}: ambiguous match across {winners}")
        if best in claimed:
            raise RuntimeError(
                f"DocumentID {doc_id} and {claimed[best]} both matched {best}; mapping is not injective"
            )
        claimed[best] = doc_id

        documents[str(doc_id)] = {
            "title": clean_title(best),
            "source_file": best,
            "passages": len(passages),
            "matched_probes": f"{score}/{len(probes)}",
        }

    if len(documents) != EXPECTED_DOCUMENTS:
        raise RuntimeError(f"expected {EXPECTED_DOCUMENTS} documents, mapped {len(documents)}")
    return {
        "schema": 1,
        "generated": date.today().isoformat(),
        "source": "https://github.com/RegNLP/ObliQADataset (StandartizedRegulatoryDocumentsTXT)",
        "note": "All 40 documents are Abu Dhabi Global Market (ADGM) regulatory publications.",
        "documents": documents,
    }


def load_document_map() -> dict[int, dict]:
    """Read the built map, keyed by int DocumentID -- for the §3.4 loader and §6.1 inventory."""
    payload = json.loads(MAP_PATH.read_text())
    return {int(k): v for k, v in payload["documents"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild an existing map")
    args = parser.parse_args()

    if MAP_PATH.exists() and not args.force:
        existing = load_document_map()
        print(f"{MAP_PATH.name} already covers {len(existing)} documents (--force to rebuild)")
        return 0

    if not OBLIQA_DOCS.exists():
        print("ObliQA corpus missing -- run: uv run python -m src.ingestion.download")
        return 1

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers=headers) as client:
        texts = fetch_standardized_texts(client)

    payload = build_map(OBLIQA_DOCS, texts)
    MAP_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Mapped {len(payload['documents'])} documents -> {MAP_PATH.relative_to(DATA_DIR.parent)}")
    for doc_id, entry in sorted(payload["documents"].items(), key=lambda kv: int(kv[0]))[:5]:
        print(f"  {doc_id:>3}  {entry['title']:<28} {entry['passages']:>5} passages")
    print("  ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
