"""Acquire the two Phase 1 (§3.2) corpora and record their provenance.

Two corpora, two very different roles:

A. SAML-D (Kaggle) -- the transaction feed being *audited*. ~996 MB / 9,504,852 rows.
B. A hybrid regulatory corpus -- the *law* audited against, which §3.4 chunks into ChromaDB:
   ObliQA (40 ADGM rulebooks), FINRA Rule 3310/3110 + Regulatory Notice 19-18, FinCEN alerts.

Neither corpus is committed -- ``data/raw`` is gitignored. What *is* committed is
``data/MANIFEST.json``: source URL, sha256, size, retrieval date and licence for every
artifact, so a gigabyte of data stays reproducible from a few KB of tracked JSON.

Usage::

    uv run python -m src.ingestion.download           # fetch anything missing or changed
    uv run python -m src.ingestion.download --check   # verify against the manifest, fetch nothing
    uv run python -m src.ingestion.download --force   # re-fetch everything

No credentials are required: SAML-D is a public Kaggle dataset and ``kagglehub`` falls back to
an unauthenticated client, while the regulatory corpus is fetched over plain HTTPS. A token at
``~/.kaggle/kaggle.json`` is only a fallback for when Kaggle refuses the anonymous download.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

USER_AGENT = "finguard-orchestrator/0.1 (dataset acquisition)"
TIMEOUT = httpx.Timeout(30.0, read=180.0)

# --- A. Transaction ledger feed ------------------------------------------------------

SAML_D_SLUG = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
SAML_D_CSV = RAW_DIR / "saml_d" / "SAML-D.csv"
SAML_D_LICENCE = "CC BY-NC-SA 4.0 (non-commercial; attribution required)"
SAML_D_CITATION = (
    "B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan and G. Aksu, "
    '"Enhancing Anti-Money Laundering: Development of a Synthetic Transaction '
    'Monitoring Dataset," 2023 IEEE International Conference on e-Business '
    "Engineering (ICEBE), Sydney, Australia, 2023, pp. 47-54, "
    "doi:10.1109/ICEBE59045.2023.00028"
)

# --- B. Regulatory knowledge base ----------------------------------------------------

OBLIQA_ZIP_URL = (
    "https://raw.githubusercontent.com/RegNLP/ObliQADataset/main/"
    "StructuredRegulatoryDocuments.zip"
)
OBLIQA_DIR = RAW_DIR / "regulations" / "obliqa"
OBLIQA_ZIP = OBLIQA_DIR / "StructuredRegulatoryDocuments.zip"
OBLIQA_DOCS = OBLIQA_DIR / "StructuredRegulatoryDocuments"
OBLIQA_EXPECTED_DOCS = 40


@dataclass(frozen=True)
class RemoteFile:
    """A regulatory document fetched verbatim from its publisher."""

    url: str
    dest: Path
    licence: str
    note: str


REGULATORY_PDFS: tuple[RemoteFile, ...] = (
    RemoteFile(
        url="https://www.finra.org/sites/default/files/2019-05/Regulatory-Notice-19-18.pdf",
        dest=RAW_DIR / "regulations" / "finra" / "regulatory-notice-19-18.pdf",
        licence="FINRA public guidance",
        note="Regulatory Notice 19-18 -- 104 money laundering red flags for broker-dealers",
    ),
    RemoteFile(
        url=(
            "https://www.fincen.gov/system/files/shared/"
            "FinCEN%20Alert%20Real%20Estate%20FINAL%20508_1-25-23%20FINAL%20FINAL.pdf"
        ),
        dest=RAW_DIR / "regulations" / "fincen" / "fin-2023-alert002-commercial-real-estate.pdf",
        licence="US Government work (public domain)",
        note="FIN-2023-Alert002 -- sanctions evasion via commercial real estate; shell company red flags",
    ),
    RemoteFile(
        url=(
            "https://www.fincen.gov/system/files/2022-03/"
            "FinCEN%20Alert%20Russian%20Elites%20High%20Value%20Assets_508%20FINAL.pdf"
        ),
        dest=RAW_DIR / "regulations" / "fincen" / "fin-2022-alert002-russian-elites.pdf",
        licence="US Government work (public domain)",
        note="FIN-2022-Alert002 -- red flags for high-value assets held through shell companies and trusts",
    ),
    RemoteFile(
        url="https://www.fincen.gov/system/files/2022-06/FinCEN%20and%20Bis%20Joint%20Alert%20FINAL.pdf",
        dest=RAW_DIR / "regulations" / "fincen" / "fin-2022-alert003-fincen-bis-joint.pdf",
        licence="US Government work (public domain)",
        note="FIN-2022-Alert003 -- export control evasion; transshipment and illicit corridors",
    ),
)

# FINRA serves its rulebook as HTML only -- there is no official PDF of the rule text.
# Each entry carries a marker phrase from the operative language: if the scrape stops
# returning it, the page structure changed and we fail loudly rather than storing chrome.
FINRA_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "3310",
        "Anti-Money Laundering Compliance Program",
        "written anti-money laundering program",
    ),
    (
        "3110",
        "Supervision",
        "system to supervise the activities",
    ),
)
FINRA_RULE_URL = "https://www.finra.org/rules-guidance/rulebooks/finra-rules/{number}"
FINRA_DIR = RAW_DIR / "regulations" / "finra"


# --- manifest ------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"schema": 1, "artifacts": {}, "citations": {}}


def save_manifest(manifest: dict) -> None:
    manifest["artifacts"] = dict(sorted(manifest["artifacts"].items()))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def rel(path: Path) -> str:
    return path.relative_to(DATA_DIR).as_posix()


def record(manifest: dict, path: Path, *, url: str, licence: str, note: str, **extra) -> None:
    manifest["artifacts"][rel(path)] = {
        "url": url,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "retrieved": date.today().isoformat(),
        "licence": licence,
        "note": note,
        **extra,
    }


def is_current(manifest: dict, path: Path) -> bool:
    """True when the file on disk matches what the manifest recorded."""
    entry = manifest["artifacts"].get(rel(path))
    if entry is None or not path.exists():
        return False
    if path.stat().st_size != entry["bytes"]:
        return False
    return sha256_file(path) == entry["sha256"]


# --- fetching ------------------------------------------------------------------------


def download(client: httpx.Client, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 16):
                handle.write(chunk)
    tmp.replace(dest)


def html_to_text(fragment: str) -> str:
    """Flatten an HTML fragment, preserving paragraph breaks for the §3.3 chunker."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", fragment)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def extract_finra_rule(page: str, marker: str) -> str:
    """Pull the rule body out of the FINRA page, discarding site chrome.

    The operative text lives in the ``block-body`` region; everything after the
    tab-content block is related-notices navigation.
    """
    anchor = page.find('id="block-body"')
    if anchor == -1:
        raise RuntimeError("FINRA page layout changed: no block-body region")
    # Both ends must land on a tag boundary: the anchor sits *inside* an opening tag, and
    # the tab-content marker sits inside the next one. Slicing at either would leave a
    # half-tag that survives tag-stripping as literal attribute text.
    start = page.index(">", anchor) + 1
    marker_at = page.find("field--name-field-tab-content", start)
    end = page.rindex("<", start, marker_at) if marker_at != -1 else len(page)

    text = html_to_text(page[start:end])
    if marker not in text:
        raise RuntimeError(f"FINRA rule text missing expected phrase: {marker!r}")
    if "block-plugin-id" in text or "field--name" in text:
        raise RuntimeError("FINRA rule text still contains page chrome")
    return text


def fetch_finra_rule(client: httpx.Client, number: str, title: str, marker: str) -> Path:
    url = FINRA_RULE_URL.format(number=number)
    response = client.get(url)
    response.raise_for_status()
    body = extract_finra_rule(response.text, marker)

    dest = FINRA_DIR / f"finra-rule-{number}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# FINRA Rule {number}. {title}\n"
        f"# Source: {url}\n"
        f"# Retrieved: {date.today().isoformat()}\n"
        "# FINRA publishes its rulebook as HTML only; this text was extracted from that page.\n"
        "# Reproduce with: uv run python -m src.ingestion.download\n\n"
    )
    dest.write_text(header + body + "\n")
    return dest


def extract_obliqa(zip_path: Path) -> int:
    """Extract the 40 structured ADGM documents, skipping macOS resource forks.

    The archive carries a ``__MACOSX/`` tree whose entries also end in ``.json``; a naive
    glob over the extracted output returns 80 files, half of which are not JSON at all.
    """
    OBLIQA_DOCS.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.startswith("__MACOSX") or not name.endswith(".json"):
                continue
            target = OBLIQA_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted += 1
    if extracted != OBLIQA_EXPECTED_DOCS:
        raise RuntimeError(
            f"expected {OBLIQA_EXPECTED_DOCS} ObliQA documents, extracted {extracted}"
        )
    return extracted


def fetch_saml_d(manifest: dict, *, force: bool) -> bool:
    """Pull SAML-D via kagglehub.

    The dataset is public, so this succeeds unauthenticated. Returns False with guidance if
    Kaggle refuses -- the usual cause is rate limiting or a licence needing acceptance, both
    of which a personal API token resolves.
    """
    if not force and is_current(manifest, SAML_D_CSV):
        print(f"  ok        {rel(SAML_D_CSV)} (unchanged)")
        return True

    import kagglehub

    try:
        cached = Path(kagglehub.dataset_download(SAML_D_SLUG))
    except Exception as error:  # noqa: BLE001 - surfaced verbatim to the user
        print(f"  SKIPPED   SAML-D: {type(error).__name__}: {error}")
        print(
            "            The anonymous download failed. Create a Kaggle API token\n"
            "            (kaggle.com -> Settings -> API -> Create New Token), save it as\n"
            "            ~/.kaggle/kaggle.json, then re-run. The regulatory corpus below\n"
            "            is unaffected and will still be acquired."
        )
        return False

    source = next(cached.rglob("SAML-D.csv"))
    SAML_D_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SAML_D_CSV.exists() or SAML_D_CSV.is_symlink():
        SAML_D_CSV.unlink()
    # kagglehub keeps its own versioned cache; symlink rather than duplicate ~1 GB.
    SAML_D_CSV.symlink_to(source)

    record(
        manifest,
        SAML_D_CSV,
        url=f"https://www.kaggle.com/datasets/{SAML_D_SLUG}",
        licence=SAML_D_LICENCE,
        note="SAML-D: 9,504,852 transactions, 12 features, 28 typologies (11 normal / 17 suspicious)",
        kaggle_slug=SAML_D_SLUG,
        kagglehub_cache=str(source),
    )
    manifest["citations"]["saml_d"] = SAML_D_CITATION
    print(f"  fetched   {rel(SAML_D_CSV)} -> {source}")
    return True


# --- entry points --------------------------------------------------------------------


def acquire(*, force: bool = False) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    print("A. Transaction ledger feed (Kaggle)")
    saml_ok = fetch_saml_d(manifest, force=force)

    print("\nB. Regulatory knowledge base")
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers=headers) as client:
        if force or not is_current(manifest, OBLIQA_ZIP):
            download(client, OBLIQA_ZIP_URL, OBLIQA_ZIP)
            count = extract_obliqa(OBLIQA_ZIP)
            record(
                manifest,
                OBLIQA_ZIP,
                url=OBLIQA_ZIP_URL,
                licence="RegNLP / ObliQA (see upstream repository)",
                note=(
                    "ADGM regulatory corpus: 40 documents, 13,732 passages, ~876k words. "
                    "DocumentID 1 is the ADGM AML Rulebook."
                ),
                extracted_documents=count,
            )
            print(f"  fetched   {rel(OBLIQA_ZIP)} ({count} documents extracted)")
        else:
            print(f"  ok        {rel(OBLIQA_ZIP)} (unchanged)")

        for remote in REGULATORY_PDFS:
            if force or not is_current(manifest, remote.dest):
                download(client, remote.url, remote.dest)
                record(
                    manifest,
                    remote.dest,
                    url=remote.url,
                    licence=remote.licence,
                    note=remote.note,
                )
                print(f"  fetched   {rel(remote.dest)}")
            else:
                print(f"  ok        {rel(remote.dest)} (unchanged)")

        for number, title, marker in FINRA_RULES:
            dest = FINRA_DIR / f"finra-rule-{number}.txt"
            if force or not dest.exists():
                dest = fetch_finra_rule(client, number, title, marker)
                record(
                    manifest,
                    dest,
                    url=FINRA_RULE_URL.format(number=number),
                    licence="FINRA rulebook (HTML source; no official PDF published)",
                    note=f"FINRA Rule {number}. {title}",
                    extraction="html-scrape",
                )
                print(f"  fetched   {rel(dest)}")
            else:
                print(f"  ok        {rel(dest)} (present)")

    # Renaming an artifact would otherwise leave its old key behind, and --check would then
    # report a file that is no longer meant to exist as MISSING forever.
    expected = {
        rel(SAML_D_CSV),
        rel(OBLIQA_ZIP),
        *(rel(remote.dest) for remote in REGULATORY_PDFS),
        *(rel(FINRA_DIR / f"finra-rule-{number}.txt") for number, _, _ in FINRA_RULES),
    }
    for stale in set(manifest["artifacts"]) - expected:
        del manifest["artifacts"][stale]
        print(f"  dropped   {stale} (no longer an expected artifact)")

    save_manifest(manifest)
    print(f"\nManifest written to {rel(MANIFEST_PATH)}")
    return 0 if saml_ok else 1


def check() -> int:
    manifest = load_manifest()
    if not manifest["artifacts"]:
        print("Manifest is empty -- run without --check to acquire the datasets.")
        return 1

    failures = 0
    for relpath, entry in manifest["artifacts"].items():
        path = DATA_DIR / relpath
        if not path.exists():
            print(f"  MISSING   {relpath}")
            failures += 1
        elif sha256_file(path) != entry["sha256"]:
            print(f"  CHANGED   {relpath}")
            failures += 1
        else:
            print(f"  ok        {relpath}")

    print(f"\n{len(manifest['artifacts']) - failures}/{len(manifest['artifacts'])} artifacts verified")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify the manifest, fetch nothing")
    group.add_argument("--force", action="store_true", help="re-fetch everything")
    args = parser.parse_args()
    return check() if args.check else acquire(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
