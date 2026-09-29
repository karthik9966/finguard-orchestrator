"""Acquire the corpora and record their provenance.

Three corpora, three roles, and the roles are what decide where each one may be used:

A. **SAML-D** (Kaggle) -- the transaction feed being *audited*. ~996 MB / 9,504,852 rows.
B. **The citable US knowledge base** -- the law audited against, and the only material a finding
   may cite. Two tiers, retrieved two different ways:
     * Tier 1, binding obligations, fetched by curated id: 31 USC 5324 (the structuring
       prohibition) and 31 CFR 1010.311, 1010.410, 1020.210, 1020.320.
     * Tier 2, illustrative indicators, fetched by semantic search: FFIEC Appendix F's red-flag
       list plus nine manual sections, FINRA Regulatory Notice 19-18's 104 red flags, three
       FinCEN alerts, and FINRA Rules 3310/3110.
C. **ObliQA** (40 ADGM rulebooks + 2,786 labelled questions) -- retained *only* as the retrieval
   benchmark. ADGM law is out of jurisdiction for this version (PRD §2), so it never enters the
   citable collection; what it still provides is the one labelled ground truth for judging
   whether a chunking or reranking change actually helps.

Every artifact therefore declares a ``role`` (transactions / obligation / indicator / benchmark)
and, where it is citable, a ``tier``, an ``authority`` and a ``jurisdiction``. That metadata is
not bookkeeping: ``role`` decides which collection an artifact may enter, ``authority`` decides
whether a chunk reaches the Tier-2 indicator pool, and ``source_id`` is the left half of the
``(source_id, section_ref)`` pairs that ``config.yaml``'s ``pattern_to_obligations`` is curated
as. ``classification_problems()`` refuses to let an artifact through without them, because every
one of those omissions fails late and quietly.

Neither B nor C is committed -- ``data/raw`` is gitignored. What *is* committed is
``data/MANIFEST.json``: source URL, sha256, size, retrieval date, licence and classification for
every artifact, so a gigabyte of data stays reproducible from a few KB of tracked JSON.

Usage::

    uv run python -m src.ingestion.download           # fetch anything missing or changed
    uv run python -m src.ingestion.download --check   # verify hashes + classification, fetch nothing
    uv run python -m src.ingestion.download --force   # re-fetch everything

No credentials are required. SAML-D is a public Kaggle dataset and ``kagglehub`` falls back to an
unauthenticated client; a token at ``~/.kaggle/kaggle.json`` is only a fallback for when Kaggle
refuses the anonymous download. Two publishers need more than plain HTTPS, and both are
documented where they are declared: govinfo answers a missing granule with an error page at HTTP
200, and FFIEC sits behind a WAF that refuses anything not shaped like a browser.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
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

# ObliQA is a *retrieval* dataset, not just a corpus: the same repository ships questions
# already labelled with the passages that answer them. That is the only ground truth we have
# for judging §3.3's chunking -- without it, "is this chunked well?" is unanswerable.
OBLIQA_QA_URL = "https://raw.githubusercontent.com/RegNLP/ObliQADataset/main/ObliQA_test.json"
OBLIQA_QA = OBLIQA_DIR / "ObliQA_test.json"
OBLIQA_EXPECTED_QUESTIONS = 2786


@dataclass(frozen=True)
class RemoteFile:
    """A regulatory document fetched verbatim from its publisher."""

    url: str
    dest: Path
    licence: str
    note: str
    # Stable identifier carried into RuleChunk.source_id and into the left half of
    # config.yaml's pattern_to_obligations pairs. It must not change once curated.
    source_id: str = ""


REGULATORY_PDFS: tuple[RemoteFile, ...] = (
    RemoteFile(
        url="https://www.finra.org/sites/default/files/2019-05/Regulatory-Notice-19-18.pdf",
        dest=RAW_DIR / "regulations" / "finra" / "regulatory-notice-19-18.pdf",
        licence="FINRA public guidance",
        note="Regulatory Notice 19-18 -- 104 money laundering red flags for broker-dealers",
        source_id="finra-rn-19-18",
    ),
    RemoteFile(
        url=(
            "https://www.fincen.gov/system/files/shared/"
            "FinCEN%20Alert%20Real%20Estate%20FINAL%20508_1-25-23%20FINAL%20FINAL.pdf"
        ),
        dest=RAW_DIR / "regulations" / "fincen" / "fin-2023-alert002-commercial-real-estate.pdf",
        licence="US Government work (public domain)",
        note="FIN-2023-Alert002 -- sanctions evasion via commercial real estate; shell company red flags",
        source_id="fincen-2023-alert002",
    ),
    RemoteFile(
        url=(
            "https://www.fincen.gov/system/files/2022-03/"
            "FinCEN%20Alert%20Russian%20Elites%20High%20Value%20Assets_508%20FINAL.pdf"
        ),
        dest=RAW_DIR / "regulations" / "fincen" / "fin-2022-alert002-russian-elites.pdf",
        licence="US Government work (public domain)",
        note="FIN-2022-Alert002 -- red flags for high-value assets held through shell companies and trusts",
        source_id="fincen-2022-alert002",
    ),
    RemoteFile(
        url="https://www.fincen.gov/system/files/2022-06/FinCEN%20and%20Bis%20Joint%20Alert%20FINAL.pdf",
        dest=RAW_DIR / "regulations" / "fincen" / "fin-2022-alert003-fincen-bis-joint.pdf",
        licence="US Government work (public domain)",
        note="FIN-2022-Alert003 -- export control evasion; transshipment and illicit corridors",
        source_id="fincen-2022-alert003",
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


# --- B1. Tier 1: the binding US obligations ------------------------------------------
#
# None of these were in the manifest before the design migration: the pre-migration corpus was
# ADGM law plus US *guidance*, with no US statute or regulation in it at all. They are what a
# finding now rests on -- PATTERN_TO_OBLIGATIONS maps each detected typology to sections in this
# set, and a report that cannot cite one of them is not a report.

# 31 USC 5324 comes from govinfo's static US Code editions rather than uscode.house.gov, which
# serves the same text wrapped in 158 KB of site chrome.
#
# The edition is pinned, and that is not caution for its own sake: **govinfo answers a missing
# granule with its "Page Not Found" page at HTTP 200**. Asking for the 2025 edition returns 44 KB
# of Drupal markup that `raise_for_status()` is perfectly happy with, and a naive fetch would
# store it as the structuring statute. The marker phrase below is the only thing between that and
# a knowledge base whose Tier-1 obligation is an error page.
USC_EDITION = 2024
USC_URL = (
    "https://www.govinfo.gov/content/pkg/USCODE-{edition}-title31/html/"
    "USCODE-{edition}-title31-subtitleIV-chap53-subchapII-sec{section}.htm"
)


@dataclass(frozen=True)
class Statute:
    """A US Code section, fetched as HTML because that is the only form govinfo serves it in."""

    section: str
    source_id: str
    title: str
    marker: str
    note: str


US_STATUTES: tuple[Statute, ...] = (
    Statute(
        section="5324",
        source_id="31usc5324",
        title="Structuring transactions to evade reporting requirement prohibited",
        marker="structure or assist in structuring",
        note=(
            "31 USC 5324 -- the structuring prohibition itself. The binding obligation behind "
            "the structuring and smurfing typologies."
        ),
    ),
)

# eCFR publishes machine-readable XML per section, which is why these are stored as XML rather
# than flattened text: DIV8/HEAD/P carries the section and paragraph structure that Phase 1b's
# tier-aware `split_by_section` needs to emit a real `section_ref`. Flattening here would throw
# away the very structure the chunker is being written to use.
ECFR_TITLE = 31
ECFR_TITLES_URL = "https://www.ecfr.gov/api/versioner/v1/titles.json"
ECFR_FULL_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
ECFR_VERSIONS_URL = "https://www.ecfr.gov/api/versioner/v1/versions/title-{title}.json"
CFR_DIR = RAW_DIR / "regulations" / "cfr"


@dataclass(frozen=True)
class CFRSection:
    """One section of 31 CFR, fetched as eCFR XML."""

    section: str
    source_id: str
    title: str
    marker: str
    note: str

    @property
    def part(self) -> str:
        return self.section.split(".")[0]


CFR_SECTIONS: tuple[CFRSection, ...] = (
    CFRSection(
        section="1010.311",
        source_id="31cfr1010.311",
        title="Filing obligations for reports of transactions in currency",
        marker="more than $10,000",
        note="31 CFR 1010.311 -- the CTR filing obligation; the $10,000 monitored threshold.",
    ),
    # Not in the LLD's list, added deliberately: config.yaml's second structuring threshold is
    # $3,000, and this is the section that makes $3,000 a threshold at all. Without it the
    # $3,000 band would be a number the system enforces with no obligation behind it, and a
    # finding on that band could not cite anything.
    CFRSection(
        section="1010.410",
        source_id="31cfr1010.410",
        title="Records to be made and retained by financial institutions",
        marker="$3,000 or more",
        note=(
            "31 CFR 1010.410 -- funds-transfer recordkeeping and the travel rule; the source of "
            "the $3,000 monitored threshold."
        ),
    ),
    CFRSection(
        section="1020.210",
        source_id="31cfr1020.210",
        title="Anti-money laundering program requirements for banks",
        marker="anti-money laundering program",
        note="31 CFR 1020.210 -- the duty to maintain an AML program at all.",
    ),
    CFRSection(
        section="1020.320",
        source_id="31cfr1020.320",
        title="Reports by banks of suspicious transactions",
        marker="suspicious transaction",
        note=(
            "31 CFR 1020.320 -- the SAR filing obligation. Every in-scope typology maps here, "
            "because reporting is the duty each of them triggers."
        ),
    ),
)


# --- B2. Tier 2: FFIEC examiner guidance ---------------------------------------------
#
# Two things worth knowing before touching this source.
#
# **It sits behind a WAF that refuses a plain client.** A request carrying only a User-Agent gets
# 403 and a 244 KB challenge page -- with HTTP 403, so `raise_for_status()` does catch it, but the
# fix is not a retry. It needs a full browser header set (Accept, Accept-Language, Sec-Fetch-*)
# and a warm-up request to pick up the challenge cookies. This is the one source in the corpus
# where the request has to look like a browser to succeed.
#
# **The appendix files are numbered, and the numbers are offset from the letters.** 05.pdf is
# Appendix D, 06.pdf is Appendix E, and Appendix F -- the red-flag list this system actually
# wants -- is 07.pdf. Nothing on the page says so; the anchors carry no text. Guessing "Appendix
# F must be 06.pdf" silently yields Appendix E, which is a two-page list of international
# organizations. Hence the page-one marker assertion on every one of these.
FFIEC_MANUAL_URL = "https://bsaaml.ffiec.gov/manual"
FFIEC_DOC_URL = "https://bsaaml.ffiec.gov/docs/manual/{path}"
FFIEC_DIR = RAW_DIR / "regulations" / "ffiec"

FFIEC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}


@dataclass(frozen=True)
class FFIECDoc:
    """One section of the FFIEC BSA/AML Examination Manual, as published PDF."""

    path: str
    source_id: str
    title: str
    marker: str
    note: str

    @property
    def dest(self) -> Path:
        return FFIEC_DIR / f"{self.source_id}.pdf"


# There is no whole-manual PDF -- both documented single-file URLs return 200 with non-PDF
# content -- so the manual has to be taken section by section. These nine are chosen against
# PRD §2's scope rather than for completeness: each one is either the examiner guidance for an
# obligation this system cites, or the product-risk discussion for a shape it detects. The
# omissions are as deliberate as the inclusions -- customer identification, beneficial ownership,
# correspondent and private banking, trade finance and PEPs are all explicitly out of scope, and
# indexing them would add 70 PDFs of distractors to buy nothing.
FFIEC_DOCS: tuple[FFIECDoc, ...] = (
    FFIECDoc(
        path="10_Appendices/07.pdf",
        source_id="ffiec-appendix-f",
        title='Appendix F: Money Laundering and Terrorist Financing "Red Flags"',
        marker="Appendix F",
        note=(
            "FFIEC BSA/AML Examination Manual Appendix F -- the red-flag list. With FINRA "
            "Regulatory Notice 19-18 this is the Tier-2 indicator corpus."
        ),
    ),
    FFIECDoc(
        path="06_AssessingComplianceWithBSARegulatoryRequirements/04.pdf",
        source_id="ffiec-manual-sar",
        title="Suspicious Activity Reporting — Overview",
        marker="Suspicious Activity Reporting",
        note="Examiner guidance on the SAR obligation of 31 CFR 1020.320.",
    ),
    FFIECDoc(
        path="06_AssessingComplianceWithBSARegulatoryRequirements/05.pdf",
        source_id="ffiec-manual-ctr",
        title="Currency Transaction Reporting",
        marker="Currency Transaction Reporting",
        note="Examiner guidance on the CTR obligation of 31 CFR 1010.311.",
    ),
    FFIECDoc(
        path="06_AssessingComplianceWithBSARegulatoryRequirements/08.pdf",
        source_id="ffiec-manual-monetary-instruments",
        title="Purchase and Sale of Monetary Instruments Recordkeeping",
        marker="Monetary Instruments",
        note="The $3,000 monetary-instrument recordkeeping threshold in examiner language.",
    ),
    FFIECDoc(
        path="06_AssessingComplianceWithBSARegulatoryRequirements/09.pdf",
        source_id="ffiec-manual-funds-transfer-recordkeeping",
        title="Funds Transfers Record Keeping — Overview",
        marker="Funds Transfers",
        note="Examiner guidance on 31 CFR 1010.410's funds-transfer recordkeeping and travel rule.",
    ),
    FFIECDoc(
        path="09_RisksAssociatedWithMoneyLaunderingAndTerroristFinancing/07.pdf",
        source_id="ffiec-risk-funds-transfers",
        title="Funds Transfers — Overview",
        marker="Funds Transfers",
        note="Product-risk discussion for wire transfers, the instrument most of the batch is.",
    ),
    FFIECDoc(
        path="09_RisksAssociatedWithMoneyLaunderingAndTerroristFinancing/08.pdf",
        source_id="ffiec-risk-ach",
        title="Automated Clearing House Transactions — Overview",
        marker="Automated Clearing House",
        note="Product-risk discussion for ACH, one of SAML-D's payment types.",
    ),
    FFIECDoc(
        path="09_RisksAssociatedWithMoneyLaunderingAndTerroristFinancing/15.pdf",
        source_id="ffiec-risk-concentration-accounts",
        title="Concentration Accounts — Overview",
        marker="Concentration Accounts",
        note=(
            "The collector-account shape in examiner language -- the closest published "
            "description of what the fan-in detector finds."
        ),
    ),
    FFIECDoc(
        path="09_RisksAssociatedWithMoneyLaunderingAndTerroristFinancing/26.pdf",
        source_id="ffiec-risk-cash-intensive",
        title="Cash-Intensive Businesses — Overview",
        marker="Cash-Intensive Businesses",
        note="Product-risk discussion behind sub-threshold cash structuring.",
    ),
)


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


# Every artifact declares what it is for. `role` decides which collection it may enter, and
# for regulation it also decides how it is retrieved: an `obligation` is fetched by curated id
# (Tier 1), an `indicator` by semantic search filtered on authority (Tier 2).
ROLES = ("transactions", "obligation", "indicator", "benchmark")
TIERS = ("statute", "regulation", "guidance")
AUTHORITIES = ("binding", "illustrative")


def classification_table() -> dict[str, dict]:
    """What every artifact *is*, derived from the source declarations above.

    One table rather than a keyword argument at each fetch site. The fetch sites are the obvious
    place to put this, and that is exactly why it is wrong here: an artifact already on disk is
    never re-fetched, so classification attached to the fetch would never reach the nine artifacts
    acquired before this migration. Deriving it from the declarations means it applies to what is
    already downloaded and to what is fetched next, and there is one place to correct a mistake.
    """
    table: dict[str, dict] = {
        rel(SAML_D_CSV): {"role": "transactions", "source_id": "saml-d"},
        # ADGM law. Benchmark only: out of jurisdiction for this version, so it never enters the
        # citable collection and carries no tier or authority at all.
        rel(OBLIQA_ZIP): {"role": "benchmark", "source_id": "obliqa", "jurisdiction": "ADGM"},
        rel(OBLIQA_QA): {"role": "benchmark", "source_id": "obliqa-qa", "jurisdiction": "ADGM"},
    }
    for remote in REGULATORY_PDFS:
        table[rel(remote.dest)] = {
            "role": "indicator",
            "source_id": remote.source_id,
            "tier": "guidance",
            "authority": "illustrative",
            "jurisdiction": "US",
        }
    for number, _title, _marker in FINRA_RULES:
        table[rel(FINRA_DIR / f"finra-rule-{number}.txt")] = {
            "role": "indicator",
            "source_id": f"finra-{number}",
            # Genuinely binding -- on FINRA members. The audited institution here is a bank, so
            # this is never mapped in pattern_to_obligations and can never ground a finding. It
            # stays indexed as a near-miss distractor, which is what makes Context Precision
            # measurable at all. Recording it as illustrative would have been simpler and false.
            "tier": "regulation",
            "authority": "binding",
            "applies_to": "broker_dealer",
            "jurisdiction": "US",
        }
    for statute in US_STATUTES:
        table[rel(RAW_DIR / "regulations" / "usc" / f"{statute.source_id}.txt")] = {
            "role": "obligation",
            "source_id": statute.source_id,
            "tier": "statute",
            "authority": "binding",
            "jurisdiction": "US",
            "version": f"USCODE-{USC_EDITION}",
        }
    for section in CFR_SECTIONS:
        table[rel(CFR_DIR / f"{section.source_id}.xml")] = {
            "role": "obligation",
            "source_id": section.source_id,
            "tier": "regulation",
            "authority": "binding",
            "jurisdiction": "US",
        }
    for doc in FFIEC_DOCS:
        table[rel(doc.dest)] = {
            "role": "indicator",
            "source_id": doc.source_id,
            "tier": "guidance",
            "authority": "illustrative",
            "jurisdiction": "US",
        }
    return table


def apply_classification(manifest: dict) -> int:
    """Stamp every manifest entry with what it is. Returns how many changed."""
    table = classification_table()
    changed = 0
    for relpath, fields in table.items():
        entry = manifest["artifacts"].get(relpath)
        if entry is None:
            continue
        # Fetch-time fields (version, effective_date) are more specific than the table's, so the
        # table never overwrites a value that is already there.
        additions = {key: value for key, value in fields.items() if entry.get(key) != value
                     and key not in entry}
        if additions:
            entry.update(additions)
            changed += 1
    return changed


def classification_problems(manifest: dict) -> list[str]:
    """Metadata the knowledge base cannot be built without.

    Checked here rather than discovered in Phase 1b, because every one of these omissions fails
    *late* and quietly: a missing `source_id` breaks the (source_id, section_ref) pairs that
    pattern_to_obligations is curated as, and a missing `authority` silently drops a chunk out of
    the Tier-2 indicator pool -- which looks like a retrieval quality problem, not a metadata bug.
    """
    problems: list[str] = []
    for relpath, entry in manifest.get("artifacts", {}).items():
        role = entry.get("role")
        if role is None:
            problems.append(f"{relpath}: no role")
            continue
        if role not in ROLES:
            problems.append(f"{relpath}: unknown role {role!r}")
        if not entry.get("source_id"):
            problems.append(f"{relpath}: no source_id")
        if role in ("obligation", "indicator"):
            if entry.get("tier") not in TIERS:
                problems.append(f"{relpath}: tier {entry.get('tier')!r} not one of {TIERS}")
            if entry.get("authority") not in AUTHORITIES:
                problems.append(
                    f"{relpath}: authority {entry.get('authority')!r} not one of {AUTHORITIES}"
                )
            # The one rule that cannot be relaxed: a citable chunk must be US law. ADGM material
            # stays in the benchmark collection, and this is the check that keeps it there.
            if entry.get("jurisdiction") != "US":
                problems.append(
                    f"{relpath}: jurisdiction {entry.get('jurisdiction')!r} -- only US material "
                    "is citable (PRD §2)"
                )
    return problems


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


def fetch_statute(client: httpx.Client, statute: Statute) -> Path:
    """Fetch one US Code section from govinfo and store it as flattened text.

    The marker assertion is load-bearing rather than defensive: govinfo serves a missing granule
    as its "Page Not Found" page at HTTP 200, so `raise_for_status()` cannot tell a statute from
    an error page. Only the text can.
    """
    url = USC_URL.format(edition=USC_EDITION, section=statute.section)
    response = client.get(url)
    response.raise_for_status()

    body = html_to_text(response.text)
    if statute.marker not in body:
        raise RuntimeError(
            f"31 USC {statute.section}: expected phrase {statute.marker!r} absent from "
            f"{len(body):,} characters -- govinfo served an error page at HTTP 200, or the "
            f"{USC_EDITION} edition no longer carries this section"
        )

    dest = RAW_DIR / "regulations" / "usc" / f"{statute.source_id}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# 31 U.S.C. § {statute.section}. {statute.title}\n"
        f"# Source: {url}\n"
        f"# Edition: United States Code, {USC_EDITION} Edition\n"
        f"# Retrieved: {date.today().isoformat()}\n"
        "# govinfo publishes the US Code as HTML; this text was extracted from that page.\n"
        "# Reproduce with: uv run python -m src.ingestion.download\n\n"
    )
    dest.write_text(header + body + "\n")
    return dest


def ecfr_issue_date(client: httpx.Client) -> str:
    """The latest issue date eCFR will serve title 31 for.

    Asked rather than assumed: the API rejects a date it has no issue for, and "today" is wrong
    on most days.
    """
    response = client.get(ECFR_TITLES_URL)
    response.raise_for_status()
    for title in response.json()["titles"]:
        if title["number"] == ECFR_TITLE:
            return title["latest_issue_date"]
    raise RuntimeError(f"eCFR does not list title {ECFR_TITLE}")


def ecfr_amendment_dates(client: httpx.Client) -> dict[str, str]:
    """Each section's last substantive amendment date, keyed by section number.

    This is the `effective_date` a citation carries. It matters for a reason specific to this
    system: a report defended in an audit years later has to show the rule *as it stood*, and
    "when we downloaded it" is not that date. 1010.311 was last amended in 2016.
    """
    response = client.get(ECFR_VERSIONS_URL.format(title=ECFR_TITLE))
    response.raise_for_status()
    wanted = {section.section for section in CFR_SECTIONS}
    dates: dict[str, str] = {}
    for version in response.json().get("content_versions", []):
        identifier = version.get("identifier")
        if identifier in wanted and version.get("amendment_date"):
            # The feed is chronological, so the last entry seen is the most recent amendment.
            dates[identifier] = version["amendment_date"]
    return dates


def fetch_cfr_section(client: httpx.Client, section: CFRSection, issue_date: str) -> Path:
    """Fetch one 31 CFR section as eCFR XML, verbatim."""
    response = client.get(
        ECFR_FULL_URL.format(date=issue_date, title=ECFR_TITLE),
        params={"part": section.part, "section": section.section},
    )
    response.raise_for_status()
    xml = response.text

    # Flatten only to check: the stored artifact stays XML, because the chunker needs its
    # structure. A section that comes back without its own number is a wrong or empty answer.
    flat = re.sub(r"<[^>]+>", " ", xml)
    if section.section not in xml:
        raise RuntimeError(f"eCFR returned XML that does not mention {section.section}")
    if section.marker not in flat:
        raise RuntimeError(
            f"31 CFR {section.section}: expected phrase {section.marker!r} absent -- the section "
            "was amended, or eCFR returned a different one"
        )

    dest = CFR_DIR / f"{section.source_id}.xml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(xml)
    return dest


def ffiec_session() -> httpx.Client:
    """A client that FFIEC's WAF will answer.

    Two things are required and neither is optional: the browser header set, and a warm-up
    request to the manual index so the challenge cookies are in the jar before any PDF is asked
    for. A plain client gets 403 and a 244 KB challenge page on every request.
    """
    client = httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers=FFIEC_HEADERS)
    warmup = client.get(FFIEC_MANUAL_URL)
    if warmup.status_code != 200:
        client.close()
        raise RuntimeError(
            f"FFIEC manual index returned {warmup.status_code} -- the WAF refused the warm-up, "
            "so no manual PDF can be fetched this run"
        )
    return client


def fetch_ffiec_doc(client: httpx.Client, doc: FFIECDoc) -> Path:
    """Fetch one FFIEC manual PDF, verifying it is the document we asked for.

    The page-one check is what catches the appendix numbering offset: Appendix F is 07.pdf, and
    06.pdf -- the intuitive guess -- is Appendix E. Both are valid PDFs of the right shape, so
    only the text distinguishes them.
    """
    url = FFIEC_DOC_URL.format(path=doc.path)
    response = client.get(url)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        raise RuntimeError(
            f"{doc.source_id}: {url} returned {len(content):,} bytes that are not a PDF "
            "(the WAF challenge page answers with HTML)"
        )

    first_page = ""
    try:
        from pypdf import PdfReader

        first_page = PdfReader(io.BytesIO(content)).pages[0].extract_text() or ""
    except Exception as error:  # a PDF we cannot read is a PDF we cannot verify
        raise RuntimeError(f"{doc.source_id}: downloaded PDF is unreadable: {error}") from error

    flattened = re.sub(r"\s+", " ", first_page)
    if doc.marker not in flattened:
        raise RuntimeError(
            f"{doc.source_id}: page 1 of {doc.path} does not mention {doc.marker!r} -- the "
            f"manual was renumbered. Page 1 begins: {flattened[:120]!r}"
        )

    doc.dest.parent.mkdir(parents=True, exist_ok=True)
    doc.dest.write_bytes(content)
    return doc.dest


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


def verify_obliqa_qa(qa_path: Path) -> tuple[int, int, int]:
    """Check the gold set against the documents on disk.

    Returns ``(questions, gold_pairs, unresolved)``. A question is only usable for scoring
    retrieval if its ``(DocumentID, PassageID)`` actually exists in the corpus we indexed --
    upstream QA sets routinely drift from the document release they were built against, so
    the dangling rate is measured rather than assumed.
    """
    records = json.loads(qa_path.read_text())
    if len(records) != OBLIQA_EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"expected {OBLIQA_EXPECTED_QUESTIONS} ObliQA questions, found {len(records)}"
        )

    known: set[tuple[int, str]] = set()
    for doc in OBLIQA_DOCS.glob("*.json"):
        for passage in json.loads(doc.read_text()):
            known.add((passage["DocumentID"], passage["PassageID"]))

    pairs = unresolved = 0
    for record_ in records:
        for passage in record_["Passages"]:
            pairs += 1
            if (passage["DocumentID"], passage["PassageID"]) not in known:
                unresolved += 1
    return len(records), pairs, unresolved


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
                    "DocumentID 1 is the ADGM AML Rulebook. Retained as the retrieval benchmark "
                    "only -- ADGM law is out of jurisdiction for this version (PRD §2) and never "
                    "enters the citable collection."
                ),
                extracted_documents=count,
            )
            print(f"  fetched   {rel(OBLIQA_ZIP)} ({count} documents extracted)")
        else:
            print(f"  ok        {rel(OBLIQA_ZIP)} (unchanged)")

        if force or not is_current(manifest, OBLIQA_QA):
            download(client, OBLIQA_QA_URL, OBLIQA_QA)
            questions, pairs, unresolved = verify_obliqa_qa(OBLIQA_QA)
            record(
                manifest,
                OBLIQA_QA,
                url=OBLIQA_QA_URL,
                licence="RegNLP / ObliQA (see upstream repository)",
                note=(
                    f"Retrieval gold set: {questions:,} questions labelled with "
                    f"{pairs:,} relevant passages. Ground truth for §3.3 chunking quality."
                ),
                questions=questions,
                gold_pairs=pairs,
                unresolved_pairs=unresolved,
            )
            print(f"  fetched   {rel(OBLIQA_QA)} ({questions:,} questions, {pairs:,} gold passages)")
            if unresolved:
                print(
                    f"            {unresolved:,}/{pairs:,} gold passages "
                    f"({unresolved / pairs:.1%}) do not resolve against the corpus on disk"
                )
        else:
            print(f"  ok        {rel(OBLIQA_QA)} (unchanged)")

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
            # is_current() rather than exists(): a file on disk that the manifest has no entry
            # for must be re-fetched and recorded, or --check can never see it. That is not
            # hypothetical -- a crashed run left 31usc5324.txt on disk and unrecorded, and the
            # next run skipped it on exists() and reported 22/22 verified with 23 artifacts.
            if force or not is_current(manifest, dest):
                dest = fetch_finra_rule(client, number, title, marker)
                record(
                    manifest,
                    dest,
                    url=FINRA_RULE_URL.format(number=number),
                    licence="FINRA rulebook (HTML source; no official PDF published)",
                    note=(
                        f"FINRA Rule {number}. {title}. Binding on FINRA members, not on banks -- "
                        "the audited institution here is a bank, so this is never mapped in "
                        "pattern_to_obligations. It stays indexed as a near-miss distractor, "
                        "which is what makes Context Precision measurable."
                    ),
                    extraction="html-scrape",
                )
                print(f"  fetched   {rel(dest)}")
            else:
                print(f"  ok        {rel(dest)} (present)")

        print("\n  Tier 1 -- binding US obligations")
        for statute in US_STATUTES:
            dest = RAW_DIR / "regulations" / "usc" / f"{statute.source_id}.txt"
            if force or not is_current(manifest, dest):
                dest = fetch_statute(client, statute)
                record(
                    manifest,
                    dest,
                    url=USC_URL.format(edition=USC_EDITION, section=statute.section),
                    licence="US Government work (public domain)",
                    note=statute.note,
                    extraction="html-scrape",
                )
                print(f"  fetched   {rel(dest)}")
            else:
                print(f"  ok        {rel(dest)} (present)")

        issue_date = ecfr_issue_date(client)
        amendments = ecfr_amendment_dates(client)
        for section in CFR_SECTIONS:
            dest = CFR_DIR / f"{section.source_id}.xml"
            if force or not is_current(manifest, dest):
                dest = fetch_cfr_section(client, section, issue_date)
                record(
                    manifest,
                    dest,
                    url=ECFR_FULL_URL.format(date=issue_date, title=ECFR_TITLE)
                    + f"?part={section.part}&section={section.section}",
                    licence="US Government work (public domain)",
                    note=section.note,
                    extraction="ecfr-xml",
                    version=f"eCFR {issue_date}",
                    effective_date=amendments.get(section.section),
                )
                print(f"  fetched   {rel(dest)} (amended {amendments.get(section.section, '?')})")
            else:
                print(f"  ok        {rel(dest)} (unchanged)")

    print("\n  Tier 2 -- FFIEC examiner guidance")
    try:
        ffiec = ffiec_session()
    except (RuntimeError, httpx.HTTPError) as error:
        # Not fatal, and deliberately so. Tier 1 is what a finding rests on and it is already
        # in hand; Tier 2 still has FINRA 19-18's 104 red flags and three FinCEN alerts. A
        # blocked examiner-guidance fetch degrades the indicator corpus, it does not stop the
        # system grounding anything.
        print(f"  SKIPPED   FFIEC unavailable: {error}")
        ffiec = None
    if ffiec is not None:
        # Not `with ffiec:` -- ffiec_session() has already issued the warm-up request,
        # which opens the client, and httpx refuses to enter an open client twice.
        try:
            for doc in FFIEC_DOCS:
                if not force and is_current(manifest, doc.dest):
                    print(f"  ok        {rel(doc.dest)} (present)")
                    continue
                try:
                    dest = fetch_ffiec_doc(ffiec, doc)
                except (RuntimeError, httpx.HTTPError) as error:
                    print(f"  FAILED    {doc.source_id}: {error}")
                    continue
                record(
                    manifest,
                    dest,
                    url=FFIEC_DOC_URL.format(path=doc.path),
                    licence="FFIEC (US interagency guidance; public domain)",
                    note=doc.note,
                    manual_path=doc.path,
                )
                print(f"  fetched   {rel(dest)}")
        finally:
            ffiec.close()

    # Renaming an artifact would otherwise leave its old key behind, and --check would then
    # report a file that is no longer meant to exist as MISSING forever.
    expected = {
        rel(SAML_D_CSV),
        rel(OBLIQA_ZIP),
        rel(OBLIQA_QA),
        *(rel(remote.dest) for remote in REGULATORY_PDFS),
        *(rel(FINRA_DIR / f"finra-rule-{number}.txt") for number, _, _ in FINRA_RULES),
        *(rel(RAW_DIR / "regulations" / "usc" / f"{s.source_id}.txt") for s in US_STATUTES),
        *(rel(CFR_DIR / f"{section.source_id}.xml") for section in CFR_SECTIONS),
        *(rel(doc.dest) for doc in FFIEC_DOCS),
    }
    for stale in set(manifest["artifacts"]) - expected:
        del manifest["artifacts"][stale]
        print(f"  dropped   {stale} (no longer an expected artifact)")

    restamped = apply_classification(manifest)
    if restamped:
        print(f"  classified {restamped} artifact(s)")

    save_manifest(manifest)
    print(f"\nManifest written to {rel(MANIFEST_PATH)}")

    problems = classification_problems(manifest)
    if problems:
        print(f"\n{len(problems)} classification problem(s) -- the knowledge base cannot be built:")
        for problem in problems:
            print(f"  {problem}")
    return 0 if saml_ok and not problems else 1


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

    problems = classification_problems(manifest)
    for problem in problems:
        print(f"  UNCLASSIFIED  {problem}")
    if problems:
        print(f"{len(problems)} artifact(s) lack the metadata the knowledge base needs")
    else:
        print("every artifact carries a role, source_id, and where applicable tier + authority")
    return 1 if failures or problems else 0


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify the manifest, fetch nothing")
    group.add_argument("--force", action="store_true", help="re-fetch everything")
    args = parser.parse_args()
    return check() if args.check else acquire(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
