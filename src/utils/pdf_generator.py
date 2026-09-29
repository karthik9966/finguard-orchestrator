"""Render SAML-D transaction rows as SWIFT MT103 messages inside monthly banking logs (§3.2.A).

SAML-D is tabular; the auditor's real input is unstructured. This module bridges the two:
it selects a slice of the ledger, synthesises SWIFT network messages from it, and emits
"Monthly Private Banking Institutional Transaction Logs" as text and PDF -- the documents
§6.1's uploader ingests and §3.4's loader parses.

Two things drive the design:

*Cases, not rows.* SAML-D's labels mark individual transactions, but the laundering pattern
lives in a cluster of them, and the cluster's anchor differs per typology: Structuring is a
fan-in of many senders into one collector account over consecutive days, so it is anchored on
the *receiver*; Smurfing is one sender making repeated sub-threshold deposits, anchored on the
*sender*. Sampling flagged rows independently would scatter these clusters and leave nothing
detectable, so we pick an anchor account per typology and take its whole run, plus that
account's ordinary traffic in the same month for context.

*Labels never enter the documents.* The ground truth goes to a sidecar CSV keyed by the
``:20:`` reference. A log that contained ``Laundering_type`` would hand the agent the answer
and make §8's eval suite meaningless.

Every synthesised identity (BIC, customer name, address, memo line) is derived from a hash of
the account number, so runs are reproducible and the same account keeps the same identity
across months. Banks and customers are fictional by construction.

Usage::

    uv run python -m src.utils.pdf_generator                          # last 3 months
    uv run python -m src.utils.pdf_generator --start 2023-01 --months 6
    uv run python -m src.utils.pdf_generator --max-messages 400 --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fpdf import FPDF

from src.ingestion.download import DATA_DIR, SAML_D_CSV

LEDGER_DIR = DATA_DIR / "processed" / "ledger"
LABELS_PATH = DATA_DIR / "processed" / "ledger_labels.csv"

# Phase 8's golden corpus lives apart from the dev one, and that separation is not filing tidiness.
# The detectors' window, band and minimum-counterparty numbers were all chosen by measuring recall
# against the dev batches; evaluating on the same data would be marking my own homework, and every
# number in config.yaml cites "measured across the four dev batches" as its evidence. A golden set
# also has to be *stable* -- a corpus that moves when someone regenerates the dev ledgers is not a
# baseline. So the eval profile writes here instead, and `ledger_labels.csv` keeps describing
# exactly the corpus those measurements were taken on.
EVAL_LEDGER_DIR = DATA_DIR / "processed" / "eval_ledger"
EVAL_LABELS_PATH = DATA_DIR / "processed" / "eval_labels.csv"


def destination(profile: str) -> tuple[Path, Path]:
    """Where a profile's logs and labels go. The eval corpus is deliberately not the dev corpus."""
    if profile == "eval":
        return EVAL_LEDGER_DIR, EVAL_LABELS_PATH
    return LEDGER_DIR, LABELS_PATH

# SAML-D's real base rate is 0.1%; a log at that rate is almost always empty of findings.
# We over-sample so each log contains something to audit, but stop well short of a batch
# that no auditor would believe.
MAX_FLAGGED_SHARE = 0.15

# How many of an anchor account's ordinary wires to keep as context. Enough that the laundering
# run sits inside real traffic rather than in isolation; few enough that one busy anchor cannot
# become the whole batch. The rest of the log is filled with unrelated background traffic, which
# is what a real month looks like.
CONTEXT_PER_ANCHOR = 12

USED_COLUMNS = [
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Is_laundering",
    "Laundering_type",
]

# SAML-D spells currencies and countries out in prose; SWIFT needs ISO codes. Both maps cover
# every value present in the dataset -- an unmapped value raises rather than silently defaulting.
CURRENCY_ISO = {
    "UK pounds": "GBP",
    "Euro": "EUR",
    "US dollar": "USD",
    "Swiss franc": "CHF",
    "Turkish lira": "TRY",
    "Dirham": "AED",
    "Moroccan dirham": "MAD",
    "Pakistani rupee": "PKR",
    "Indian rupee": "INR",
    "Naira": "NGN",
    "Yen": "JPY",
    "Mexican Peso": "MXN",
    "Albanian lek": "ALL",
}

COUNTRY_ISO = {
    "UK": "GB",
    "USA": "US",
    "UAE": "AE",
    "Switzerland": "CH",
    "Turkey": "TR",
    "Morocco": "MA",
    "Pakistan": "PK",
    "India": "IN",
    "Nigeria": "NG",
    "Japan": "JP",
    "Mexico": "MX",
    "Albania": "AL",
    "Spain": "ES",
    "Germany": "DE",
    "Italy": "IT",
    "France": "FR",
    "Austria": "AT",
    "Netherlands": "NL",
}

CITY = {
    "GB": "LONDON",
    "US": "NEW YORK NY",
    "AE": "ABU DHABI",
    "CH": "ZURICH",
    "TR": "ISTANBUL",
    "MA": "CASABLANCA",
    "PK": "KARACHI",
    "IN": "MUMBAI",
    "NG": "LAGOS",
    "JP": "TOKYO",
    "MX": "MEXICO CITY",
    "AL": "TIRANA",
    "ES": "MADRID",
    "DE": "FRANKFURT AM MAIN",
    "IT": "MILANO",
    "FR": "PARIS",
    "AT": "WIEN",
    "NL": "AMSTERDAM",
}

# Invented four-letter institution codes: a synthetic SAR must not name a real bank.
BANK_CODES = [
    "ADVN", "BRGT", "CLDN", "DRWD", "ELMR", "FNWK", "GRSV", "HLBR",
    "IRTN", "JSPR", "KLWD", "LNDG", "MRDN", "NRGT", "OKHM", "PLGR",
    "QNBY", "RVSD", "STNW", "THRL", "UPTN", "VRDL", "WSTM", "YRKG",
]
BRANCH_CODES = ["XXX", "2L1", "3AX", "1BR", "4CN", "5DP"]

FORENAMES = [
    "JAMES", "MARIA", "AHMED", "SOFIA", "DANIEL", "PRIYA", "OMAR", "ELENA",
    "THOMAS", "AISHA", "LUCAS", "NADIA", "HENRY", "CLARA", "YUSUF", "ROSA",
]
SURNAMES = [
    "CARTWRIGHT", "OKONKWO", "HALVORSEN", "RAMIREZ", "ABADI", "WHITFIELD",
    "DEMIREL", "LINDQVIST", "MARCHETTI", "BOUCHARD", "NAKAMURA", "ELLINGTON",
    "VASQUEZ", "KOWALSKI", "FITZGERALD", "ADEYEMI",
]
COMPANY_STEMS = [
    "MERIDIAN", "BLACKROCK PARK", "ASHFORD", "CALDERA", "NORTHWIND", "STELLAR BAY",
    "ORCHARD LANE", "VERDANT", "KESTREL", "SUMMIT ROW", "IRONGATE", "PALEWATER",
]
COMPANY_SUFFIXES = ["HOLDINGS LTD", "TRADING LLC", "CAPITAL PARTNERS", "GROUP SA", "VENTURES LTD"]
STREETS = ["THREADNEEDLE ST", "KINGSWAY", "HARBOUR ROAD", "OLD MILL LANE", "CANAL VIEW", "MARKET SQUARE"]

# Memo lines are drawn independently of the label: a memo that correlated with
# Laundering_type would leak the answer as surely as printing the label itself.
MEMOS = [
    "/RFB/INVOICE SETTLEMENT",
    "/RFB/CONSULTANCY FEES",
    "/RFB/TRADE SETTLEMENT",
    "/RFB/FAMILY SUPPORT",
    "/RFB/PROPERTY DEPOSIT",
    "/RFB/CONTRACT MILESTONE",
    "/RFB/EQUIPMENT PURCHASE",
    "/RFB/INTERCOMPANY TRANSFER",
    "/RFB/PROFESSIONAL SERVICES",
    "/RFB/LOAN REPAYMENT",
]
CHARGE_CODES = ["SHA", "OUR", "BEN"]

INSTITUTION = "NORTHGATE PRIVATE BANK"
DIVISION = "INSTITUTIONAL CLIENT SERVICES"

# How a typology's cluster is found in the ledger. Getting this wrong silently produces a
# log with a laundering label but no laundering *pattern*: an anchored search over Cycle
# returns one wire, because its 382 flagged edges span 382 distinct senders and 382 distinct
# receivers and nothing concentrates.
ANCHORED = "anchored"      # one account collects or disperses the run
CHAINED = "chained"        # the pattern is a path: A -> B -> C -> A
SINGLE_WIRE = "single_wire"  # the pattern IS one transaction; the signal is the amount
# The pattern is a connected structure with no single anchor: two layers, a hub with both sides,
# a block of senders. Anchoring one of these plants only the edges touching one account -- which is
# what happened to Scatter-Gather before v2: anchored on its source, it was planted as its scatter
# leg alone, and the gather leg never reached the ledger.
COMPONENT = "component"

TYPOLOGY_SHAPE = {
    # Explicit rather than falling through to the ANCHORED default. Both are in scope per PRD §2,
    # and the two detectors that matter most should not depend on what `shape_of` happens to do
    # with an unknown label.
    "Fan_In": ANCHORED,
    "Fan_Out": ANCHORED,
    "Cycle": CHAINED,
    "Scatter-Gather": COMPONENT,
    "Gather-Scatter": COMPONENT,
    "Deposit-Send": COMPONENT,
    "Layered_Fan_In": COMPONENT,
    "Layered_Fan_Out": COMPONENT,
    "Bipartite": COMPONENT,
    "Stacked Bipartite": COMPONENT,
    "Over-Invoicing": SINGLE_WIRE,
    "Single_large": SINGLE_WIRE,
}


def shape_of(typology: str) -> str:
    """Fan-in/fan-out shapes are the common case, so anchoring is the default."""
    return TYPOLOGY_SHAPE.get(typology, ANCHORED)


# SAML-D labels that map onto the nine in-scope patterns (PRD v2 §1), seeded first when picking
# monthly cases. Layered_Fan_In/Out both become `layered_fan`, Bipartite and Stacked Bipartite both
# `bipartite`.
#
# v1 excluded Deposit-Send, Gather-Scatter, Layered and Bipartite; v2 brings them in. Smurfing goes
# the other way: PRD v2 §2 defers it ("structuring-adjacent"), and it cannot stay planted as
# structuring -- all 932 of its rows are cash deposits at a median $2,629, so it would sit in the
# structuring gold set and look like deposit-send's cash leg at the same time.
PRIORITY_TYPOLOGIES = [
    "Structuring",       # -> structuring
    "Fan_In",            # -> fan_in
    "Fan_Out",           # -> fan_out
    "Cycle",             # -> cycle
    "Scatter-Gather",    # -> scatter_gather
    "Gather-Scatter",    # -> gather_scatter
    "Deposit-Send",      # -> deposit_send
    "Layered_Fan_In",    # -> layered_fan
    "Layered_Fan_Out",   # -> layered_fan
    "Bipartite",         # -> bipartite
    "Stacked Bipartite", # -> bipartite
]

# Labelled suspicious, but out of scope per PRD v2 §2. Kept in the ledgers as *unflagged* context so
# precision is measurable against activity that genuinely looks odd -- never planted as a case,
# because a detector is not expected to find them and recall must not be diluted by them.
OUT_OF_SCOPE_TYPOLOGIES = frozenset({
    "Smurfing", "Cash_Withdrawal", "Behavioural_Change_1", "Behavioural_Change_2",
    "Single_large", "Over-Invoicing",
})

# PRD v2 §5.1 "Option 1": SAML-D is jurisdiction-neutral, so the two threshold-sensitive patterns
# draw gold instances whose amounts already sit just under the US $10,000 CTR level. A cluster
# qualifies when at least half its threshold-relevant rows (every row for structuring, the cash
# deposits for deposit-send) fall in this band. A *data selection* criterion, deliberately not tied
# to the structuring detector's configured band: tuning the band must not move the gold set.
#
# Measured over all of SAML-D: 21 of 224 receiver-anchored structuring clusters qualify, and 45 of
# 473 Deposit-Send cash deposits are in the band.
ALIGNED_BAND = (8_000, 10_000)

# Smallest planted instance, where a typology's whole pattern is smaller than `min_cluster`.
# Deposit-send is one deposit and one send: SAML-D spreads a hub's ~6 pairs over ~250 days, so a
# month usually holds exactly one pair, and a 3-edge minimum was rejecting the complete pattern.
MIN_CLUSTER_OF = {"Deposit-Send": 2}
THRESHOLD_SENSITIVE = frozenset({"Structuring", "Deposit-Send"})


def threshold_aligned(typology: str, rows: pd.DataFrame) -> bool:
    """Whether a cluster is an Option-1 instance: its relevant amounts hug $10,000 from below."""
    if typology == "Deposit-Send" and "Payment_type" in rows:
        rows = rows[rows.Payment_type == "Cash Deposit"]
    if rows.empty:
        return False
    low, high = ALIGNED_BAND
    return bool(((rows.Amount >= low) & (rows.Amount < high)).mean() >= 0.5)


def partition_of(key: int) -> str:
    """Which corpus a SAML-D cluster belongs to, fixed by its anchor account.

    The dev and eval corpora slice the same months, and both take the largest clusters first -- so
    before this, 405 of the 467 flagged rows in the dev directory were planted in the eval corpus
    as well, and "held out" was not true. A hash of the anchor splits SAML-D's clusters in two once
    and for all: a cluster is tuning data or golden data, never both, whichever months or sizes a
    profile asks for.
    """
    digest = hashlib.sha256(f"finguard-partition:{key}".encode()).digest()
    return "eval" if digest[0] % 2 else "dev"


def admissible(typology: str, rows: pd.DataFrame, config: "SliceConfig", key: int) -> bool:
    """Whether a profile may plant this cluster: Option 1 first, then the dev/eval partition.

    ``aligned`` (the golden corpus) takes *every* threshold-aligned instance of the two sensitive
    typologies whatever its partition -- there are only ~21 structuring ones in all of SAML-D --
    and ``exclude_aligned`` (the tuning corpus) never takes one, so they stay held out. Everything
    else follows `partition_of`.
    """
    if typology in THRESHOLD_SENSITIVE and config.threshold_selection != "any":
        aligned = threshold_aligned(typology, rows)
        if config.threshold_selection == "aligned":
            return aligned
        if aligned:
            return False
    return config.partition is None or partition_of(key) == config.partition


def components(cases: pd.DataFrame) -> list[pd.Index]:
    """Weakly connected components of one typology's edges, largest first, stable on ties."""
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for sender, receiver in zip(cases.Sender_account, cases.Receiver_account):
        parent[find(int(sender))] = find(int(receiver))
    groups: dict[int, list] = defaultdict(list)
    for index, sender in zip(cases.index, cases.Sender_account):
        groups[find(int(sender))].append(index)
    return sorted((pd.Index(sorted(g)) for g in groups.values()), key=lambda g: (-len(g), g[0]))


# --- synthetic identities ------------------------------------------------------------


def account_rng(account: int, salt: str = "") -> random.Random:
    """Stable per-account randomness: one account keeps one identity across every run."""
    return random.Random(f"finguard:{salt}:{account}")


def iso_country(location: str) -> str:
    try:
        return COUNTRY_ISO[location]
    except KeyError:
        raise KeyError(f"unmapped SAML-D bank location: {location!r}") from None


def iso_currency(currency: str) -> str:
    try:
        return CURRENCY_ISO[currency]
    except KeyError:
        raise KeyError(f"unmapped SAML-D currency: {currency!r}") from None


def bic_for(account: int, location: str) -> str:
    rng = account_rng(account, "bic")
    return rng.choice(BANK_CODES) + iso_country(location) + rng.choice(["2L", "3A", "AA", "1B"]) + rng.choice(BRANCH_CODES)


def party_for(account: int, location: str) -> tuple[str, str, str]:
    """Return (name, street line, city line) for an account."""
    rng = account_rng(account, "party")
    country = iso_country(location)
    if rng.random() < 0.3:
        name = f"{rng.choice(COMPANY_STEMS)} {rng.choice(COMPANY_SUFFIXES)}"
    else:
        name = f"{rng.choice(FORENAMES)} {rng.choice(SURNAMES)}"
    street = f"{rng.randint(1, 240)} {rng.choice(STREETS)}"
    return name, street, f"{CITY[country]} {country}"


def uetr_for(reference: str) -> str:
    rng = random.Random(f"finguard:uetr:{reference}")
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


# --- message rendering ---------------------------------------------------------------


def mt103(row: pd.Series, reference: str) -> list[str]:
    """One MT103 customer credit transfer: blocks 1, 2, 3 and 4 (§3.2.A)."""
    sender_bic = bic_for(int(row.Sender_account), row.Sender_bank_location)
    receiver_bic = bic_for(int(row.Receiver_account), row.Receiver_bank_location)

    ordering_name, ordering_street, ordering_city = party_for(
        int(row.Sender_account), row.Sender_bank_location
    )
    beneficiary_name, beneficiary_street, beneficiary_city = party_for(
        int(row.Receiver_account), row.Receiver_bank_location
    )

    value_date = pd.Timestamp(row.Date).strftime("%y%m%d")
    currency = iso_currency(row.Payment_currency)
    amount = f"{row.Amount:,.2f}".replace(",", "").replace(".", ",")

    rng = random.Random(f"finguard:msg:{reference}")
    return [
        f"{{1:F01{sender_bic}0000000000}}{{2:I103{receiver_bic}N}}{{3:{{121:{uetr_for(reference)}}}}}{{4:",
        f":20:{reference}",
        ":23B:CRED",
        f":32A:{value_date}{currency}{amount}",
        f":50K:/{int(row.Sender_account)}",
        ordering_name,
        ordering_street,
        ordering_city,
        f":52A:{sender_bic}",
        f":57A:{receiver_bic}",
        f":59:/{int(row.Receiver_account)}",
        beneficiary_name,
        beneficiary_street,
        beneficiary_city,
        f":70:{rng.choice(MEMOS)}",
        f":71A:{rng.choice(CHARGE_CODES)}",
        f":72:/INS/{row.Payment_type.upper()} {row.Time}",
        "-}",
    ]


def statement_header(period: pd.Period, message_count: int, account_count: int) -> list[str]:
    start = period.start_time.strftime("%d %b %Y").upper()
    end = period.end_time.strftime("%d %b %Y").upper()
    return [
        "=" * 78,
        f"{INSTITUTION} - {DIVISION}",
        "MONTHLY PRIVATE BANKING INSTITUTIONAL TRANSACTION LOG",
        "=" * 78,
        f"Statement reference : NPB-LOG-{period}",
        f"Reporting period    : {start} to {end}",
        f"Messages in batch   : {message_count}",
        f"Distinct accounts   : {account_count}",
        "Message standard    : SWIFT MT103 (single customer credit transfer)",
        "Source              : synthetic ledger derived from SAML-D (Oztas et al., 2023)",
        "=" * 78,
        "",
    ]


def render_text(period: pd.Period, frame: pd.DataFrame) -> str:
    accounts = pd.concat([frame.Sender_account, frame.Receiver_account]).nunique()
    lines = statement_header(period, len(frame), accounts)
    for _, row in frame.iterrows():
        lines.extend(mt103(row, row.Reference))
        lines.append("")
    lines.append(f"END OF STATEMENT - {len(frame)} MESSAGES")
    return "\n".join(lines) + "\n"


def render_pdf(text: str, dest: Path) -> None:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Courier", size=7)
    for line in text.splitlines():
        # Core PDF fonts are latin-1; the log is ASCII by construction, but be explicit.
        pdf.cell(0, 3.1, line.encode("latin-1", "replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(dest))


# --- slice selection -----------------------------------------------------------------


@dataclass(frozen=True)
class SliceConfig:
    start: str
    months: int
    max_messages: int
    cases_per_month: int
    min_cluster: int
    seed: int
    # Which named emission this is. Carried so a caller can ask for "dev" rather than restating
    # four numbers, and so the profile that produced a corpus is recoverable from the call.
    profile: str = "dev"
    # How many *distinct* clusters to plant per typology per month. 1 is the Phase 2 behaviour and
    # stays the default, so the dev and large corpora regenerate identically. Phase 8 needs 15
    # instances per pattern for the golden set, and with six in-scope typologies and six unused
    # months one-per-typology tops out at 36 -- hence the knob. Declared after `profile` so the
    # existing positional call sites keep meaning what they say.
    clusters_per_typology: int = 1
    # PRD v2 Option 1 -- see `admissible`. `any` keeps the large batch as it was.
    threshold_selection: str = "any"
    # Which half of SAML-D's clusters this profile may plant -- see `partition_of`. None plants
    # from both, which only a corpus nobody measures against may do.
    partition: str | None = None


# The emissions Phase 2 calls for. `dev` is three working batches plus a clean control; `large`
# is the one batch big enough to measure Evaluation Design's five-minute KPI against, and the
# detectors meet it now rather than in Phase 8.
#
# ~500 rather than the old 220: five typologies need room to plant without crowding each other,
# where 220 was sized for four geometric primitives.
#
# v2: nine patterns from eleven SAML-D labels, and every profile plants every typology every month.
# Dev and eval now share months but never a cluster -- see `partition_of`.
PROFILES = {
    # Dev spans seven months in two runs because 2023-04 and 2023-05 are the large batch and the
    # clean control, and a log is named by its month. Two clusters per typology per month gives
    # ~14 tuning instances per label: three, which is what one month each gave, is too few to tell
    # a threshold from noise.
    "dev": [
        SliceConfig("2022-12", 4, 1_500, 11, 3, 20260814, "dev", clusters_per_typology=2,
                    threshold_selection="exclude_aligned", partition="dev"),
        SliceConfig("2023-06", 3, 1_500, 11, 3, 20260814, "dev", clusters_per_typology=2,
                    threshold_selection="exclude_aligned", partition="dev"),
        SliceConfig("2023-05", 1, 500, 0, 3, 20260814, "dev-control"),
    ],
    # 2023-04: outside the dev window, and inside SAML-D's range. The corpus ends at 2023-08,
    # so a month after that silently produces nothing.
    "large": [SliceConfig("2023-04", 1, 10_000, 12, 3, 20260814, "large",
                          threshold_selection="exclude_aligned", partition="dev")],
    # Phase 8's golden corpus. Six months SAML-D has and nothing else uses -- dev holds 2023-06..08,
    # the control is 2023-05 and the 10k batch is 2023-04 -- with three clusters per typology per
    # month. That yields ~90 planted instances, comfortably above the 15 per pattern the golden set
    # selects, and leaves the existing corpora untouched so every number measured against them
    # still stands.
    # v2: 11 months at 2,400 messages, all eleven in-scope labels, two clusters of each per month.
    # The sizing is a consequence of MAX_FLAGGED_SHARE: ~22 clusters of ~9 transactions stays near
    # 8% flagged, and the golden set needs 135. It uses the same months as the dev corpus on purpose
    # -- a different slice of the same source, in its own directory, rather than a different period
    # whose typology mix would be an accident of the calendar -- and Option 1's aligned clusters are
    # the eval profile's alone: dev is built with `exclude_aligned`.
    "eval": [SliceConfig("2022-10", 11, 2_400, 11, 3, 20260814, "eval", clusters_per_typology=3,
                         threshold_selection="aligned", partition="eval")],
}


# Structural typologies are planted only when the *whole* SAML-D cluster falls inside the batch
# month. SAML-D's gather-scatter, layered and bipartite clusters run 13-23 days, so a month boundary
# routinely cuts one in half, and the half that lands in the month is a different shape -- measured
# on the dev corpus, most "Gather-Scatter" instances planted without this rule had only their
# scatter side, which is a fan-out wearing the wrong label. PRD v2 §2.7 scopes cross-month schemes
# out, so a golden instance must be one a single batch can contain. Deposit-Send is exempt: its
# hubs span ~250 days and each deposit-then-send pair is a complete instance on its own.
WHOLE_CLUSTER_ONLY = frozenset({
    "Scatter-Gather", "Gather-Scatter", "Layered_Fan_In", "Layered_Fan_Out",
    "Bipartite", "Stacked Bipartite",
})
ROW_KEY = ["Date", "Time", "Sender_account", "Receiver_account", "Amount"]


def contained_clusters(flagged: pd.DataFrame) -> set[tuple]:
    """Row keys of every structural-typology row whose whole cluster sits in one month."""
    keys: set[tuple] = set()
    for typology, rows in flagged[flagged.Laundering_type.isin(WHOLE_CLUSTER_ONLY)].groupby(
        "Laundering_type"
    ):
        for component in components(rows):
            members = rows.loc[component]
            if members.Date.str.slice(0, 7).nunique() == 1:
                keys.update(map(tuple, members[ROW_KEY].itertuples(index=False)))
    return keys


def load_window(csv: Path, periods: list[pd.Period]) -> pd.DataFrame:
    """Stream the 9.5M-row CSV and keep only the months we are rendering.

    Every flagged row in the file is kept aside while streaming, because whether a cluster is whole
    inside a month can only be decided against the months either side of it -- see
    `WHOLE_CLUSTER_ONLY`. The result carries that as a `Contained` column.
    """
    wanted = {str(p) for p in periods}
    frames = []
    flagged = []
    for chunk in pd.read_csv(csv, usecols=USED_COLUMNS, chunksize=1_000_000):
        month = chunk.Date.str.slice(0, 7)
        frames.append(chunk[month.isin(wanted)])
        flagged.append(chunk[chunk.Is_laundering == 1])
    window = pd.concat(frames, ignore_index=True)
    window["Period"] = window.Date.str.slice(0, 7)
    contained = contained_clusters(pd.concat(flagged, ignore_index=True))
    window["Contained"] = [
        key in contained for key in map(tuple, window[ROW_KEY].itertuples(index=False))
    ]
    return window


def anchor_side(cases: pd.DataFrame) -> str:
    """Whichever endpoint concentrates the typology is the account the pattern hangs off."""
    by_sender = cases.Sender_account.value_counts()
    by_receiver = cases.Receiver_account.value_counts()
    return "Sender_account" if by_sender.max() >= by_receiver.max() else "Receiver_account"


def select_chain(cases: pd.DataFrame, max_hops: int = 10) -> pd.Index:
    """Follow a chained typology's edges through the graph and return the longest run.

    Money in a laundering ring moves A -> B -> C -> A, losing 10-20% per hop to the
    launderer's cut, so no account appears twice and ``value_counts`` finds nothing. The
    pattern is only visible by walking. Within a single month these chains run 10-15 hops,
    which is why the graph is built per month: the whole ring then lands in one log, where
    an auditor -- or the agent -- can actually follow it.
    """
    edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, row in cases.iterrows():
        edges[int(row.Sender_account)].append((int(row.Receiver_account), index)) # type: ignore

    longest: list[int] = []
    for start in edges:
        node, visited, path = start, {start}, []
        while len(path) < max_hops:
            step = next(
                (edge for edge in edges.get(node, ()) if edge[0] not in visited or edge[0] == start),
                None,
            )
            if step is None:
                break
            receiver, index = step
            path.append(index)
            if receiver == start:
                break  # ring closed
            visited.add(receiver)
            node = receiver
        if len(path) > len(longest):
            longest = path
    return pd.Index(longest)


def select_cases(
    month: pd.DataFrame,
    config: SliceConfig,
    rng: random.Random,
    rotation: int = 0,
    cases_wanted: int | None = None,
) -> pd.DataFrame:
    """Pick whole laundering clusters plus the anchor accounts' ordinary traffic.

    How a cluster is found depends on the typology's shape (see ``TYPOLOGY_SHAPE``):
    fan-shaped runs hang off one account, rings must be walked, and a couple of typologies
    are legitimately a single wire.

    ``rotation`` advances the priority list month over month, so a multi-month corpus
    exercises a spread of typologies instead of repeating the same three.
    """
    suspicious = month[month.Is_laundering == 1]
    if suspicious.empty:
        return month.iloc[0:0]

    present = [t for t in PRIORITY_TYPOLOGIES if t in set(suspicious.Laundering_type)]
    if present:
        offset = rotation % len(present)
        present = present[offset:] + present[:offset]
    # Anything left over that is still in scope. Out-of-scope typologies are never planted as a
    # case: PRD §2 excludes them, so a detector is not expected to find them, and seeding one
    # dilutes recall with activity the system is right to ignore. The old list did plant them --
    # Deposit-Send, Layered_Fan_In and Over-Invoicing were all in the ledgers -- which is why
    # three of the five in-scope detectors had nothing to find.
    others = sorted(
        set(suspicious.Laundering_type) - set(present) - OUT_OF_SCOPE_TYPOLOGIES
    )
    rng.shuffle(others)

    selected_indices: list[pd.Index] = []
    anchors: set[int] = set()
    wanted = cases_wanted or config.cases_per_month
    for typology in (present + others)[:wanted]:
        cases = suspicious[suspicious.Laundering_type == typology]
        shape = shape_of(typology)

        if shape == SINGLE_WIRE:
            # Nothing to cluster: an over-invoiced payment is suspicious because $2.7M is
            # implausible for the stated trade, not because it repeats.
            selected_indices.append(cases.nlargest(config.clusters_per_typology, "Amount").index)
            continue

        if shape == COMPONENT:
            taken = 0
            for component in components(cases):
                minimum = MIN_CLUSTER_OF.get(typology, config.min_cluster)
                if taken >= config.clusters_per_typology or len(component) < minimum:
                    break
                rows = cases.loc[component]
                if typology in WHOLE_CLUSTER_ONLY and not rows.Contained.all():
                    continue
                key = int(min(rows.Sender_account.min(), rows.Receiver_account.min()))
                if not admissible(typology, rows, config, key):
                    continue
                selected_indices.append(component)
                # The busiest account carries the context traffic and the US domicile.
                busiest = pd.concat([rows.Sender_account, rows.Receiver_account]).value_counts()
                anchors.add(int(busiest.index[0]))
                taken += 1
            continue

        if shape == CHAINED:
            # Several rings per month where the month's edges support it, taken one at a time from
            # what the previous walk did not use. Disjoint by construction, so a second chain is a
            # genuine second ring rather than the first one with an edge missing -- and it has to
            # clear `min_cluster` on its own, so a two-hop remnant is not planted as a pattern.
            remaining = cases
            walked = 0
            while walked < config.clusters_per_typology:
                chain = select_chain(remaining)
                if len(chain) < config.min_cluster:
                    break
                if not admissible(typology, remaining.loc[chain], config,
                                  int(remaining.loc[chain].Sender_account.min())):
                    remaining = remaining.drop(chain)
                    continue
                selected_indices.append(chain)
                anchors.update(int(a) for a in remaining.loc[chain].Sender_account)
                remaining = remaining.drop(chain)
                walked += 1
            if walked:
                continue
            # Too few edges this month to form a ring -- fall through and anchor instead.

        side = anchor_side(cases)
        counts = cases[side].value_counts()
        qualifying = counts[counts >= config.min_cluster]
        qualifying = qualifying[[
            admissible(typology, cases[cases[side] == account], config, int(account))
            for account in qualifying.index
        ]]
        if qualifying.empty:
            if config.partition is not None or (
                typology in THRESHOLD_SENSITIVE and config.threshold_selection != "any"
            ):
                continue  # no admissible instance this month; planting a wrong one is worse
            qualifying = counts.head(1)
        # Several anchors rather than only the busiest, so one month can contribute more than one
        # instance of a typology. Taken in descending size and then by account, which is stable
        # across runs -- a golden dataset whose membership moves between builds is not golden.
        chosen = sorted(qualifying.head(config.clusters_per_typology).index)
        for anchor_account in chosen:
            anchor = int(anchor_account)
            anchors.add(anchor)
            selected_indices.append(cases[cases[side] == anchor].index)

    # Each entry in `selected_indices` *is* one planted instance, so the id is recorded here rather
    # than reconstructed later. Pattern-level recall -- "did the system report this fan-in?" -- needs
    # to know which transactions form one instance, and a consumer guessing at it from anchors and
    # typologies would be re-deriving a decision this function already made. `Cycle` is a walked
    # chain with no anchor at all, so there is no reliable way to guess it from outside.
    flagged = month.loc[sorted({i for idx in selected_indices for i in idx})].copy()
    flagged["Cluster"] = pd.NA
    for ordinal, index in enumerate(selected_indices, start=1):
        typology = str(month.loc[index[0], "Laundering_type"])
        flagged.loc[flagged.index.intersection(index), "Cluster"] = f"{typology}-{ordinal:02d}"
    # The collector account's legitimate traffic is what makes the run look like a pattern
    # rather than a list of isolated transfers -- but it has to be *some* of that traffic.
    # Taking all of it lets one anchor swamp the log: an anchor whose ordinary month happens
    # to include a 180-wire Normal_Fan_In consumed the entire context budget and produced a
    # "monthly log" that was 85% one account receiving money on a single day. No detector can
    # work on that, and no auditor would recognise it as a month of private banking.
    clean = month[month.Is_laundering == 0].copy()
    clean["Cluster"] = pd.NA
    kept: set[int] = set()
    for anchor in sorted(anchors):
        touching = clean[(clean.Sender_account == anchor) | (clean.Receiver_account == anchor)]
        if len(touching) > CONTEXT_PER_ANCHOR:
            touching = touching.sample(CONTEXT_PER_ANCHOR, random_state=rng.randrange(2**31))
        kept.update(touching.index)
    return pd.concat([flagged, clean.loc[sorted(kept)]]).drop_duplicates()


def build_month(
    month: pd.DataFrame,
    config: SliceConfig,
    rng: random.Random,
    rotation: int = 0,
    cases_wanted: int | None = None,
) -> pd.DataFrame:
    cases = select_cases(month, config, rng, rotation, cases_wanted)

    # A 13-hop ring drags in context traffic for 13 accounts, which alone can exceed the
    # batch size. Trim context to fit the budget; never drop a flagged wire, or the labels
    # would describe cases the log does not contain.
    flagged = cases[cases.Is_laundering == 1]
    context = cases[cases.Is_laundering == 0]
    context_budget = max(config.max_messages - len(flagged), 0)
    if len(context) > context_budget:
        context = context.sample(context_budget, random_state=rng.randrange(2**31))
    cases = pd.concat([flagged, context])

    remaining = max(config.max_messages - len(cases), 0)
    background = month[(month.Is_laundering == 0) & (~month.index.isin(cases.index))]
    if remaining and len(background) > remaining:
        background = background.sample(remaining, random_state=rng.randrange(2**31))
    combined = pd.concat([cases, background.head(remaining)])
    return combined.sort_values(["Date", "Time"]).reset_index(drop=True)


def build_month_within_budget(
    month: pd.DataFrame, config: SliceConfig, period: pd.Period, rotation: int
) -> pd.DataFrame:
    """Fit as many laundering clusters as the plausibility ceiling allows.

    Real batches are overwhelmingly clean. Packing clusters in until a fifth of the batch
    is suspicious produces a log no auditor would recognise -- and an easy win for the
    agent. Drop clusters until the flagged share is credible; each attempt is seeded from
    the run seed so the result stays reproducible whichever attempt wins.
    """
    if config.cases_per_month == 0:
        # A control batch: ordinary traffic only. Without one, the router's "no candidates ->
        # no model, $0.00" path can be unit-tested but never demonstrated on a real document,
        # because every other batch has patterns planted in it by construction.
        rng = random.Random(f"{config.seed}:{period}:clean")
        clean = month[month.Is_laundering == 0]
        if len(clean) > config.max_messages:
            clean = clean.sample(config.max_messages, random_state=rng.randrange(2**31))
        return clean.sort_values(["Date", "Time"]).reset_index(drop=True)

    for wanted in range(config.cases_per_month, 0, -1):
        rng = random.Random(f"{config.seed}:{period}:{wanted}")
        # Stride by the cluster count so consecutive months draw disjoint typologies.
        frame = build_month(month, config, rng, rotation * config.cases_per_month, wanted)
        share = frame.Is_laundering.mean() if len(frame) else 0.0
        if share <= MAX_FLAGGED_SHARE or wanted == 1:
            if wanted != config.cases_per_month:
                print(f"    {period}: capped at {wanted} clusters (flagged share ceiling)")
            return frame
    raise AssertionError("unreachable: the wanted == 1 branch always returns")



# --- re-domiciling: one US institution's ledger (Phase 2) ------------------------------
#
# The corpus is US law now, and a finding citing 31 CFR 1020.320 against a GB->GB wire is
# incoherent. SAML-D cannot supply US traffic: it is 96.6% UK-origin, and only 36 suspicious
# USA-sender rows exist in all 9.5M -- none of them fan-in, fan-out or smurfing. So the anchor
# leg is re-domiciled instead, and the counterparties keep their SAML-D countries so the
# cross-border corridors stay real.
#
# This must happen on the frame, before rendering: `iso_country` and `iso_currency` raise on an
# unmapped value and `mt103` calls them while it renders.

HOME_LOCATION = "USA"
# SAML-D's own spelling; CURRENCY_ISO maps it to USD.
HOME_CURRENCY = "US dollar"


def institution_accounts(frame: pd.DataFrame) -> set[int]:
    """The accounts this bank holds -- the anchor of each planted cluster.

    ``anchor_side`` already decides which endpoint concentrates a typology, and that endpoint is
    exactly the account the institution's analyst is looking at. Reusing it here keeps the
    domicile consistent with the shape the cluster was selected for.
    """
    ours: set[int] = set()
    flagged = frame[frame.Is_laundering == 1]
    for _, cases in flagged.groupby("Laundering_type"):
        if cases.empty:
            continue
        ours.update(int(account) for account in cases[anchor_side(cases)])
    return ours


def redomicile(frame: pd.DataFrame) -> pd.DataFrame:
    """Put one leg of every message at the US institution, and denominate in USD.

    Amounts are **relabelled, not converted**. SAML-D's values were never really pounds, and the
    structuring detector keys on the $10,000 CTR and $3,000 recordkeeping thresholds: converting
    at an FX rate would lift a cluster sitting just under 10,000 straight over the threshold and
    stop it being structuring at all. Relabelling preserves the relative magnitudes that make a
    cluster a cluster, and leaves it where the rule can see it.
    """
    frame = frame.copy()
    ours = institution_accounts(frame)

    sender_is_ours = frame.Sender_account.astype("int64").isin(ours)
    receiver_is_ours = frame.Receiver_account.astype("int64").isin(ours)
    # Every message is on this bank's ledger, so one leg is always ours. Where the row belongs to
    # no planted cluster, the sender is the customer by default.
    take_sender = sender_is_ours | ~receiver_is_ours

    frame.loc[take_sender, "Sender_bank_location"] = HOME_LOCATION
    frame.loc[~take_sender, "Receiver_bank_location"] = HOME_LOCATION
    frame["Payment_currency"] = HOME_CURRENCY
    frame["Received_currency"] = HOME_CURRENCY
    return frame


def assign_references(frame: pd.DataFrame, counter: int) -> tuple[pd.DataFrame, int]:
    """:20: is limited to 16 characters -- FGO + YYMMDD + 5-digit sequence fits in 14."""
    references = []
    for date in frame.Date:
        counter += 1
        references.append(f"FGO{pd.Timestamp(date).strftime('%y%m%d')}{counter:05d}")
    return frame.assign(Reference=references), counter


# --- entry point ---------------------------------------------------------------------


def generate(config: SliceConfig, *, append: bool = False) -> pd.DataFrame:
    if not SAML_D_CSV.exists():
        raise SystemExit("SAML-D missing -- run: uv run python -m src.ingestion.download")

    ledger_dir, labels_path = destination(config.profile)
    periods = [pd.Period(config.start, freq="M") + i for i in range(config.months)]
    print(f"Loading {periods[0]}..{periods[-1]} from {SAML_D_CSV.name}")
    window = load_window(SAML_D_CSV, periods)
    print(f"  {len(window):,} rows in window ({int(window.Is_laundering.sum()):,} flagged)")

    ledger_dir.mkdir(parents=True, exist_ok=True)
    # Logs from a previous run would outlive the labels CSV, leaving the sidecar describing a
    # corpus that no longer matches what is on disk. In append mode only the months being
    # rewritten are cleared, so a control batch can be added without rebuilding the corpus.
    stale = (
        [p for period in periods for p in ledger_dir.glob(f"{period}_private_banking_log.*")]
        if append
        else list(ledger_dir.glob("*_private_banking_log.*"))
    )
    for path in stale:
        path.unlink()

    existing = (
        pd.read_csv(labels_path) if append and labels_path.exists() else pd.DataFrame()
    )
    # :20: references must stay unique across the whole corpus, not just within one run.
    counter = int(existing.Reference.str[-5:].astype(int).max()) if len(existing) else 0
    labels = []

    for rotation, period in enumerate(periods):
        month = window[window.Period == str(period)]
        if month.empty:
            print(f"  {period}: no rows, skipped")
            continue

        frame = redomicile(build_month_within_budget(month, config, period, rotation))
        frame, counter = assign_references(frame, counter)

        text = render_text(period, frame)
        stem = f"{period}_private_banking_log"
        (ledger_dir / f"{stem}.txt").write_text(text)
        render_pdf(text, ledger_dir / f"{stem}.pdf")

        flagged = int(frame.Is_laundering.sum())
        typologies = sorted(set(frame.loc[frame.Is_laundering == 1, "Laundering_type"]))
        labels.append(frame.assign(Log_file=f"{stem}.pdf"))
        print(
            f"  {period}: {len(frame):>4} messages, {flagged:>3} flagged "
            f"({flagged / len(frame):.1%}) -- {', '.join(typologies)}"
        )

    if not labels:
        # Every requested month was empty. Concatenating nothing raises deep inside pandas with
        # "No objects to concatenate", which says nothing about the cause -- and the cause is
        # almost always a month outside SAML-D's 2022-10..2023-08 range.
        raise SystemExit(
            f"no rows for {periods[0]}..{periods[-1]} -- SAML-D covers 2022-10 to 2023-08"
        )

    ledger_labels = pd.concat(labels, ignore_index=True)
    if len(existing):
        rewritten = set(ledger_labels.Log_file)
        ledger_labels = pd.concat(
            [existing[~existing.Log_file.isin(rewritten)], ledger_labels], ignore_index=True
        ).sort_values(["Log_file", "Date", "Time"], ignore_index=True)
    ledger_labels.to_csv(labels_path, index=False)
    print(f"\nLogs      -> {ledger_dir.relative_to(DATA_DIR.parent)}")
    print(f"Labels    -> {labels_path.relative_to(DATA_DIR.parent)} ({len(ledger_labels):,} rows)")
    return ledger_labels


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile", choices=sorted(PROFILES),
        help="emit a named batch set instead of one ad-hoc slice",
    )
    parser.add_argument("--start", default="2023-06", help="first month, YYYY-MM")
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--max-messages", type=int, default=220, help="messages per monthly log")
    parser.add_argument("--cases-per-month", type=int, default=3, help="laundering clusters per log")
    parser.add_argument("--min-cluster", type=int, default=3, help="minimum wires per cluster")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep logs for other months and merge into the existing labels CSV",
    )
    args = parser.parse_args()

    if args.profile:
        # The control batch is appended so it joins the labels rather than replacing them; the
        # first config in a profile clears the ledger, the rest add to it.
        for index, config in enumerate(PROFILES[args.profile]):
            print(f"\n[{config.profile}] {config.months} x {config.max_messages} messages")
            generate(config=config, append=args.append or index > 0)
        return 0

    generate(
        append=args.append,
        config=SliceConfig(
            start=args.start,
            months=args.months,
            max_messages=args.max_messages,
            cases_per_month=args.cases_per_month,
            min_cluster=args.min_cluster,
            seed=args.seed,
            profile="adhoc",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
