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

TYPOLOGY_SHAPE = {
    "Cycle": CHAINED,
    "Over-Invoicing": SINGLE_WIRE,
    "Single_large": SINGLE_WIRE,
}


def shape_of(typology: str) -> str:
    """Fan-in/fan-out shapes are the common case, so anchoring is the default."""
    return TYPOLOGY_SHAPE.get(typology, ANCHORED)


# Typologies the blueprint calls out by name are seeded first when picking monthly cases.
PRIORITY_TYPOLOGIES = [
    "Structuring",
    "Smurfing",
    "Deposit-Send",
    "Cycle",
    "Scatter-Gather",
    "Gather-Scatter",
    "Layered_Fan_In",
    "Layered_Fan_Out",
    "Over-Invoicing",
]


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


def load_window(csv: Path, periods: list[pd.Period]) -> pd.DataFrame:
    """Stream the 9.5M-row CSV and keep only the months we are rendering."""
    wanted = {str(p) for p in periods}
    frames = []
    for chunk in pd.read_csv(csv, usecols=USED_COLUMNS, chunksize=1_000_000):
        month = chunk.Date.str.slice(0, 7)
        frames.append(chunk[month.isin(wanted)])
    window = pd.concat(frames, ignore_index=True)
    window["Period"] = window.Date.str.slice(0, 7)
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
        edges[int(row.Sender_account)].append((int(row.Receiver_account), index))

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
    others = sorted(set(suspicious.Laundering_type) - set(present))
    rng.shuffle(others)

    selected_indices: list[pd.Index] = []
    anchors: set[int] = set()
    for typology in (present + others)[: cases_wanted or config.cases_per_month]:
        cases = suspicious[suspicious.Laundering_type == typology]
        shape = shape_of(typology)

        if shape == SINGLE_WIRE:
            # Nothing to cluster: an over-invoiced payment is suspicious because £2.7M is
            # implausible for the stated trade, not because it repeats.
            selected_indices.append(cases.nlargest(1, "Amount").index)
            continue

        if shape == CHAINED:
            chain = select_chain(cases)
            if len(chain) >= config.min_cluster:
                selected_indices.append(chain)
                anchors.update(int(a) for a in cases.loc[chain].Sender_account)
                continue
            # Too few edges this month to form a ring -- fall through and anchor instead.

        side = anchor_side(cases)
        counts = cases[side].value_counts()
        counts = counts[counts >= config.min_cluster]
        if counts.empty:
            counts = cases[side].value_counts().head(1)
        anchor = int(counts.index[0])
        anchors.add(anchor)
        selected_indices.append(cases[cases[side] == anchor].index)

    flagged = month.loc[sorted({i for idx in selected_indices for i in idx})]
    # The collector account's legitimate traffic is what makes the run look like a pattern
    # rather than a list of isolated transfers -- but it has to be *some* of that traffic.
    # Taking all of it lets one anchor swamp the log: an anchor whose ordinary month happens
    # to include a 180-wire Normal_Fan_In consumed the entire context budget and produced a
    # "monthly log" that was 85% one account receiving money on a single day. No detector can
    # work on that, and no auditor would recognise it as a month of private banking.
    clean = month[month.Is_laundering == 0]
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

    periods = [pd.Period(config.start, freq="M") + i for i in range(config.months)]
    print(f"Loading {periods[0]}..{periods[-1]} from {SAML_D_CSV.name}")
    window = load_window(SAML_D_CSV, periods)
    print(f"  {len(window):,} rows in window ({int(window.Is_laundering.sum()):,} flagged)")

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    # Logs from a previous run would outlive the labels CSV, leaving the sidecar describing a
    # corpus that no longer matches what is on disk. In append mode only the months being
    # rewritten are cleared, so a control batch can be added without rebuilding the corpus.
    stale = (
        [p for period in periods for p in LEDGER_DIR.glob(f"{period}_private_banking_log.*")]
        if append
        else list(LEDGER_DIR.glob("*_private_banking_log.*"))
    )
    for path in stale:
        path.unlink()

    existing = (
        pd.read_csv(LABELS_PATH) if append and LABELS_PATH.exists() else pd.DataFrame()
    )
    # :20: references must stay unique across the whole corpus, not just within one run.
    counter = int(existing.Reference.str[-5:].astype(int).max()) if len(existing) else 0
    labels = []

    for rotation, period in enumerate(periods):
        month = window[window.Period == str(period)]
        if month.empty:
            print(f"  {period}: no rows, skipped")
            continue

        frame = build_month_within_budget(month, config, period, rotation)
        frame, counter = assign_references(frame, counter)

        text = render_text(period, frame)
        stem = f"{period}_private_banking_log"
        (LEDGER_DIR / f"{stem}.txt").write_text(text)
        render_pdf(text, LEDGER_DIR / f"{stem}.pdf")

        flagged = int(frame.Is_laundering.sum())
        typologies = sorted(set(frame.loc[frame.Is_laundering == 1, "Laundering_type"]))
        labels.append(frame.assign(Log_file=f"{stem}.pdf"))
        print(
            f"  {period}: {len(frame):>4} messages, {flagged:>3} flagged "
            f"({flagged / len(frame):.1%}) -- {', '.join(typologies)}"
        )

    ledger_labels = pd.concat(labels, ignore_index=True)
    if len(existing):
        rewritten = set(ledger_labels.Log_file)
        ledger_labels = pd.concat(
            [existing[~existing.Log_file.isin(rewritten)], ledger_labels], ignore_index=True
        ).sort_values(["Log_file", "Date", "Time"], ignore_index=True)
    ledger_labels.to_csv(LABELS_PATH, index=False)
    print(f"\nLogs      -> {LEDGER_DIR.relative_to(DATA_DIR.parent)}")
    print(f"Labels    -> {LABELS_PATH.relative_to(DATA_DIR.parent)} ({len(ledger_labels):,} rows)")
    return ledger_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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

    generate(
        append=args.append,
        config=SliceConfig(
            start=args.start,
            months=args.months,
            max_messages=args.max_messages,
            cases_per_month=args.cases_per_month,
            min_cluster=args.min_cluster,
            seed=args.seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
