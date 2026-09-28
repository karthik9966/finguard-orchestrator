"""`detection_confidence = f(tightness, count, window compactness, aggregate ratio)`.

How sure the deterministic pass is that a shape is real -- never how suspicious it is. The
model decides suspicion, after it has been shown the rule.

**Tightness is measured on the threshold-band subset, not the whole group.** That is the recorded
weakness of the pre-migration scoring: one legitimate $34,121 wire inside an otherwise tight
cluster moved its coefficient of variation from 0.024 to 1.039 -- 43x -- so a score keyed on the
whole group went blind to the tight subset within it.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from decimal import Decimal
from typing import Sequence

from src.config import get_config


def coefficient_of_variation(amounts: Sequence[Decimal]) -> float:
    """Standard deviation as a fraction of the mean -- spread on a comparable scale.

    Ported from `utils/detectors.py`. Ten payments averaging 5,673 with a deviation of 139 give
    0.024: effectively the same payment ten times. Unitless on purpose, so a $5,000 cluster and a
    $500,000 cluster are judged on shape rather than size.
    """
    if len(amounts) < 2:
        return 0.0
    values = [float(amount) for amount in amounts]
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _tightness(amounts: Sequence[Decimal]) -> float:
    """1.0 when every amount is the same; falls away as they spread."""
    return max(0.0, 1.0 - min(coefficient_of_variation(amounts), 1.0))


def _member_count(count: int, minimum: int) -> float:
    """Saturating: three transfers is the floor, and past about twice that, more adds little."""
    if count <= minimum:
        return 0.0
    return min(1.0, (count - minimum) / max(minimum, 1))


def _window_compactness(timestamps: Sequence[datetime], window_days: int) -> float:
    """1.0 when the whole group lands in an instant, 0.0 when it fills the window.

    A day of activity inside a seven-day window is a far stronger signal than the same count
    spread across all seven, and the old primitives could not tell the two apart at all.
    """
    if len(timestamps) < 2 or window_days <= 0:
        return 1.0
    span = (max(timestamps) - min(timestamps)).total_seconds() / 86_400
    return max(0.0, 1.0 - min(span / window_days, 1.0))


def _aggregate_ratio(total: Decimal, threshold: Decimal | None) -> float:
    """How far past the threshold the group totals. None when no threshold applies."""
    if threshold is None or threshold <= 0:
        return 0.5
    return min(1.0, float(total) / float(threshold) / 2.0)


def score(
    *,
    amounts: Sequence[Decimal],
    timestamps: Sequence[datetime],
    minimum_members: int,
    band_amounts: Sequence[Decimal] | None = None,
    threshold: Decimal | None = None,
) -> float:
    """Combine the four signals with the weights from config.yaml.

    `band_amounts` is the threshold-band subset where one exists; tightness is measured on it
    rather than on `amounts` for the reason in this module's docstring.
    """
    detection = get_config().detection
    weights = detection.confidence_weights
    total = sum(amounts, Decimal(0))

    value = (
        weights.tightness * _tightness(band_amounts if band_amounts else amounts)
        + weights.member_count * _member_count(len(amounts), minimum_members)
        + weights.window_compactness * _window_compactness(timestamps, detection.window_days)
        + weights.aggregate_ratio * _aggregate_ratio(total, threshold)
    )
    return round(min(1.0, max(0.0, value)), 4)
