"""Global Temporal Split (GTS) — the Yambda evaluation protocol.

The paper splits by absolute time, not per user: a fixed training window, a
short gap (so interactions straddling the boundary are dropped), then a test
window. Model state is frozen at the test-window start; users with empty
history at that point are discarded.

    |<--------- train (300 days) --------->|<-30min gap->|<- test (1 day) ->|

Timestamp units in Yambda are uint32. The unit is verified at runtime by
`describe_timespan` (the full span should land near ~331 days); pass the
matching `seconds_per_unit` if it differs from raw seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ymrec.config import GTS_TRAIN_DAYS, GTS_GAP_MINUTES, GTS_TEST_DAYS


@dataclass(frozen=True)
class SplitBounds:
    t_min: int
    train_end: int
    test_start: int
    test_end: int
    seconds_per_unit: float


def describe_timespan(events: pl.DataFrame, ts_col: str = "timestamp") -> dict[str, float]:
    """Report the raw timestamp range and the implied span in days (assuming seconds).

    Use this once to confirm the timestamp unit before trusting the split.
    """
    t_min = int(events[ts_col].min())
    t_max = int(events[ts_col].max())
    span = t_max - t_min
    return {
        "t_min": t_min,
        "t_max": t_max,
        "span_units": span,
        "span_days_if_seconds": span / 86400.0,
    }


def compute_bounds(
    events: pl.DataFrame,
    ts_col: str = "timestamp",
    train_days: int = GTS_TRAIN_DAYS,
    gap_minutes: int = GTS_GAP_MINUTES,
    test_days: int = GTS_TEST_DAYS,
    seconds_per_unit: float = 1.0,
) -> SplitBounds:
    """Compute the GTS boundaries from the data's start time.

    `seconds_per_unit` converts one timestamp unit to seconds (1.0 if timestamps
    are already in seconds).
    """
    t_min = int(events[ts_col].min())
    day = 86400.0 / seconds_per_unit
    minute = 60.0 / seconds_per_unit
    train_end = int(t_min + train_days * day)
    test_start = int(train_end + gap_minutes * minute)
    test_end = int(test_start + test_days * day)
    return SplitBounds(t_min, train_end, test_start, test_end, seconds_per_unit)


def split(
    events: pl.DataFrame,
    ts_col: str = "timestamp",
    bounds: SplitBounds | None = None,
    **kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, SplitBounds]:
    """Split a flat event table into (train, test) by absolute time.

    Returns (train_df, test_df, bounds). Test keeps only events in
    [test_start, test_end). The 30-minute gap region is dropped entirely.
    """
    if bounds is None:
        bounds = compute_bounds(events, ts_col=ts_col, **kwargs)
    train = events.filter(pl.col(ts_col) < bounds.train_end)
    test = events.filter(
        (pl.col(ts_col) >= bounds.test_start) & (pl.col(ts_col) < bounds.test_end)
    )
    return train, test, bounds
