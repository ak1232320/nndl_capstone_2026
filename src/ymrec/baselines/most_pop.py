"""MostPop baseline: recommend the globally most-played tracks.

Popularity is the count of Listen+ events per item in the training window.
Every user gets the same ranked list, optionally with their already-seen
(train) items removed.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def popularity_ranking(train: pl.DataFrame, item_col: str = "item_id") -> np.ndarray:
    """Return item ids ordered by descending train popularity (most popular first)."""
    counts = (
        train.group_by(item_col)
        .len()
        .sort("len", descending=True)
    )
    return counts[item_col].to_numpy()


def recommend(
    ranking: np.ndarray,
    users: list[int],
    k: int,
    seen: dict[int, set[int]] | None = None,
) -> np.ndarray:
    """Build a (n_users, k) array of recommendations from a global ranking.

    If `seen` is given, each user's already-interacted items are removed from
    their list before truncating to k.
    """
    if seen is None:
        top = ranking[:k]
        return np.tile(top, (len(users), 1))

    ranking_list = ranking.tolist()
    out = np.empty((len(users), k), dtype=ranking.dtype)
    for i, u in enumerate(users):
        s = seen.get(u)
        if not s:
            out[i] = ranking[:k]
            continue
        # Walk the global ranking, skipping seen items, until we collect k.
        picks = []
        for it in ranking_list:
            if it in s:
                continue
            picks.append(it)
            if len(picks) == k:
                break
        out[i] = picks
    return out
