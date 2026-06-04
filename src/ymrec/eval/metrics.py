"""Ranking metrics for full-catalogue evaluation: NDCG@k, Recall@k, Coverage@k.

The Yambda baselines rank EVERY catalogue item for each user (no sampled
negatives), which is why absolute NDCG@10 values are small (~0.02-0.08). We
mirror that: a model produces a ranked top-K list per user, and we score it
against the user's held-out relevant items.

Definitions (standard; to be validated by reproducing MostPop NDCG@10 = 0.0186):
    DCG@k    = sum_{i=1..k} rel_i / log2(i + 1)          rel_i in {0, 1}
    IDCG@k   = sum_{i=1..min(k, R)} 1 / log2(i + 1)      R = #relevant
    NDCG@k   = DCG@k / IDCG@k
    Recall@k = (#relevant items in top-k) / R
    Coverage@k = |union of top-k items over all users| / catalogue_size
Users with no relevant items are skipped.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _dcg_weights(k: int) -> np.ndarray:
    # positions 1..k -> 1/log2(2..k+1)
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate_ranking(
    topk_items: np.ndarray,
    relevant: Sequence[set[int]],
    n_items: int,
    ks: Sequence[int] = (10, 100),
) -> dict[str, float]:
    """Compute NDCG@k, Recall@k and Coverage@k averaged over users.

    Args:
        topk_items: int array, shape (n_users, K), each row a ranked list of item
            ids (best first). K must be >= max(ks).
        relevant:   per-user set of held-out relevant item ids (same order/length
            as rows of topk_items).
        n_items:    catalogue size (for Coverage).
        ks:         cutoffs to report.

    Returns:
        Flat dict, e.g. {"ndcg@10": ..., "recall@10": ..., "coverage@10": ..., ...}.
    """
    topk_items = np.asarray(topk_items)
    n_users, K = topk_items.shape
    if K < max(ks):
        raise ValueError(f"topk_items has K={K} but max(ks)={max(ks)}")
    if len(relevant) != n_users:
        raise ValueError("relevant must have one entry per user row")

    weights = _dcg_weights(max(ks))
    out: dict[str, list[float]] = {}
    for k in ks:
        out[f"ndcg@{k}"] = []
        out[f"recall@{k}"] = []
    covered: dict[int, set[int]] = {k: set() for k in ks}

    for u in range(n_users):
        rel = relevant[u]
        row = topk_items[u]
        for k in ks:
            row_k = row[:k]
            covered[k].update(int(x) for x in row_k)
            R = len(rel)
            if R == 0:
                continue
            hits = np.fromiter((1.0 if int(it) in rel else 0.0 for it in row_k),
                               dtype=np.float64, count=k)
            dcg = float(np.dot(hits, weights[:k]))
            idcg = float(weights[:min(k, R)].sum())
            out[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
            out[f"recall@{k}"].append(float(hits.sum()) / R)

    result: dict[str, float] = {}
    for k in ks:
        result[f"ndcg@{k}"] = float(np.mean(out[f"ndcg@{k}"])) if out[f"ndcg@{k}"] else 0.0
        result[f"recall@{k}"] = float(np.mean(out[f"recall@{k}"])) if out[f"recall@{k}"] else 0.0
        result[f"coverage@{k}"] = len(covered[k]) / n_items if n_items else 0.0
    return result
