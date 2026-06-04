"""Milestone 0 — reproduce the paper's MostPop NDCG@10 on Yambda-50M.

Validates the whole eval harness (GTS split + metrics) against a known number:
paper reports MostPop NDCG@10 = 0.0186 (Listen+, 50M). Downloads listens.parquet
(~369 MB) on first run.

Run:  uv run python scripts/milestone0_mostpop.py
"""
from __future__ import annotations

import polars as pl

from ymrec.config import LISTEN_POSITIVE_RATIO, TOPK, PAPER_BASELINES_NDCG10
from ymrec.data.yambda import interactions_path
from ymrec.eval import split as gts
from ymrec.eval.metrics import evaluate_ranking
from ymrec.baselines.most_pop import popularity_ranking, recommend

K = max(TOPK)


def per_user_sets(df: pl.DataFrame) -> dict[int, set[int]]:
    agg = df.group_by("uid").agg(pl.col("item_id").unique().alias("items"))
    return {int(u): set(items) for u, items in zip(agg["uid"], agg["items"])}


def main() -> None:
    path = interactions_path("listens", size="50m", layout="flat")
    print(f"listens parquet: {path}")

    listens = pl.read_parquet(
        path, columns=["uid", "item_id", "timestamp", "played_ratio_pct"]
    )
    print(f"raw listen events: {listens.height:,}")

    # Listen+ = played at least LISTEN_POSITIVE_RATIO percent of the track.
    pos = listens.filter(pl.col("played_ratio_pct") >= LISTEN_POSITIVE_RATIO)
    print(f"Listen+ events (>= {LISTEN_POSITIVE_RATIO}%): {pos.height:,}")

    span = gts.describe_timespan(pos)
    print(f"timespan: {span}")

    train, test, bounds = gts.split(pos)
    print(f"bounds: {bounds}")
    print(f"train Listen+: {train.height:,}   test Listen+: {test.height:,}")

    n_items = pos["item_id"].n_unique()
    print(f"catalogue (distinct items in Listen+): {n_items:,}")

    train_seen = per_user_sets(train)
    test_rel = per_user_sets(test)
    users_with_hist = sorted(set(test_rel) & set(train_seen))
    users_all = sorted(test_rel)
    print(f"test users — all: {len(users_all):,}   with train history: {len(users_with_hist):,}")

    ranking = popularity_ranking(train)
    paper = PAPER_BASELINES_NDCG10["50m"]["MostPop"]

    configs = [
        ("with-history, no-seen-filter", users_with_hist, None),
        ("all-users, no-seen-filter", users_all, None),
        ("with-history, seen-filtered", users_with_hist, train_seen),
    ]
    for label, users, seen in configs:
        relevant = [test_rel[u] for u in users]
        recs = recommend(ranking, users, K, seen=seen)
        res = evaluate_ranking(recs, relevant, n_items=n_items, ks=TOPK)
        print(f"\n--- MostPop ({label}) ---")
        for key in ("ndcg@10", "ndcg@100", "recall@10", "recall@100",
                    "coverage@10", "coverage@100"):
            print(f"  {key:14s} {res[key]:.4f}")
        print(f"  paper NDCG@10 = {paper:.4f}  (ours = {res['ndcg@10']:.4f})")


if __name__ == "__main__":
    main()
