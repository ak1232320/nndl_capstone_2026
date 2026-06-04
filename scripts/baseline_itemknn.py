"""ItemKNN baseline on Yambda-50M — the strongest published baseline (NDCG@10 0.0781).

Run:  uv run python scripts/baseline_itemknn.py
"""
from __future__ import annotations

import time

from ymrec.config import TOPK, PAPER_BASELINES_NDCG10
from ymrec.data.prep import prepare
from ymrec.eval.metrics import evaluate_ranking
from ymrec.baselines.item_knn import fit_recommend

K = max(TOPK)


def main() -> None:
    t0 = time.time()
    p = prepare(size="50m")
    print(f"prepared in {time.time()-t0:.1f}s: "
          f"n_users={p.n_users:,} n_items={p.n_items:,} "
          f"train_nnz={p.train_ui.nnz:,} eval_users={len(p.eval_user_idx):,}")

    paper = PAPER_BASELINES_NDCG10["50m"]["ItemKNN"]
    configs = [
        ("cosine", 100, False),
        ("cosine", 100, True),
        ("tfidf", 100, False),
        ("bm25", 100, False),
    ]
    for variant, neighbors, filter_seen in configs:
        t = time.time()
        recs = fit_recommend(
            p.train_ui, p.eval_user_idx, p.item_ids, K,
            neighbors=neighbors, variant=variant, filter_seen=filter_seen,
        )
        res = evaluate_ranking(recs, p.relevant, n_items=p.n_items, ks=TOPK)
        tag = f"{variant} K={neighbors} filter_seen={filter_seen}"
        print(f"\n--- ItemKNN ({tag})  [{time.time()-t:.1f}s] ---")
        for key in ("ndcg@10", "ndcg@100", "recall@10", "recall@100",
                    "coverage@10", "coverage@100"):
            print(f"  {key:14s} {res[key]:.4f}")
        print(f"  paper ItemKNN NDCG@10 = {paper:.4f}  (ours = {res['ndcg@10']:.4f})")


if __name__ == "__main__":
    main()
