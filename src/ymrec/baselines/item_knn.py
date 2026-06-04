"""Item-based k-NN collaborative filtering via `implicit.nearest_neighbours`.

Computes top-`neighbors` item-item similarities from the train user-item matrix,
then scores each user by summing similarities over their history. This is the
strongest published baseline on Yambda-50M (NDCG@10 = 0.0781).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

_MODELS = {"cosine": "CosineRecommender", "tfidf": "TFIDFRecommender", "bm25": "BM25Recommender"}


def fit_recommend(
    train_ui: csr_matrix,
    eval_user_idx: np.ndarray,
    item_ids: np.ndarray,
    k_rec: int,
    neighbors: int = 100,
    variant: str = "cosine",
    filter_seen: bool = False,
) -> np.ndarray:
    """Fit ItemKNN and return top-`k_rec` recommendations as ORIGINAL item ids.

    Returns an int array (n_eval, k_rec); padding (when a user has fewer than
    k_rec scorable items) is -1, which matches nothing in the relevant sets.
    """
    import implicit.nearest_neighbours as nn

    model = getattr(nn, _MODELS[variant])(K=neighbors)
    model.fit(train_ui, show_progress=False)
    ids, _ = model.recommend(
        eval_user_idx,
        train_ui[eval_user_idx],
        N=k_rec,
        filter_already_liked_items=filter_seen,
    )
    ids = np.asarray(ids)
    # Map dense idx -> original item id; keep -1 padding as -1.
    return np.where(ids >= 0, item_ids[np.clip(ids, 0, None)], -1)
