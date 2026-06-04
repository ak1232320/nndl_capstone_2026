"""Shared preparation: Listen+ -> GTS split -> dense-id sparse matrix + eval sets.

Yambda uids/item_ids are sparse in a large id space (uid up to 1e6, item_id up
to 9.39e6) even though 50M has only ~10k users / ~934k items, so we remap to
dense indices for matrices / embeddings. The item vocabulary is built from the
TRAIN window (only train items are recommendable); test items outside it are
cold and simply never get recommended (they still count in a user's relevant
set, so recall is honest).

Evaluation is done in ORIGINAL item-id space: model outputs (dense idx) are
mapped back through `item_ids`, and relevant sets are original test item ids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from ymrec.config import LISTEN_POSITIVE_RATIO, DEFAULT_SIZE, Size
from ymrec.data.yambda import interactions_path
from ymrec.eval import split as gts


@dataclass
class Prepared:
    train_ui: csr_matrix       # (n_users, n_items) interaction counts on dense idx
    user_ids: np.ndarray       # dense uidx -> original uid (sorted)
    item_ids: np.ndarray       # dense iidx -> original item_id (sorted, TRAIN vocab)
    eval_user_idx: np.ndarray  # dense user idx that are evaluated
    relevant: list[set[int]]   # per eval user: ORIGINAL item ids in test Listen+
    bounds: gts.SplitBounds

    @property
    def n_users(self) -> int:
        return self.train_ui.shape[0]

    @property
    def n_items(self) -> int:
        return self.train_ui.shape[1]


def load_listen_plus(size: Size = DEFAULT_SIZE, token: str | None = None) -> pl.DataFrame:
    """Load Listen+ events (played_ratio_pct >= threshold) as (uid, item_id, timestamp)."""
    path = interactions_path("listens", size=size, layout="flat", token=token)
    listens = pl.read_parquet(
        path, columns=["uid", "item_id", "timestamp", "played_ratio_pct"]
    )
    return (
        listens.filter(pl.col("played_ratio_pct") >= LISTEN_POSITIVE_RATIO)
        .select("uid", "item_id", "timestamp")
    )


def prepare(size: Size = DEFAULT_SIZE, token: str | None = None) -> Prepared:
    pos = load_listen_plus(size=size, token=token)
    train, test, bounds = gts.split(pos)

    # Dense vocab from TRAIN (sorted -> searchsorted gives exact indices).
    user_ids = np.sort(train["uid"].unique().to_numpy())
    item_ids = np.sort(train["item_id"].unique().to_numpy())
    n_users, n_items = len(user_ids), len(item_ids)

    uidx = np.searchsorted(user_ids, train["uid"].to_numpy())
    iidx = np.searchsorted(item_ids, train["item_id"].to_numpy())
    data = np.ones(len(uidx), dtype=np.float32)
    # COO -> CSR sums duplicates, giving per (user,item) play counts.
    train_ui = csr_matrix((data, (uidx, iidx)), shape=(n_users, n_items))
    train_ui.sum_duplicates()

    # Test relevant sets, in original item-id space, for users with train history.
    test_agg = test.group_by("uid").agg(pl.col("item_id").unique().alias("items"))
    user_set = set(user_ids.tolist())
    pairs: list[tuple[int, set[int]]] = []
    for u, items in zip(test_agg["uid"].to_list(), test_agg["items"].to_list()):
        if u in user_set:
            pairs.append((int(np.searchsorted(user_ids, u)), set(int(x) for x in items)))
    pairs.sort(key=lambda p: p[0])
    eval_user_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    relevant = [p[1] for p in pairs]

    return Prepared(train_ui, user_ids, item_ids, eval_user_idx, relevant, bounds)
