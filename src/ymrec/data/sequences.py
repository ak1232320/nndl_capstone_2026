"""Build per-user listening sequences for SASRec, sharing the baselines' protocol.

Same GTS split, same Listen+ definition, same train-derived item vocabulary as
`prep.py`, so SASRec metrics are directly comparable to MostPop / ItemKNN.

Model item ids are 1..n_items (0 is the padding token). Evaluation is done in
ORIGINAL item-id space: model idx j (0-based over items 1..n_items) maps back to
`item_ids[j]`, and relevant sets are original test item ids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ymrec.config import DEFAULT_SIZE, Size
from ymrec.data.prep import load_listen_plus
from ymrec.eval import split as gts


@dataclass
class SeqData:
    seqs: list[np.ndarray]          # per train user: model item idx (1..n_items), chronological, capped to maxlen
    user_item_sets: list[set[int]]  # per train user: set of model idx (for negative sampling)
    user_ids: np.ndarray            # train uid (sorted), aligned with seqs
    item_ids: np.ndarray            # vocab: model idx j (0-based) -> original item id item_ids[j]
    eval_pos: np.ndarray            # indices into seqs/user_ids for evaluated users
    relevant: list[set[int]]        # original test item ids per eval user (aligned with eval_pos)
    maxlen: int
    bounds: gts.SplitBounds

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_users(self) -> int:
        return len(self.seqs)


def build_sequences(
    size: Size = DEFAULT_SIZE, maxlen: int = 200, token: str | None = None
) -> SeqData:
    pos = load_listen_plus(size=size, token=token)
    train, test, bounds = gts.split(pos)

    item_ids = np.sort(train["item_id"].unique().to_numpy())   # 0-based vocab -> original id
    user_ids = np.sort(train["uid"].unique().to_numpy())

    # Per-user chronological item lists (original ids).
    grp = (
        train.sort(["uid", "timestamp"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id"))
    )
    seq_by_uid = {int(u): items for u, items in zip(grp["uid"].to_list(), grp["item_id"].to_list())}

    seqs: list[np.ndarray] = []
    user_item_sets: list[set[int]] = []
    for u in user_ids:
        orig = np.asarray(seq_by_uid[int(u)], dtype=np.int64)
        midx = (np.searchsorted(item_ids, orig) + 1).astype(np.int64)  # 1..n_items
        midx = midx[-maxlen:]  # keep most recent maxlen
        seqs.append(midx)
        user_item_sets.append(set(int(x) for x in midx))

    # Eval users (have train history) and their test relevant sets (original ids).
    test_agg = test.group_by("uid").agg(pl.col("item_id").unique().alias("items"))
    uid_to_pos = {int(u): i for i, u in enumerate(user_ids)}
    eval_pos: list[int] = []
    relevant: list[set[int]] = []
    for u, items in zip(test_agg["uid"].to_list(), test_agg["items"].to_list()):
        if int(u) in uid_to_pos:
            eval_pos.append(uid_to_pos[int(u)])
            relevant.append(set(int(x) for x in items))
    order = np.argsort(eval_pos)
    eval_pos_arr = np.asarray(eval_pos, dtype=np.int64)[order]
    relevant = [relevant[i] for i in order]

    return SeqData(
        seqs=seqs,
        user_item_sets=user_item_sets,
        user_ids=user_ids,
        item_ids=item_ids,
        eval_pos=eval_pos_arr,
        relevant=relevant,
        maxlen=maxlen,
        bounds=bounds,
    )
