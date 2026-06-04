"""Load Yambda audio embeddings, aligned to our item vocabulary.

The global `embeddings.parquet` (13.8 GB, 7.72M tracks, columns item_id / embed /
normalized_embed) covers only ~82% of the catalogue, so some of our 629k train
items have no audio vector (cold on the content side) — we return a coverage
mask for them.

`normalized_embed` is already L2-normalised, so a dot product is cosine
similarity — ideal for the content tower.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ymrec.data.yambda import embeddings_path


def embedding_dim(path, column: str = "normalized_embed") -> int:
    row = pl.scan_parquet(path).select(column).head(1).collect()
    return len(row[column][0])


def load_aligned_embeddings(
    item_ids: np.ndarray, token: str | None = None, column: str = "normalized_embed"
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (emb, mask, dim) where emb[j] is the audio vector for original id
    item_ids[j] (zeros if absent) and mask[j] is True iff that item has a vector.

    `item_ids` must be sorted ascending (our vocab is). Downloads the 13.8 GB
    embeddings file on first call.
    """
    path = embeddings_path(token)
    dim = embedding_dim(path, column)

    vocab = pl.Series("item_id", item_ids)
    df = (
        pl.scan_parquet(path)
        .select(["item_id", column])
        .filter(pl.col("item_id").is_in(vocab))
        .collect(streaming=True)
    )

    got_ids = df["item_id"].to_numpy()
    vecs = np.asarray(df[column].to_list(), dtype=np.float32)  # (k, dim)

    n = len(item_ids)
    emb = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    idx = np.searchsorted(item_ids, got_ids)
    emb[idx] = vecs
    mask[idx] = True
    return emb, mask, dim
