"""Download and load Yambda parquet files from Hugging Face.

Layout on HF (verified 2026-06-04):
    flat/{size}/{event}.parquet         one row per event, sorted by (uid, timestamp)
    sequential/{size}/{event}.parquet   one row per user: uid, item_ids[], timestamps[], ...
    embeddings.parquet                  GLOBAL, 13.8 GB, audio embeddings for 7.72M tracks
    album_item_mapping.parquet / artist_item_mapping.parquet

We pull individual files with `hf_hub_download` (cached) rather than
`load_dataset`, so we fetch only what a given stage needs. The 13.8 GB
embeddings file is intentionally NOT downloaded by the early stages — only the
two-tower / hybrid models need it.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

from ymrec.config import HF_REPO_ID, HF_REPO_TYPE, DATA_ROOT, Size, Layout, EventType


def _download(filename: str, token: str | None = None) -> Path:
    """Download a single file from the dataset repo into the local HF cache.

    Returns the local path. `token` is the HF read token (or None to use the
    cached login / HF_TOKEN env var).
    """
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=filename,
        token=token,
        local_dir=DATA_ROOT / "hf",
    )
    return Path(path)


def interactions_path(
    event: EventType,
    size: Size = "50m",
    layout: Layout = "flat",
    token: str | None = None,
) -> Path:
    """Path to a downloaded interactions parquet, fetching it if absent."""
    return _download(f"{layout}/{size}/{event}.parquet", token=token)


def load_interactions(
    event: EventType,
    size: Size = "50m",
    layout: Layout = "flat",
    token: str | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load an interactions table as a Polars DataFrame."""
    path = interactions_path(event, size=size, layout=layout, token=token)
    return pl.read_parquet(path, columns=columns)


def scan_interactions(
    event: EventType,
    size: Size = "50m",
    layout: Layout = "flat",
    token: str | None = None,
) -> pl.LazyFrame:
    """Lazily scan an interactions table (preferred for the large `listens` file)."""
    path = interactions_path(event, size=size, layout=layout, token=token)
    return pl.scan_parquet(path)


def embeddings_path(token: str | None = None) -> Path:
    """Path to the GLOBAL embeddings parquet (13.8 GB) — downloads on first call.

    Only call this for the two-tower / hybrid stage. Filter to the items present
    in your working size variant immediately and persist the small subset.
    """
    return _download("embeddings.parquet", token=token)
