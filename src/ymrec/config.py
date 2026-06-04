"""Central configuration: dataset coordinates, paths, and protocol constants.

All facts here were verified against the HF dataset card and the paper
(arXiv:2505.22238) on 2026-06-04. See pitch.md for project framing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# --- Hugging Face dataset coordinates -------------------------------------
HF_REPO_ID = "yandex/yambda"
HF_REPO_TYPE = "dataset"

Size = Literal["50m", "500m", "5b"]
Layout = Literal["flat", "sequential"]
EventType = Literal["listens", "likes", "dislikes", "unlikes", "undislikes", "multi_event"]

EVENT_TYPES: tuple[str, ...] = (
    "listens", "likes", "dislikes", "unlikes", "undislikes", "multi_event",
)

# Working set for the capstone (fits Kaggle GPU; matches published baselines).
DEFAULT_SIZE: Size = "50m"

# --- Local paths -----------------------------------------------------------
# Override the data root with the YMREC_DATA env var (e.g. Kaggle /kaggle/input).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("YMREC_DATA", PROJECT_ROOT / "data"))
ARTIFACTS_ROOT = Path(os.environ.get("YMREC_ARTIFACTS", PROJECT_ROOT / "artifacts"))

# --- Evaluation protocol (Global Temporal Split, per the paper) -----------
# Training period 300 days, 30-minute gap, 1-day test window.
GTS_TRAIN_DAYS = 300
GTS_GAP_MINUTES = 30
GTS_TEST_DAYS = 1

# A "Listen+" positive = track played for at least this fraction of its length.
LISTEN_POSITIVE_RATIO = 50  # played_ratio_pct >= 50

# Metric cutoffs reported in the paper.
TOPK = (10, 100)

# --- Published baselines (Listen+, NDCG@10) for sanity-checking ourselves --
# Source: arXiv:2505.22238, Table of Listen+ results.
PAPER_BASELINES_NDCG10 = {
    "50m":  {"MostPop": 0.0186, "DecayPop": 0.0260, "ItemKNN": 0.0781,
             "iALS": 0.0407, "BPR": 0.0389, "SANSA": 0.0069, "SASRec": 0.0748},
    "500m": {"ItemKNN": 0.0708, "iALS": 0.0384, "BPR": 0.0400, "SASRec": 0.0754},
    "5b":   {"iALS": 0.0388, "BPR": 0.0408, "SASRec": 0.0647},
}
