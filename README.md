# YMusic — Next-Gen Music Recommender on Yambda

Capstone (HSE Master's · NNDL · Representation Learning). A neural recommender on
real Yandex Music listening data ([Yambda](https://huggingface.co/datasets/yandex/yambda)):
a **SASRec** sequence model fused with an **audio-content** tower, predicting the
next track for a streaming user.

Team: Anna Grishkina · Valeria Karpova · Aleksey Kosychev.
Repo: https://github.com/ak1232320/nndl_capstone_2026
**Full write-up: [REPORT.md](REPORT.md) · [REPORT.pdf](REPORT.pdf)** · pitch: [pitch.md](pitch.md)

## Results (our harness, Yambda-50M, Listen+, full-catalogue ranking)

All numbers from one shared protocol (Global Temporal Split, train-vocab,
seen items **not** filtered), so they are directly comparable.

| Model | NDCG@10 | Note |
|-------|---------|------|
| MostPop | 0.0171 | popularity |
| ItemKNN — cosine | 0.0418 | |
| ItemKNN — tfidf | 0.0451 | |
| ItemKNN — bm25 | **0.0709** | strongest classical baseline |
| **SASRec** | **0.0735** | beats all baselines; 98% of the paper's 0.0748 |
| Hybrid — joint embedding fusion | 0.0581 | content overfits → **negative result** |
| **Late fusion (frozen SASRec + content)** | **0.0798** | β≈0.75–1.0; **+9.7% ± 2.0% over SASRec** (5 user splits) |

**Headline:** late fusion lifts NDCG@10 from 0.0728 (SASRec) to **0.0798**, a
**+9.7% ± 2.0%** gain that holds across 5 held-out user splits (β tuned on a
separate validation split each time — no leakage; every split positive,
+6.6%…+12.3%). Naive joint fusion overfit and lost (0.0581); principled late
fusion won — that contrast is the modelling story.

**Where the gain comes from (honest finding):** content helps on **popular**
items (+12–13% on head) and slightly *hurts* the extreme long tail (−8.5% on
items with ≤5 train events). So the audio signal acts as a **taste-alignment
refinement on mainstream tracks**, *not* the cold-start/long-tail fix the pitch
hypothesised. The user content vector is the mean audio of the history (a taste
centroid), which boosts sonically-central tracks rather than rescuing rare ones.

Reference baselines (Yandex paper, arXiv:2505.22238, 50M, NDCG@10): MostPop
0.0186, BPR 0.0389, iALS 0.0407, SASRec 0.0748, **ItemKNN 0.0781**. Our SASRec
reproduces the paper to 98%, so our harness ≈ theirs (the ~9% gap on ItemKNN is
undertuning, not a protocol difference).

> Note: the original pitch's KPI (BPR 0.38, target 0.44) was a factual error —
> real full-catalogue NDCG@10 values are ~0.02–0.08. The corrected bar is to beat
> the strongest in-harness baseline.

## Where things run

Code is written locally; **all execution is on Kaggle** (notebook-driven). The
50M interaction files (~369 MB) and the audio embeddings (13.8 GB, filtered to
our 629k items) both download in ~1–2 min on Kaggle.

- **HuggingFace** = data source (Yambda). Anonymous download works; an `HF_TOKEN`
  Kaggle Secret avoids rate limits.
- **Kaggle** = compute (free T4 / P100). GPU for SASRec / hybrid / fusion.

## Kaggle workflow

Each notebook installs this package from GitHub, so notebooks stay thin and the
code stays versioned:

```python
!pip install -q --no-cache-dir --upgrade "git+https://github.com/ak1232320/nndl_capstone_2026.git"
```

Settings: **Internet On**; **GPU** for `02`/`04`/`05`/`06`; `HF_TOKEN` in *Add-ons →
Secrets* (optional). Iterate: edit code locally → push → re-run the install cell
(use a fresh kernel to avoid a stale pip git cache).

### Notebooks

**▶ [`RUN_ALL.ipynb`](notebooks/RUN_ALL.ipynb) reproduces the whole project in one
notebook** (data → baselines → SASRec → joint hybrid → late fusion → robustness →
tail → summary table). GPU + Internet on, *Run All*, ≈ 45–55 min on a T4. **Start
here to reproduce everything.** The per-stage notebooks below are for focused runs:

| Notebook | What |
|----------|------|
| `00_kaggle_smoke` | install + reproduce MostPop (harness sanity check) |
| `01_baselines` | MostPop + ItemKNN (cosine/tfidf/bm25), one table |
| `02_sasrec` | train SASRec on GPU |
| `03_content_emb_prep` | one-time: filter 13.8 GB audio embeddings → compact `.npy` (optional; `04`–`06` load inline) |
| `04_hybrid` | joint embedding-fusion hybrid (the negative-result experiment) |
| `05_fusion` | late fusion: frozen SASRec + content, validation-tuned β |
| `06_robustness` | multi-seed robustness (+9.7%±2.0%) + head/tail analysis |
| `07_report_figures` | report figures (matplotlib; no data/GPU) |

## Local dev (no training — Kaggle only)

```powershell
uv sync --extra dev          # create .venv and install deps
uv run pytest                # unit-test the eval metrics
```

## Layout

```
src/ymrec/
  config.py            dataset coordinates, paths, GTS constants, paper baselines
  data/yambda.py       download/load Yambda parquets (flat / sequential / embeddings)
  data/prep.py         Listen+ → GTS split → dense-id remap → sparse train matrix
  data/sequences.py    per-user Listen+ sequences for SASRec
  data/embeddings.py   audio embeddings filtered + aligned to the item vocab
  eval/split.py        Global Temporal Split (300d / 30min gap / 1d)
  eval/metrics.py      NDCG@k, Recall@k, Coverage@k (full-catalogue ranking)
  baselines/           MostPop, ItemKNN (cosine/tfidf/bm25 via implicit)
  models/sasrec.py     SASRec causal-attention Transformer + train/eval
  models/hybrid.py     content-augmented item embeddings (joint fusion)
  models/fusion.py     late score-level fusion (validation-tuned weight) + robustness/tail
notebooks/             Kaggle notebooks 00–07 (see table above)
scripts/               report_figures.py, build_pdf.py, baseline / milestone scripts
figures/               report figures (PNG)
tests/                 unit tests for the eval harness
REPORT.md / REPORT.pdf write-up · pitch.md / Pitch_YMusic.pptx pitch deck
```

## Protocol notes

- **Global Temporal Split**: 300d train / 30min gap / 1d test; metrics NDCG /
  Recall / Coverage @10 and @100; ranking over the full train catalogue.
- **Listen+** = a play of ≥ 50% of the track.
- **Never filter already-heard tracks** — re-listening is real in music; filtering
  seen items collapses every model (e.g. MostPop 0.0171 → 0.0033).
- IDs are sparse in a large space → remapped to dense indices before matrices /
  embeddings.
