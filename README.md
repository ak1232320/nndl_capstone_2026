# YMusic — Next-Gen Music Recommender on Yambda

Capstone (HSE Master's · NNDL · Representation Learning). A hybrid neural
recommender on real Yandex Music listening data ([Yambda](https://huggingface.co/datasets/yandex/yambda)):
a **SASRec** sequence model + a **two-tower** content network on audio
embeddings, fused to predict the next track for a streaming user.

Team: Anna Grishkina · Valeria Karpova · Aleksey Kosychev.

## Where things run

| Stage | Where | Why |
|-------|-------|-----|
| Eval harness, MostPop, ItemKNN baselines | **Local** (CPU) | 50M interaction files are small (<1 GB) |
| SASRec, two-tower, hybrid | **Kaggle GPU** | needs a GPU; two-tower also needs the 13.8 GB embeddings |

- **HuggingFace** = data source (Yambda lives there). Needs a read token.
- **Kaggle** = training compute (free P100 / T4×2). Needs phone verification for GPU + internet.

## Setup (local dev)

```powershell
uv sync                      # create .venv and install deps
uv run pytest                # sanity-check the eval metrics
```

Set a HuggingFace read token before downloading data:

```powershell
$env:HF_TOKEN = "hf_..."
```

## Layout

```
src/ymrec/
  config.py          dataset coordinates, paths, GTS constants, paper baselines
  data/yambda.py     download/load Yambda parquets (flat / sequential / embeddings)
  eval/split.py      Global Temporal Split (300d / 30min gap / 1d)
  eval/metrics.py    NDCG@k, Recall@k, Coverage@k (full-catalogue ranking)
  baselines/         MostPop, ItemKNN, ...        (next)
  models/            SASRec, two-tower, hybrid     (next)
tests/               unit tests for the eval harness
```

## Roadmap

1. **Eval harness** — GTS split + metrics. ✅ scaffolded
2. **Milestone 0** — reproduce paper's MostPop NDCG@10 ≈ 0.0186 on 50M (validates split+metric).
3. **Baselines** — ItemKNN (the bar to beat: 0.0781), BPR, iALS.
4. **SASRec** — causal-attention transformer over listen history (Kaggle GPU).
5. **Two-tower** — user/item towers on audio embeddings (cold-start).
6. **Hybrid** — learned fusion. Target: NDCG@10 ≥ ~0.086 (+15% over SASRec).
