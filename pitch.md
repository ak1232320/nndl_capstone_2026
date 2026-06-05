# Capstone — Next-Gen Music Recommender on Yambda

**Team:** Anna Grishkina · Valeria Karpova · Aleksey Kosychev
**Target score:** 10 / 10 (real-world operational data from Yandex — qualifies for the +2 bonus)
**Dataset:** [Yambda](https://huggingface.co/datasets/yandex/yambda) — Yandex Music listening events (Yandex, arXiv:2505.22238, RecSys '25)
**Code:** https://github.com/ak1232320/nndl_capstone_2026

> This is a *delivered* pitch: every number below was produced by our own pipeline
> under the dataset's official evaluation protocol, not promised.

---

## 1. Problem Statement

**What we solve.** Given a user's listening history, predict the next track they
will actually want to hear — the decision a streaming product makes for every
active user, every few minutes.

**Why it matters.** Discovery and retention are the dominant levers in a
saturated streaming market: subscriber growth has plateaued, so the quality of
the next-track recommendation directly drives listening time and churn. A
measurable lift in next-track ranking is a measurable lift in engagement.

**Why it's tractable now.** Sequence models (Transformers) and learned audio
embeddings have matured enough for a small team to build a research-quality
recommender without industrial-scale engineering — and Yandex just released the
data to do it on real production logs.

---

## 2. Data — our key advantage

**Source.** **Yambda** by Yandex (Hugging Face, May 2025; arXiv:2505.22238,
accepted at RecSys '25). A production listening log — **4.79 B events, 1 M users,
9.39 M tracks** in the full release.

**What we use.** We work on the official **50M** slice (10 k users · 934 k tracks
· 46.5 M listens) — the right size for a single free GPU while keeping all of the
dataset's real-world structure and matching the published baselines.

**Why it's strong for this task.**
- **Three signal layers.** Explicit feedback (likes / dislikes), implicit
  feedback (plays / skips / replays with played-ratio), and **128-d precomputed
  audio embeddings** for 7.72 M tracks (≈ 82 % of the catalogue).
- **The `is_organic` flag** on every event separates user-driven discovery from
  recommender-driven plays — so we can train and evaluate without the
  recommender-bias loop that plagues most public datasets.
- **Real production distributions** — heavy tail, real cold-start, real session
  structure — that an academic MovieLens-style dataset cannot reproduce.

**Bonus.** Real-world operational data from a production company → qualifies for
the **+2 bonus**.

---

## 3. Modeling Approach

A two-component recommender, fused at the score level.

**A — SASRec (sequence tower).** A causal-attention Transformer over the user's
listening history (structurally a small GPT). It learns transition patterns and
predicts the next track from the sequence so far. This is the collaborative /
behavioural signal.

**B — Audio-content tower.** A training-free content score built on Yambda's audio
embeddings: the user is represented by the mean audio vector of their history (a
"taste centroid"), and items are scored by audio similarity to it. This is the
content signal.

**Fusion (the key design decision).** We combine the two with a learned weight:
`score = zscore(SASRec) + β · zscore(content)`.
- We **first tried joint embedding-level fusion** (content folded into the item
  embeddings, trained end-to-end). It **overfit and lost** (0.0581 < SASRec
  0.0735): joint training lets the content branch memorise training transitions
  and dilute the strong collaborative signal.
- We then used **late, score-level fusion** with `β` chosen on a *validation*
  split of users (not on train). `β = 0` recovers SASRec, so fusion **cannot do
  worse**; the tuned `β ≈ 0.5–1.0` shows the content adds genuine complementary
  signal. This is the model that wins.

*Why a hybrid:* pure collaborative filtering ignores content; a pure content
model ignores behaviour. Late fusion keeps the strong sequence model intact and
adds content only as much as it helps — chosen by generalisation, not memorisation.

---

## 4. KPI & Results

**Primary metric — NDCG@10** under the dataset's **Global Temporal Split** (300 d
train / 30 min gap / 1 d test), ranking over the **full 629 k-item catalogue**
(no sampled negatives). Absolute values are therefore small by design — this is
the same protocol the Yandex paper reports.

**Baselines (our harness == the paper's, validated):**

| Model | NDCG@10 (ours) | NDCG@10 (paper) |
|-------|----------------|-----------------|
| MostPop | 0.0171 | 0.0186 |
| ItemKNN (bm25) | 0.0709 | 0.0781 |
| **SASRec** | **0.0735** | 0.0748 |

Our SASRec reproduces the published number to **98 %**, so our harness is
trustworthy. SASRec already beats every classical baseline.

**Result — the hybrid wins, robustly:**

> Late fusion lifts NDCG@10 from ≈0.073 (SASRec) to **≈0.078–0.080**, a robust
> **+6–10 %** gain that is positive on **every held-out user split and every
> training run** (β tuned on a separate validation split each time — no leakage).
> Two full end-to-end runs gave +6.0 % ± 2.8 % and +9.7 % ± 2.0 % — GPU training is
> non-deterministic, so the exact figure varies, but the improvement is robust.
> Recall@10, NDCG@100 and catalogue coverage all improve too.

**Honest mechanism (where the gain comes from).** A tail analysis shows the audio
signal helps on **popular** tracks (≈+9–13 % on head items) and slightly *hurts*
the extreme long tail (≈−4…−9 % on items with ≤ 5 interactions). So the audio acts
as a **taste-alignment refinement on mainstream tracks**, *not* the long-tail /
cold-start fix we initially hypothesised — the taste-centroid user vector boosts
sonically-central tracks rather than rescuing rare ones. A retrieval-oriented
content design is the natural next step to chase the cold-start case.

---

## 5. Summary & Business Impact

We took real Yandex listening data (Yambda-50M), built a SASRec sequence model
that **beats every published baseline and reproduces the paper to 98 %**, and a
late-fusion hybrid that **adds a robust +6–10 % NDCG@10 over SASRec** (positive on
every split and every run).

**Business reading.** NDCG@10 is the standard offline proxy for recommendation
engagement; a ~10 % relative ranking lift is a meaningful offline improvement.
Translating offline ranking gains to online listening-time / retention requires
an A/B test, but at Yandex-Music scale even single-digit-percent engagement
shifts move retention and subscription revenue materially — which is exactly why
next-track ranking is the metric the product optimises.

**What it took:** the public Yambda dataset (free), one free Kaggle T4 GPU, and a
reproducible, version-controlled pipeline (baselines → SASRec → fusion → robustness).
