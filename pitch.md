# Capstone Project Pitch — Next-Gen Music Recommender on Yambda

**Team:** Anna Grishkina · Valeria Karpova · Aleksey Kosychev
**Target score:** 10 / 10 (real-world operational data from Yandex — qualifies for the +2 bonus)
**Dataset:** [Yambda](https://huggingface.co/datasets/yandex/yambda) — Yandex Music listening events, released May 2025

---

## 1. Problem Statement

**What we are solving.** Given a catalogue of ten million tracks, predict the next track a user wants to listen to. This is the central decision a music-streaming product makes every few minutes for every active user.

**Why it matters now.**
- About 70 % of streaming users say they struggle to discover new music — bad recommendations directly cause churn, and churn is the #1 commercial metric in a saturated streaming market.
- The streaming market itself is saturated: subscriber growth has plateaued and retention now dominates economics. A 15 % relative lift in next-track accuracy is the difference between catching up and leading the market.
- Modern sequence models (Transformers) and learned audio embeddings have matured to the point where a small team can credibly target this gap without industrial-scale engineering.

---

## 2. Data — our key advantage

**Source.** **Yambda** by Yandex, published on Hugging Face in **May 2025** (less than a year old). Not a toy dataset — a production listening log with **4.79 billion real events** from **1 million users** across **9.39 million tracks**.

**What makes it strong for this task.**
- **Three signal layers in one feed.** Explicit feedback (likes / dislikes), implicit feedback (every play, skip, replay), and precomputed audio embeddings — neural representations of how each track actually sounds.
- **The `is_organic` flag.** Every event is marked as "user found the track on their own" or "the recommender surfaced it." This lets us train and evaluate without the recommender-bias loop that plagues most public listening datasets.
- **Real distributions.** Heavy-tail catalogue, real cold-start (millions of tracks with very few interactions), real session structure — all the operational properties an academic MovieLens-style dataset cannot reproduce.

**Bonus justification.** Yambda is real-world operational data from an actual production company, not a sampled / sanitised academic dataset → the project clearly qualifies for the **+2 real-world bonus** under the rubric.

---

## 3. Modeling Approach

**Architecture — a hybrid two-component network.**

**Component A — SASRec** (Self-Attentive Sequential Recommender). A causal-attention Transformer over the user's listening history, structurally close to a small GPT. It learns transition patterns: "rock → pop → jazz-like" and predicts the next track conditional on the sequence so far.

**Component B — Two-Tower content network.** A user-tower / item-tower model that consumes the precomputed Yambda audio embeddings. This branch can score tracks that no one has listened to yet, directly solving the cold-start problem that breaks pure collaborative-filtering systems.

**Fusion.** Outputs of the two components are combined with a learned weighting (per-user-cohort or global) so the model picks the right balance of behavioural signal and content signal automatically — no hand-tuned weights.

**Why this approach for this problem.**
- Pure matrix factorisation loses sequence information.
- Pure collaborative filtering fails on new tracks (cold start).
- A hybrid SASRec + Two-Tower handles both regimes within one trainable system, and reuses building blocks already validated in our earlier MiniGPT and StockGPT homework — so we ship a research-quality baseline without inventing architecture.

---

## 4. KPI & Business Impact

**Primary metric — NDCG@10.** Normalised Discounted Cumulative Gain at top-10. It answers two questions at once: *how many relevant tracks are in the top-10?* and *how well are they ordered?* (a great track at position 1 scores higher than the same track at position 10). NDCG@10 is the canonical ranking-quality metric for recommender systems.

**Targets.**
- Yandex's published BPR baseline reaches NDCG@10 ≈ **0.38**.
- Our goal: **NDCG@10 ≥ 0.44** — a **+15 %** relative lift.

**Expected business impact** (based on published industry studies of NDCG-to-engagement transfer):
- **+18 – 22 %** listening time per active user.
- **−10 %** monthly churn.
- An additional **3 – 5 M ₽ / month** in subscription revenue at the scale of Yandex Music.

---

## 5. Summary & Call to Action

We take real Yandex listening data (Yambda, < 1 year old, 4.79 B events) and build a hybrid neural recommender — SASRec for sequence modelling plus a two-tower content network on audio embeddings — to predict the next track for a streaming user. Our success bar is NDCG@10 ≥ 0.44 (+15 % over the published Yandex baseline), and the resulting business impact is measurable as longer listening sessions, lower churn, and additional subscription revenue.

**What we need to execute.**
- Access to the Yambda dataset on Hugging Face (free, already public).
- A GPU instance with **80 – 160 GB** of memory (Colab Pro+, Yandex Cloud, or similar) for **1 – 2 weeks** of training and ablations.
- Two weeks of project time end-to-end: pipeline + baselines week one, hybrid model + evaluation week two.
