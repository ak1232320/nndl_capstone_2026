"""Late (score-level) fusion of a trained SASRec with a frozen audio-content tower.

Why late fusion: joint embedding-fusion overfit (content memorised train
transitions and diluted the collaborative signal -> 0.0581 < SASRec 0.0735).
Here SASRec stays frozen and strong, and the content contribution is weighted by
a single beta chosen on a VALIDATION split of users (not on train), so it cannot
hurt: beta=0 recovers SASRec. This is the pitch's "outputs combined with learned
weights".

Content tower (training-free, so it cannot overfit):
    user vector  = mean audio embedding over the user's history (warm items)
    item vector  = the item's audio embedding
    content_score(u, j) = user_vec . audio_j

Fusion (scores standardised per user so the scales match):
    score(u, j) = zscore_j(sasrec) + beta * zscore_j(content)
"""
from __future__ import annotations

import numpy as np
import torch

from ymrec.config import TOPK
from ymrec.data.sequences import SeqData
from ymrec.eval.metrics import evaluate_ranking

DEFAULT_BETAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


def content_user_vectors(data: SeqData, content_emb: np.ndarray) -> np.ndarray:
    """Mean audio embedding of each eval user's history (warm items only)."""
    dim = content_emb.shape[1]
    out = np.zeros((len(data.eval_pos), dim), dtype=np.float32)
    for r, up in enumerate(data.eval_pos):
        hist = data.seqs[up]                 # model ids 1..n_items
        vecs = content_emb[hist - 1]         # (h, dim); zero rows for cold items
        warm = np.abs(vecs).sum(1) > 0
        cnt = int(warm.sum())
        if cnt:
            out[r] = vecs.sum(0) / cnt
    return out


def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)


@torch.no_grad()
def fused_recs(
    sasrec_model,
    content_t: torch.Tensor,        # (n_items, dim) on device
    user_content: np.ndarray,       # (n_eval, dim), aligned to data.eval_pos
    data: SeqData,
    beta: float,
    subset: np.ndarray,             # indices into data.eval_pos
    device: str,
    K: int,
    chunk: int = 128,
) -> np.ndarray:
    """Top-K recommendations (original item ids) for `subset` at fusion weight beta."""
    sasrec_model.eval()
    L = sasrec_model.maxlen
    recs = np.empty((len(subset), K), dtype=np.int64)
    for start in range(0, len(subset), chunk):
        idxs = subset[start : start + chunk]
        evps = data.eval_pos[idxs]
        seqs = np.zeros((len(idxs), L), dtype=np.int64)
        for r, up in enumerate(evps):
            s = data.seqs[up][-L:]
            seqs[r, L - len(s):] = s
        seqs_t = torch.from_numpy(seqs).to(device)
        uc = torch.from_numpy(user_content[idxs]).to(device)  # (B, dim)
        sas = sasrec_model.score_all(seqs_t)                  # (B, n_items)
        con = uc @ content_t.t()                              # (B, n_items)
        fused = _zscore(sas) + beta * _zscore(con)
        top = torch.topk(fused, K, dim=1).indices.cpu().numpy()
        recs[start : start + len(idxs)] = data.item_ids[top]
    return recs


def fused_evaluate(sasrec_model, content_t, user_content, data, beta, subset, device,
                   ks=TOPK, chunk=128) -> dict:
    recs = fused_recs(sasrec_model, content_t, user_content, data, beta, subset, device,
                      K=max(ks), chunk=chunk)
    relevant = [data.relevant[i] for i in subset]
    return evaluate_ranking(recs, relevant, n_items=data.n_items, ks=ks)


def _prep(content_emb, data, device):
    user_content = content_user_vectors(data, content_emb)
    content_t = torch.from_numpy(np.asarray(content_emb, dtype=np.float32)).to(device)
    return user_content, content_t


def tune_fusion(sasrec_model, content_emb, data, betas=DEFAULT_BETAS, val_frac=0.5,
                device=None, ks=TOPK, seed=0) -> dict:
    """Pick beta on a validation split of users, then report on the test split.

    Returns best_beta, test metrics at that beta, and the SASRec-only (beta=0)
    test metrics on the SAME users for a clean comparison.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    user_content, content_t = _prep(content_emb, data, device)

    n = len(data.eval_pos)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_frac)
    val, test = perm[:n_val], perm[n_val:]

    curve = []
    for b in betas:
        m = fused_evaluate(sasrec_model, content_t, user_content, data, b, val, device, ks)
        curve.append((float(b), m["ndcg@10"]))
        print(f"  beta={b:<5} val NDCG@10={m['ndcg@10']:.4f}")
    best_beta = max(curve, key=lambda x: x[1])[0]

    test_fused = fused_evaluate(sasrec_model, content_t, user_content, data, best_beta, test, device, ks)
    test_sasrec = fused_evaluate(sasrec_model, content_t, user_content, data, 0.0, test, device, ks)
    return {"best_beta": best_beta, "test_fused": test_fused,
            "test_sasrec_only": test_sasrec, "val_curve": curve}


def robustness(sasrec_model, content_emb, data, seeds=(0, 1, 2, 3, 4),
               betas=DEFAULT_BETAS, val_frac=0.5, device=None, ks=TOPK) -> dict:
    """Repeat tune-on-val / report-on-test over several user splits.

    Reuses the one trained SASRec; only the val/test split changes per seed.
    Returns per-seed rows and mean +/- std of the fused/SASRec NDCG@10 and lift.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    user_content, content_t = _prep(content_emb, data, device)
    n = len(data.eval_pos)

    rows = []
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(n)
        n_val = int(n * val_frac)
        val, test = perm[:n_val], perm[n_val:]
        curve = [(float(b), fused_evaluate(sasrec_model, content_t, user_content, data, b, val, device, ks)["ndcg@10"])
                 for b in betas]
        best_beta = max(curve, key=lambda x: x[1])[0]
        fu = fused_evaluate(sasrec_model, content_t, user_content, data, best_beta, test, device, ks)["ndcg@10"]
        sa = fused_evaluate(sasrec_model, content_t, user_content, data, 0.0, test, device, ks)["ndcg@10"]
        lift = 100.0 * (fu - sa) / sa
        rows.append({"seed": seed, "best_beta": best_beta, "sasrec": sa, "fused": fu, "lift_%": lift})
        print(f"seed {seed}: beta={best_beta:<5} SASRec={sa:.4f} Fused={fu:.4f} lift={lift:+.1f}%")

    fused = np.array([r["fused"] for r in rows])
    sas = np.array([r["sasrec"] for r in rows])
    lifts = np.array([r["lift_%"] for r in rows])
    summary = {
        "sasrec_mean": float(sas.mean()), "sasrec_std": float(sas.std()),
        "fused_mean": float(fused.mean()), "fused_std": float(fused.std()),
        "lift_mean_%": float(lifts.mean()), "lift_std_%": float(lifts.std()),
        "betas": [r["best_beta"] for r in rows],
    }
    return {"rows": rows, "summary": summary}


def _item_train_counts(data: SeqData) -> np.ndarray:
    """pop[m] = number of train Listen+ events for model item m (1..n_items)."""
    pop = np.zeros(data.n_items + 1, dtype=np.int64)
    for s in data.seqs:
        np.add.at(pop, s, 1)
    return pop


def tail_analysis(sasrec_model, content_emb, data, beta, device=None,
                  thresholds=(5, 20, 100), ks=(10,)) -> list[dict]:
    """Where does content help? Split each user's relevant items into popularity
    slices and compare fused vs SASRec NDCG@10 on each.

    Recommendations are full-catalogue; we just credit hits on the sliced
    relevant items (tail = <= threshold train interactions; head = above).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    user_content, content_t = _prep(content_emb, data, device)
    allsub = np.arange(len(data.eval_pos))
    K = max(ks)
    recs_fused = fused_recs(sasrec_model, content_t, user_content, data, beta, allsub, device, K)
    recs_sas = fused_recs(sasrec_model, content_t, user_content, data, 0.0, allsub, device, K)

    pop = _item_train_counts(data)
    id2count = {int(o): int(pop[i + 1]) for i, o in enumerate(data.item_ids)}

    rows = []
    for thr in thresholds:
        for name, keep_tail in [(f"tail<= {thr}", True), (f"head> {thr}", False)]:
            rel = []
            for rset in data.relevant:
                if keep_tail:
                    rel.append({o for o in rset if id2count.get(int(o), 0) <= thr})
                else:
                    rel.append({o for o in rset if id2count.get(int(o), 0) > thr})
            n_users = sum(1 for r in rel if r)
            sa = evaluate_ranking(recs_sas, rel, data.n_items, ks)["ndcg@10"]
            fu = evaluate_ranking(recs_fused, rel, data.n_items, ks)["ndcg@10"]
            lift = 100.0 * (fu - sa) / sa if sa > 0 else 0.0
            rows.append({"slice": name, "users": n_users,
                         "sasrec_ndcg@10": round(sa, 4), "fused_ndcg@10": round(fu, 4),
                         "lift_%": round(lift, 1)})
            print(f"{name:12s} users={n_users:<5} SASRec={sa:.4f} Fused={fu:.4f} lift={lift:+.1f}%")
    return rows
