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
def fused_evaluate(
    sasrec_model,
    content_t: torch.Tensor,        # (n_items, dim) on device
    user_content: np.ndarray,       # (n_eval, dim), aligned to data.eval_pos
    data: SeqData,
    beta: float,
    subset: np.ndarray,             # indices into data.eval_pos
    device: str,
    ks=TOPK,
    chunk: int = 128,
) -> dict:
    sasrec_model.eval()
    K = max(ks)
    L = sasrec_model.maxlen
    recs = np.empty((len(subset), K), dtype=np.int64)
    relevant = [data.relevant[i] for i in subset]
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
    return evaluate_ranking(recs, relevant, n_items=data.n_items, ks=ks)


def tune_fusion(
    sasrec_model,
    content_emb: np.ndarray,
    data: SeqData,
    betas=DEFAULT_BETAS,
    val_frac: float = 0.5,
    device: str | None = None,
    ks=TOPK,
    seed: int = 0,
) -> dict:
    """Pick beta on a validation split of eval users, then report on the test split.

    Returns best_beta, test metrics at that beta, and the SASRec-only (beta=0)
    test metrics on the SAME users for a clean comparison.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    user_content = content_user_vectors(data, content_emb)
    content_t = torch.from_numpy(np.asarray(content_emb, dtype=np.float32)).to(device)

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
    return {
        "best_beta": best_beta,
        "test_fused": test_fused,
        "test_sasrec_only": test_sasrec,
        "val_curve": curve,
    }
