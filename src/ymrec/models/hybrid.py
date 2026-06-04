"""Hybrid SASRec — sequence model with a content (two-tower) item branch.

Effective item embedding fused from a collaborative part and an audio-content
part with a learned fusion weight alpha:

    v_j = e_j + alpha * mask_j * ContentMLP(audio_j)

- e_j: collaborative item embedding (learned, captures behaviour).
- ContentMLP(audio_j): content tower on the frozen 128-d Yambda audio embedding
  (helps long-tail / weakly-observed items).
- mask_j: 0 for items with no audio embedding -> they fall back to pure SASRec.
- alpha: learned scalar; alpha -> 0 recovers SASRec, so its final value tells us
  how much the audio content actually helps.

The user side is the SASRec sequence representation, so score(u, j) = h_u . v_j.
Trained end-to-end with the same BCE + 1-negative objective via `sasrec.fit`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ymrec.data.sequences import SeqData
from ymrec.models.sasrec import SASRec, fit


class HybridSASRec(SASRec):
    def __init__(
        self,
        n_items: int,
        content_emb: np.ndarray,   # (n_items, content_dim), aligned to the item vocab
        content_mask: np.ndarray,  # (n_items,) bool — item has an audio embedding
        maxlen: int = 200,
        d: int = 64,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__(n_items, maxlen, d, n_blocks, n_heads, dropout)
        content_dim = content_emb.shape[1]

        # Frozen audio buffer with a zero pad row at index 0 (model ids are 1..n_items).
        A = torch.zeros(n_items + 1, content_dim, dtype=torch.float32)
        A[1:] = torch.from_numpy(np.asarray(content_emb, dtype=np.float32))
        self.register_buffer("A", A)
        cmask = torch.zeros(n_items + 1, dtype=torch.float32)
        cmask[1:] = torch.from_numpy(np.asarray(content_mask, dtype=np.float32))
        self.register_buffer("cmask", cmask)

        self.content_mlp = nn.Sequential(
            nn.Linear(content_dim, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _content(self, ids: torch.Tensor) -> torch.Tensor:
        return self.cmask[ids].unsqueeze(-1) * self.content_mlp(self.A[ids])

    def _eff_emb(self, ids: torch.Tensor) -> torch.Tensor:
        return self.item_emb(ids) + self.alpha * self._content(ids)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        feats = self.log2feats(log_seqs)
        pos_logits = (feats * self._eff_emb(pos_seqs)).sum(-1)
        neg_logits = (feats * self._eff_emb(neg_seqs)).sum(-1)
        return pos_logits, neg_logits

    @torch.no_grad()
    def score_all(self, log_seqs: torch.Tensor) -> torch.Tensor:
        feats = self.log2feats(log_seqs)[:, -1, :]
        content_all = self.cmask.unsqueeze(-1) * self.content_mlp(self.A)  # (n_items+1, d)
        V = self.item_emb.weight + self.alpha * content_all
        return (feats @ V.t())[:, 1:]


def train_and_eval(
    data: SeqData,
    content_emb: np.ndarray,
    content_mask: np.ndarray,
    d: int = 64,
    n_blocks: int = 2,
    n_heads: int = 1,
    dropout: float = 0.2,
    **fit_kwargs,
) -> tuple[HybridSASRec, dict]:
    model = HybridSASRec(
        data.n_items, content_emb, content_mask, data.maxlen, d, n_blocks, n_heads, dropout
    )
    return fit(model, data, **fit_kwargs)
