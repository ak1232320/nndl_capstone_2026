"""SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

A causal-attention Transformer over a user's listening history (structurally a
small GPT). Trained with the original BCE + one-negative-per-position objective;
evaluated by full-catalogue ranking from the final-position representation, in
the same GTS protocol as the baselines (seen items are NOT filtered — music
re-listening is real, as the baselines confirmed).

`torch` is imported here (not in the base package) so the rest of ymrec works
without it; on Kaggle torch is preinstalled.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ymrec.config import TOPK
from ymrec.data.sequences import SeqData
from ymrec.eval.metrics import evaluate_ranking


class _PointWiseFFN(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        maxlen: int = 200,
        d: int = 64,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.maxlen = maxlen
        self.d = d
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)  # 0 = pad
        self.pos_emb = nn.Embedding(maxlen, d)
        self.emb_drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList(nn.LayerNorm(d, eps=1e-8) for _ in range(n_blocks))
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_blocks)
        )
        self.ffn_ln = nn.ModuleList(nn.LayerNorm(d, eps=1e-8) for _ in range(n_blocks))
        self.ffn = nn.ModuleList(_PointWiseFFN(d, dropout) for _ in range(n_blocks))
        self.last_ln = nn.LayerNorm(d, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()  # keep pad row at zero

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """log_seqs: (B, L) long -> (B, L, d) sequence representations."""
        B, L = log_seqs.shape
        x = self.item_emb(log_seqs) * (self.d ** 0.5)
        positions = torch.arange(L, device=log_seqs.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        x = self.emb_drop(x)
        pad = (log_seqs == 0).unsqueeze(-1)  # (B, L, 1)
        x = x.masked_fill(pad, 0.0)
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=log_seqs.device), diagonal=1
        )
        for i in range(len(self.attn)):
            q = self.attn_ln[i](x)
            a, _ = self.attn[i](q, x, x, attn_mask=causal, need_weights=False)
            x = q + a
            x = x + self.ffn[i](self.ffn_ln[i](x))
            x = x.masked_fill(pad, 0.0)
        return self.last_ln(x)

    def forward(
        self, log_seqs: torch.Tensor, pos_seqs: torch.Tensor, neg_seqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.log2feats(log_seqs)
        pos_logits = (feats * self.item_emb(pos_seqs)).sum(-1)
        neg_logits = (feats * self.item_emb(neg_seqs)).sum(-1)
        return pos_logits, neg_logits

    @torch.no_grad()
    def score_all(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, n_items) scores for items 1..n_items (pad column dropped)."""
        feats = self.log2feats(log_seqs)[:, -1, :]
        return (feats @ self.item_emb.weight.t())[:, 1:]


class _TrainDataset(Dataset):
    """Classic SASRec sampler: right-aligned seq, next-item targets, 1 negative."""

    def __init__(self, data: SeqData):
        self.seqs = data.seqs
        self.sets = data.user_item_sets
        self.maxlen = data.maxlen
        self.n_items = data.n_items

    def __len__(self) -> int:
        return len(self.seqs)

    def _neg(self, ts: set[int]) -> int:
        t = np.random.randint(1, self.n_items + 1)
        while t in ts:
            t = np.random.randint(1, self.n_items + 1)
        return t

    def __getitem__(self, i: int):
        items = self.seqs[i]
        L = self.maxlen
        seq = np.zeros(L, dtype=np.int64)
        pos = np.zeros(L, dtype=np.int64)
        neg = np.zeros(L, dtype=np.int64)
        if len(items) >= 2:
            ts = self.sets[i]
            nxt = int(items[-1])
            idx = L - 1
            for it in reversed(items[:-1].tolist()):
                seq[idx] = it
                pos[idx] = nxt
                neg[idx] = self._neg(ts)
                nxt = it
                idx -= 1
                if idx == -1:
                    break
        return seq, pos, neg


@torch.no_grad()
def evaluate(model: SASRec, data: SeqData, device: str, ks=TOPK, chunk: int = 128) -> dict:
    model.eval()
    K = max(ks)
    L = model.maxlen
    recs = np.empty((len(data.eval_pos), K), dtype=np.int64)
    for start in range(0, len(data.eval_pos), chunk):
        batch = data.eval_pos[start : start + chunk]
        seqs = np.zeros((len(batch), L), dtype=np.int64)
        for r, up in enumerate(batch):
            s = data.seqs[up][-L:]
            seqs[r, L - len(s):] = s
        scores = model.score_all(torch.from_numpy(seqs).to(device))
        top = torch.topk(scores, K, dim=1).indices.cpu().numpy()  # 0-based over items 1..n_items
        recs[start : start + len(batch)] = data.item_ids[top]
    return evaluate_ranking(recs, data.relevant, n_items=data.n_items, ks=ks)


def fit(
    model: nn.Module,
    data: SeqData,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    eval_every: int = 10,
    device: str | None = None,
    seed: int = 42,
    ks=TOPK,
) -> tuple[nn.Module, dict]:
    """Train any SASRec-compatible model (forward(seq,pos,neg) + score_all)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    bce = nn.BCEWithLogitsLoss()
    loader = DataLoader(_TrainDataset(data), batch_size=batch_size, shuffle=True)

    best = {"ndcg@10": -1.0}
    for epoch in range(1, epochs + 1):
        model.train()
        last_loss = 0.0
        for seq, pos, neg in loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = model(seq, pos, neg)
            mask = pos != 0
            if not bool(mask.any()):
                continue
            loss = bce(pos_logits[mask], torch.ones_like(pos_logits[mask])) + bce(
                neg_logits[mask], torch.zeros_like(neg_logits[mask])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        if epoch % eval_every == 0 or epoch == epochs:
            m = evaluate(model, data, device, ks=ks)
            if m["ndcg@10"] > best["ndcg@10"]:
                best = {"epoch": epoch, **m}
            extra = f"  alpha={model.alpha.item():.3f}" if hasattr(model, "alpha") else ""
            print(f"epoch {epoch:3d}  loss={last_loss:.4f}  "
                  f"NDCG@10={m['ndcg@10']:.4f}  NDCG@100={m['ndcg@100']:.4f}  "
                  f"Recall@100={m['recall@100']:.4f}{extra}")
    return model, best


def train_and_eval(
    data: SeqData,
    d: int = 64,
    n_blocks: int = 2,
    n_heads: int = 1,
    dropout: float = 0.2,
    **fit_kwargs,
) -> tuple[SASRec, dict]:
    model = SASRec(data.n_items, data.maxlen, d, n_blocks, n_heads, dropout)
    return fit(model, data, **fit_kwargs)
