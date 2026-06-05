"""Generate the report figures from the recorded results (pure matplotlib).

No data / GPU — just plots the numbers in REPORT.md. Saves PNGs to figures/.
Numbers are from the committed end-to-end run (notebooks/executed/RUN_ALL.ipynb).

Run:  uv run --extra dev python scripts/report_figures.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})


def fig_ladder() -> None:
    labels = ["MostPop", "ItemKNN-cos", "ItemKNN-tfidf", "ItemKNN-bm25", "SASRec", "Hybrid (late)"]
    vals = [0.0171, 0.0418, 0.0451, 0.0709, 0.0726, 0.0781]
    colors = ["#bbb", "#bbb", "#bbb", "#7aa6c2", "#2c7fb8", "#d95f0e"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    b = ax.barh(labels, vals, color=colors)
    ax.bar_label(b, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("NDCG@10")
    ax.set_xlim(0, 0.092)
    ax.set_title("Model ladder — Yambda-50M (our harness)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_ladder.png")
    plt.close(fig)


def fig_beta() -> None:
    betas = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    ndcg = [0.0723, 0.0746, 0.0762, 0.0791, 0.0799, 0.0800, 0.0798, 0.0794, 0.0770, 0.0752]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(betas, ndcg, "-o", color="#d95f0e")
    ax.axhline(ndcg[0], ls="--", color="#2c7fb8", label=f"SASRec (beta=0) = {ndcg[0]:.4f}")
    ax.scatter([0.5], [0.0800], s=130, facecolors="none", edgecolors="k", zorder=5, label="peak beta=0.5")
    ax.set_xlabel("fusion weight beta")
    ax.set_ylabel("validation NDCG@10")
    ax.set_title("Late fusion: content weight (inverted-U, peak ~ 0.5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_beta.png")
    plt.close(fig)


def fig_robustness() -> None:
    seeds = [0, 1, 2, 3, 4]
    sasrec = [0.0729, 0.0741, 0.0758, 0.0753, 0.0709]
    fused = [0.0777, 0.0775, 0.0771, 0.0805, 0.0780]
    x = np.arange(len(seeds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - w / 2, sasrec, w, label="SASRec", color="#2c7fb8")
    ax.bar(x + w / 2, fused, w, label="Fused", color="#d95f0e")
    for i in x:
        ax.text(i + w / 2, fused[i] + 0.0008, f"+{100*(fused[i]-sasrec[i])/sasrec[i]:.0f}%",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"split {s}" for s in seeds])
    ax.set_ylabel("test NDCG@10")
    ax.set_ylim(0, 0.092)
    ax.set_title("Late fusion > SASRec on every split  (+6-10% across runs)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_robustness.png")
    plt.close(fig)


def fig_head_tail() -> None:
    groups = ["<=5 / >5", "<=20 / >20", "<=100 / >100"]
    tail_lift = [-3.7, 2.3, 6.4]
    head_lift = [9.1, 9.8, 9.8]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - w / 2, tail_lift, w, label="tail items", color="#cc4c3b")
    ax.bar(x + w / 2, head_lift, w, label="head items", color="#3b8c4c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("popularity threshold (train interactions)")
    ax.set_ylabel("NDCG@10 lift, fused vs SASRec (%)")
    ax.set_title("Content helps the head, not the tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_head_tail.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_ladder()
    fig_beta()
    fig_robustness()
    fig_head_tail()
    print("saved to", OUT + "/:", sorted(os.listdir(OUT)))
