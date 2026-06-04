"""Unit tests for the ranking metrics — pure math, no data needed."""
import math

import numpy as np

from ymrec.eval.metrics import evaluate_ranking


def test_perfect_ranking():
    # relevant item is at rank 1 -> NDCG = Recall = 1.
    topk = np.array([[5, 3, 1, 9, 2, 7, 8, 4, 6, 0]])
    res = evaluate_ranking(topk, [{5}], n_items=10, ks=(10,))
    assert math.isclose(res["ndcg@10"], 1.0)
    assert math.isclose(res["recall@10"], 1.0)


def test_relevant_at_rank_2():
    # relevant item at rank 2 -> DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    topk = np.array([[3, 5, 1, 9, 2, 7, 8, 4, 6, 0]])
    res = evaluate_ranking(topk, [{5}], n_items=10, ks=(10,))
    assert math.isclose(res["ndcg@10"], 1.0 / math.log2(3), rel_tol=1e-9)
    assert math.isclose(res["recall@10"], 1.0)


def test_two_relevant_partial_hit():
    # relevant = {5, 11}; only 5 retrieved (rank 1). 11 is outside the catalogue top-k.
    topk = np.array([[5, 3, 1, 9, 2, 7, 8, 4, 6, 0]])
    res = evaluate_ranking(topk, [{5, 11}], n_items=20, ks=(10,))
    # IDCG with R=2: 1/log2(2) + 1/log2(3); DCG: 1/log2(2).
    idcg = 1.0 + 1.0 / math.log2(3)
    assert math.isclose(res["ndcg@10"], 1.0 / idcg, rel_tol=1e-9)
    assert math.isclose(res["recall@10"], 0.5)  # 1 of 2 relevant retrieved


def test_coverage():
    # two users, union of top-1 lists covers 2 of 10 items.
    topk = np.array([[0, 1, 2], [1, 3, 4]])
    res = evaluate_ranking(topk, [{0}, {1}], n_items=10, ks=(1,))
    assert math.isclose(res["coverage@1"], 2 / 10)


def test_empty_relevant_skipped():
    topk = np.array([[0, 1, 2], [3, 4, 5]])
    res = evaluate_ranking(topk, [{0}, set()], n_items=10, ks=(1,))
    # only the first user counts toward NDCG (perfect); coverage still counts both rows.
    assert math.isclose(res["ndcg@1"], 1.0)
    assert math.isclose(res["coverage@1"], 2 / 10)
