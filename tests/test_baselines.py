"""
Checks of the label-free baselines and the tie-aware metrics, run with: python tests/test_baselines.py (from the repository root)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_baselines import compute_edgebank_degsum_scores
from utils.metrics import expected_precision_at, get_anomaly_detection_metrics


def test_toy_graph():
    #        edge:  e1  e2  e3  e4  e5  e6
    src = np.array([1, 2, 1, 3, 3, 2])
    dst = np.array([10, 10, 10, 11, 11, 11])
    times = np.array([1, 2, 3, 4, 4, 5], dtype=float)
    edgebank, degsum = compute_edgebank_degsum_scores(src, dst, times)
    # e1 new; e2 new; e3 repeats e1; e4 new; e5 repeats e4 at the same timestamp, so it is not seen before: new; e6 new
    assert edgebank.tolist() == [1, 1, 0, 1, 1, 1], edgebank
    # degrees before each edge:
    # e1: deg1=0 + deg10=0 = 0
    # e2: deg2=0 + deg10=1 (e1) = 1
    # e3: deg1=1 (e1) + deg10=2 (e1, e2) = 3
    # e4: deg3=0 + deg11=0 = 0
    # e5: same timestamp as e4, so deg3=0 + deg11=0 = 0
    # e6: deg2=1 (e2) + deg11=2 (e4, e5) = 3
    assert degsum.tolist() == [0, -1, -3, 0, 0, -3], degsum


def test_expected_precision_ties():
    scores = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    labels = np.array([1, 0, 0, 0, 1, 0])
    # no edge above the 2nd score, 2 slots filled from a tie group of 4 with 1 positive: 2 * 1/4 / 2
    assert np.isclose(expected_precision_at(scores, labels, k=2), 0.25)
    # 4 edges above the 5th score (1 positive), 1 slot from the tie group of 2 with 1 positive: (1 + 1 * 1/2) / 5
    assert np.isclose(expected_precision_at(scores, labels, k=5), 0.3)
    # no ties: plain precision at k
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([1, 0, 1, 0])
    assert np.isclose(expected_precision_at(scores, labels, k=3), 2 / 3)


def test_random_scores():
    random_state = np.random.RandomState(0)
    labels = (random_state.rand(100000) < 0.01).astype(int)
    metrics = get_anomaly_detection_metrics(scores=random_state.rand(len(labels)), labels=labels, num_bootstraps=200)
    assert abs(metrics['roc_auc'] - 0.5) < 0.03, metrics
    assert metrics['average_precision_ci_low'] <= metrics['chance_average_precision'] <= metrics['average_precision_ci_high'], metrics
    # perfect and inverted scores
    assert np.isclose(get_anomaly_detection_metrics(scores=labels.astype(float), labels=labels, num_bootstraps=0)['average_precision'], 1.0)
    assert get_anomaly_detection_metrics(scores=-labels.astype(float), labels=labels, num_bootstraps=0)['recall_at_k'] == 0.0


if __name__ == "__main__":
    for test in [test_toy_graph, test_expected_precision_ties, test_random_scores]:
        test()
        print(f'{test.__name__} passed')
