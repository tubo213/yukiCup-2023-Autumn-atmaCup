import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction


def threshold_search(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    best_threshold = 0.0
    best_score = 0.0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {"threshold": best_threshold, "f1": best_score}
    return search_result


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def compute_metrics(p: EvalPrediction):
    preds = sigmoid(p.predictions)
    labels = p.label_ids
    score = threshold_search(labels, preds)["f1"]
    metrics = {"f1_score": score}

    return metrics
