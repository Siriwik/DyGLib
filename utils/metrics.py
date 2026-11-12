import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}


def get_anomaly_detection_metrics(scores: np.ndarray, labels: np.ndarray, num_bootstraps: int = 1000, seed: int = 0):
    """
    get metrics for the edge anomaly detection task, computed over all the edges at once (not per batch)
    :param scores: ndarray, shape (num_edges, ), anomaly scores, higher means more anomalous
    :param labels: ndarray, shape (num_edges, ), 1 for anomalous edges and 0 for normal edges
    :param num_bootstraps: int, number of resamples of the edges for the 95% interval of average precision, 0 to skip
    :param seed: int, random seed of the bootstrap
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    labels = (labels != 0).astype(int)
    num_anomalies = int(labels.sum())
    # edges sorted from the most to the least anomalous
    ranked_labels = labels[np.argsort(-scores, kind='stable')]

    metrics = {'num_edges': len(labels),
               'num_anomalies': num_anomalies,
               'chance_average_precision': num_anomalies / len(labels),
               'average_precision': average_precision_score(y_true=labels, y_score=scores),
               'roc_auc': roc_auc_score(y_true=labels, y_score=scores),
               # recall among the top-K edges with K = number of anomalies, which is equal to precision at K
               'recall_at_k': ranked_labels[:num_anomalies].mean(),
               'precision_at_100': ranked_labels[:100].mean()}

    if num_bootstraps > 0:
        random_state = np.random.RandomState(seed)
        bootstrap_average_precisions = []
        for _ in range(num_bootstraps):
            indices = random_state.randint(0, len(labels), size=len(labels))
            # a resample without anomalies has no average precision
            if labels[indices].sum() > 0:
                bootstrap_average_precisions.append(average_precision_score(y_true=labels[indices], y_score=scores[indices]))
        metrics['average_precision_ci_low'], metrics['average_precision_ci_high'] = np.percentile(bootstrap_average_precisions, [2.5, 97.5])

    return {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}
