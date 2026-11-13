"""
Score the validation and test edges with label-free baselines and evaluate them against the organic labels of the dataset
(JODIE state labels: user banned on Wikipedia and Reddit, user dropped out on MOOC).

Baselines (protocol section 6), higher score means more anomalous:
    EdgeBank (unlimited memory): 1 if the pair (u, v) was never seen before the edge, else 0
    DegSum: -(deg(u) + deg(v)), degrees counted from the edges before the edge
    Random: uniform random scores, one file per seed
"Before" means at a strictly earlier timestamp: edges with the same timestamp do not see each other.
Runs on the CPU, the labels are only used for the metrics.
"""
import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd

from utils.metrics import get_anomaly_detection_metrics
from utils.DataLoader import get_link_prediction_data, Data


def compute_edgebank_degsum_scores(src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
    """
    compute the EdgeBank (unlimited memory) and Deg-Sum scores of every edge of a time-sorted edge stream, the history of an edge
    is all the edges at strictly earlier timestamps
    :param src_node_ids: ndarray, shape (num_edges, )
    :param dst_node_ids: ndarray, shape (num_edges, )
    :param node_interact_times: ndarray, shape (num_edges, ), sorted in non-decreasing order
    :return:
        edgebank_scores, degsum_scores, ndarrays with shape (num_edges, )
    """
    assert np.all(np.diff(node_interact_times) >= 0), 'edges must be sorted by time'
    num_edges = len(src_node_ids)
    edgebank_scores, degsum_scores = np.zeros(num_edges), np.zeros(num_edges)
    seen_pairs, node_degrees = set(), defaultdict(int)

    # boundaries of the groups of edges with the same timestamp
    group_starts = np.flatnonzero(np.r_[True, node_interact_times[1:] != node_interact_times[:-1]])
    group_ends = np.r_[group_starts[1:], num_edges]
    for group_start, group_end in zip(group_starts, group_ends):
        # score the whole group with the history before its timestamp
        for edge_index in range(group_start, group_end):
            src_node_id, dst_node_id = src_node_ids[edge_index], dst_node_ids[edge_index]
            edgebank_scores[edge_index] = 0.0 if (src_node_id, dst_node_id) in seen_pairs else 1.0
            degsum_scores[edge_index] = -(node_degrees[src_node_id] + node_degrees[dst_node_id])
        # then add the group to the history
        for edge_index in range(group_start, group_end):
            src_node_id, dst_node_id = src_node_ids[edge_index], dst_node_ids[edge_index]
            seen_pairs.add((src_node_id, dst_node_id))
            node_degrees[src_node_id] += 1
            node_degrees[dst_node_id] += 1

    return edgebank_scores, degsum_scores


def select_edges(full_data: Data, full_scores: np.ndarray, data: Data):
    """
    select the scores of the edges of data from the scores of all the edges in full_data, matched by edge id
    """
    edge_positions = np.full(full_data.edge_ids.max() + 1, -1)
    edge_positions[full_data.edge_ids] = np.arange(len(full_data.edge_ids))
    positions = edge_positions[data.edge_ids]
    assert np.all(positions >= 0)
    return full_scores[positions]


def print_metrics(name: str, split: str, metrics: dict):
    for metric_name in ['average_precision', 'roc_auc', 'recall_at_k', 'precision_at_100']:
        print(f"{name}, {split} {metric_name}, {metrics[metric_name]:.4f}")
    print(f"{name}, {split} average_precision 95% interval, [{metrics['average_precision_ci_low']:.4f}, {metrics['average_precision_ci_high']:.4f}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Score the validation and test edges with label-free baselines')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset to be used')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.4, help='ratio of test set')
    parser.add_argument('--num_random_seeds', type=int, default=5, help='number of seeds of the random baseline, seeds 0, 1, ...')
    args = parser.parse_args()
    print(args)

    # the new node hold-out is off as in the model runs, it does not change the validation and test edges
    _, _, full_data, _, val_data, test_data, _, _ = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, new_node_holdout=False)

    train_ratio = 1 - args.val_ratio - args.test_ratio
    split_name = f'{round(100 * train_ratio)}-{round(100 * args.val_ratio)}-{round(100 * args.test_ratio)}'

    edgebank_scores, degsum_scores = compute_edgebank_degsum_scores(full_data.src_node_ids, full_data.dst_node_ids, full_data.node_interact_times)

    # (baseline name, seed, scores of all the edges)
    baseline_runs = [('EdgeBank', 0, edgebank_scores), ('DegSum', 0, degsum_scores)]
    for seed in range(args.num_random_seeds):
        baseline_runs.append(('Random', seed, np.random.RandomState(seed).rand(full_data.num_interactions)))

    random_metrics = {'validate': [], 'test': []}
    for baseline_name, seed, full_scores in baseline_runs:
        score_folder = f"./scores/{args.dataset_name}/{split_name}/{baseline_name}/none"
        os.makedirs(score_folder, exist_ok=True)

        edge_ids, scores, metrics_per_split = [], [], {}
        for split, data in [('validate', val_data), ('test', test_data)]:
            split_scores = select_edges(full_data=full_data, full_scores=full_scores, data=data)
            metrics = get_anomaly_detection_metrics(scores=split_scores, labels=data.labels)
            metrics_per_split[split] = metrics
            if baseline_name == 'Random':
                random_metrics[split].append(metrics)
            edge_ids.append(data.edge_ids)
            scores.append(split_scores)

            print(f"{baseline_name} seed {seed}, {split}: {metrics['num_anomalies']:.0f} anomalies in {metrics['num_edges']:.0f} edges, "
                  f"chance average_precision {metrics['chance_average_precision']:.4f}")
            print_metrics(name=f'{baseline_name} seed {seed}', split=split, metrics=metrics)

        pd.DataFrame({'edge_id': np.concatenate(edge_ids), 'score': np.concatenate(scores)}).to_csv(os.path.join(score_folder, f"seed{seed}.csv"), index=False)
        with open(os.path.join(score_folder, f"seed{seed}_metrics.json"), 'w') as file:
            file.write(json.dumps(metrics_per_split, indent=4))

    print(f'Random, metrics over {args.num_random_seeds} seeds:')
    for split in random_metrics:
        for metric_name in ['average_precision', 'roc_auc', 'recall_at_k', 'precision_at_100']:
            metric_values = [metrics[metric_name] for metrics in random_metrics[split]]
            print(f'average {split} {metric_name}, {np.mean(metric_values):.4f} ± {np.std(metric_values, ddof=1) if len(metric_values) > 1 else 0:.4f}')
