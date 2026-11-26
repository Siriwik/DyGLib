"""
Inject synthetic anomalies into the test period and score them with the label-free baselines (EdgeBank, Deg-Sum, random),
to compare injected and organic anomalies on the same dataset and test edges. No training.

Injected edges are inserted into the edge stream (they become history for later edges, as in an injection benchmark).
Evaluation set: the real test edges (label 0, organic labels ignored) plus the injected edges (label 1).
Injection types, each injected edge built from the test period:
    random: (u', v', t') with u' uniform over source nodes, v' uniform over destination nodes, t' uniform over the test period
    S:      a real test edge (u, v, t, m) with v replaced by a uniform destination node v'
    T:      a real test edge (u, v, t, m) moved to a uniform time t' in the test period
    C:      a real test edge (u, v, t, m) with the message m of another test edge (invisible to EdgeBank and Deg-Sum)
The organic result on the same test edges is computed alongside for the side-by-side table.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

from score_baselines import compute_edgebank_degsum_scores
from utils.metrics import get_anomaly_detection_metrics
from utils.DataLoader import get_link_prediction_data

INJECTION_TYPES = ['random', 'S', 'T', 'C']
METRIC_NAMES = ['average_precision', 'roc_auc', 'recall_at_k', 'precision_at_100']


def inject(src_node_ids, dst_node_ids, times, test_mask, injection_type, num_injected, random_state):
    """
    build num_injected anomalous edges of one type from the test period
    :return: src, dst, times of the injected edges, ndarrays with shape (num_injected, )
    """
    test_indices = np.flatnonzero(test_mask)
    test_start, test_end = times[test_indices].min(), times[test_indices].max()
    all_src, all_dst = np.unique(src_node_ids), np.unique(dst_node_ids)
    base = random_state.choice(test_indices, size=num_injected, replace=True)
    inj_src, inj_dst, inj_times = src_node_ids[base].copy(), dst_node_ids[base].copy(), times[base].copy()
    if injection_type == 'random':
        inj_src = random_state.choice(all_src, size=num_injected)
        inj_dst = random_state.choice(all_dst, size=num_injected)
        inj_times = random_state.uniform(test_start, test_end, size=num_injected)
    elif injection_type == 'S':
        inj_dst = random_state.choice(all_dst, size=num_injected)
    elif injection_type == 'T':
        inj_times = random_state.uniform(test_start, test_end, size=num_injected)
    elif injection_type == 'C':
        pass  # only the message changes, which the baselines do not read
    else:
        raise ValueError(injection_type)
    return inj_src, inj_dst, inj_times


def score_stream(src_node_ids, dst_node_ids, times, is_injected):
    """
    merge real and injected edges in time order (real edges first at equal times, stable), score the merged stream
    :return: edgebank, degsum scores in the input order
    """
    order = np.lexsort((is_injected, times))
    edgebank_sorted, degsum_sorted = compute_edgebank_degsum_scores(src_node_ids[order], dst_node_ids[order], times[order])
    edgebank, degsum = np.empty_like(edgebank_sorted), np.empty_like(degsum_sorted)
    edgebank[order], degsum[order] = edgebank_sorted, degsum_sorted
    return edgebank, degsum


def summarize(runs):
    return {metric_name: (float(np.mean([r[metric_name] for r in runs])),
                          float(np.std([r[metric_name] for r in runs], ddof=1)) if len(runs) > 1 else 0.0) for metric_name in METRIC_NAMES}


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Score injected test-period anomalies with label-free baselines')
    parser.add_argument('--dataset_name', type=str, default='mooc', help='dataset to be used')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.4, help='ratio of test set')
    parser.add_argument('--num_injected', type=int, default=-1, help='number of injected anomalies, -1 for the number of organic test positives')
    parser.add_argument('--num_seeds', type=int, default=5, help='number of injection seeds, seeds 0, 1, ...')
    parser.add_argument('--num_bootstraps', type=int, default=1000, help='bootstrap resamples of the AP interval, seed 0 only')
    args = parser.parse_args()
    print(args)

    _, _, full_data, _, _, test_data, _, _ = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, new_node_holdout=False)

    train_ratio = 1 - args.val_ratio - args.test_ratio
    split_name = f'{round(100 * train_ratio)}-{round(100 * args.val_ratio)}-{round(100 * args.test_ratio)}'
    out_folder = f"./scores/{args.dataset_name}/{split_name}/injected"
    os.makedirs(out_folder, exist_ok=True)

    src, dst, times = full_data.src_node_ids, full_data.dst_node_ids, full_data.node_interact_times
    test_mask = np.isin(full_data.edge_ids, test_data.edge_ids)
    num_injected = int(test_data.labels.sum()) if args.num_injected < 0 else args.num_injected
    print(f'{test_mask.sum()} real test edges, {int(test_data.labels.sum())} organic positives, injecting {num_injected} per type and seed')

    results = {'num_injected': num_injected, 'organic': {}, 'injected': {}, 'positive_profile': {}}

    # organic: scores on the real stream only
    edgebank, degsum = compute_edgebank_degsum_scores(src, dst, times)
    organic_labels = full_data.labels[test_mask]
    real_degsum = degsum[test_mask]
    for name, scores in [('EdgeBank', edgebank[test_mask]), ('DegSum', degsum[test_mask])]:
        results['organic'][name] = get_anomaly_detection_metrics(scores=scores, labels=organic_labels, num_bootstraps=args.num_bootstraps)
    results['organic']['Random'] = summarize([get_anomaly_detection_metrics(np.random.RandomState(s).rand(test_mask.sum()), organic_labels, num_bootstraps=0)
                                              for s in range(args.num_seeds)])
    pos = organic_labels != 0
    results['positive_profile']['organic'] = {'share_new_pair': float(edgebank[test_mask][pos].mean()),
                                              'median_degree_sum': float(np.median(-real_degsum[pos]))}
    results['positive_profile']['real test edges'] = {'share_new_pair': float(edgebank[test_mask].mean()),
                                                      'median_degree_sum': float(np.median(-real_degsum))}

    for injection_type in INJECTION_TYPES:
        runs = {'EdgeBank': [], 'DegSum': [], 'Random': []}
        profile = []
        for seed in range(args.num_seeds):
            random_state = np.random.RandomState(seed)
            inj_src, inj_dst, inj_times = inject(src, dst, times, test_mask, injection_type, num_injected, random_state)
            all_src, all_dst = np.concatenate([src, inj_src]), np.concatenate([dst, inj_dst])
            all_times = np.concatenate([times, inj_times])
            is_injected = np.r_[np.zeros(len(src), dtype=int), np.ones(num_injected, dtype=int)]
            edgebank, degsum = score_stream(all_src, all_dst, all_times, is_injected)

            eval_mask = np.r_[test_mask, np.ones(num_injected, dtype=bool)]
            labels = is_injected[eval_mask]
            num_boot = args.num_bootstraps if seed == 0 else 0
            runs['EdgeBank'].append(get_anomaly_detection_metrics(edgebank[eval_mask], labels, num_bootstraps=num_boot))
            runs['DegSum'].append(get_anomaly_detection_metrics(degsum[eval_mask], labels, num_bootstraps=num_boot))
            runs['Random'].append(get_anomaly_detection_metrics(random_state.rand(eval_mask.sum()), labels, num_bootstraps=0))
            inj = is_injected == 1
            profile.append((edgebank[inj].mean(), np.median(-degsum[inj])))

            print(f'{injection_type} seed {seed}: ' + ', '.join(f"{n} AP {runs[n][-1]['average_precision']:.4f} AUROC {runs[n][-1]['roc_auc']:.4f}" for n in runs))

        results['injected'][injection_type] = {name: {'per_seed': r, 'mean_std': summarize(r)} for name, r in runs.items()}
        results['positive_profile'][f'injected {injection_type}'] = {'share_new_pair': float(np.mean([p[0] for p in profile])),
                                                                    'median_degree_sum': float(np.mean([p[1] for p in profile]))}

    with open(os.path.join(out_folder, 'metrics.json'), 'w') as file:
        file.write(json.dumps(results, indent=4))

    # side-by-side table
    chance = results['organic']['EdgeBank']['chance_average_precision']
    print(f'\nTest AP / AUROC, organic vs injected ({num_injected} injected, {args.num_seeds} seeds, organic chance AP {chance:.4f})')
    print(f"{'labels':<16}{'EdgeBank AP':>14}{'AUROC':>8}{'DegSum AP':>14}{'AUROC':>8}{'Random AP':>12}")
    o = results['organic']
    print(f"{'organic':<16}{o['EdgeBank']['average_precision']:>14.4f}{o['EdgeBank']['roc_auc']:>8.3f}"
          f"{o['DegSum']['average_precision']:>14.4f}{o['DegSum']['roc_auc']:>8.3f}{o['Random']['average_precision'][0]:>12.4f}")
    for t in INJECTION_TYPES:
        m = {n: results['injected'][t][n]['mean_std'] for n in runs}
        print(f"{'injected ' + t:<16}{m['EdgeBank']['average_precision'][0]:>8.4f}±{m['EdgeBank']['average_precision'][1]:.3f}{m['EdgeBank']['roc_auc'][0]:>8.3f}"
              f"{m['DegSum']['average_precision'][0]:>8.4f}±{m['DegSum']['average_precision'][1]:.3f}{m['DegSum']['roc_auc'][0]:>8.3f}{m['Random']['average_precision'][0]:>12.4f}")
    print('\nPositive profile (share of positives that are a never-seen pair, median degree sum before the edge):')
    for k, v in results['positive_profile'].items():
        print(f"  {k:<18} new pair {v['share_new_pair']:.3f}   median deg-sum {v['median_degree_sum']:.0f}")
