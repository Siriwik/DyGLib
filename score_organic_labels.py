"""
Score the validation and test edges of trained link prediction models against the organic labels of the dataset
(JODIE state labels: user banned on Wikipedia and Reddit, user dropped out on MOOC).

The anomaly score of an edge is 1 - p, where p is the predicted probability that the edge exists. No negatives are sampled
and the labels are not used by the model. Takes the same arguments as train_link_prediction.py and reads its checkpoints.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_neighbor_sampler
from utils.metrics import get_anomaly_detection_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, Data
from utils.load_configs import get_link_prediction_args

# the only training signal so far: random negatives
TRAINING_SIGNAL = 'rand'


def compute_edge_logits(model_name: str, model: nn.Module, data: Data, batch_size: int, num_neighbors: int = 20, time_gap: int = 2000):
    """
    compute the logit of every (real) edge in data, processed in time order. Memory-based models update their memories with the edges
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param data: Data, edges to be scored
    :param batch_size: int, batch size
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
        ndarray, shape (data.num_interactions, )
    """
    idx_data_loader = get_idx_data_loader(indices_list=list(range(len(data.src_node_ids))), batch_size=batch_size, shuffle=False)

    logits = []
    with torch.no_grad():
        for data_indices in tqdm(idx_data_loader, ncols=120):
            data_indices = data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                data.src_node_ids[data_indices], data.dst_node_ids[data_indices], data.node_interact_times[data_indices], data.edge_ids[data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            logits.append(model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).double().cpu().numpy())

    return np.concatenate(logits)


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=True)

    # get data, the new node data is not used
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _ = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # validation and test neighbor sampler uses all the graph information
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    train_ratio = 1 - args.val_ratio - args.test_ratio
    split_name = f'{round(100 * train_ratio)}-{round(100 * args.val_ratio)}-{round(100 * args.test_ratio)}'
    score_folder = f"./scores/{args.dataset_name}/{split_name}/{args.model_name}/{TRAINING_SIGNAL}"
    os.makedirs(score_folder, exist_ok=True)

    metric_all_runs = {'validate': [], 'test': []}

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)

        # load the best model saved by train_link_prediction.py
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}"
        model.load_state_dict(torch.load(os.path.join(save_model_folder, f"{args.save_model_name}.pkl"), map_location='cpu'))
        model = convert_to_gpu(model, device=args.device)
        model.eval()

        compute_logits_kwargs = dict(model_name=args.model_name, model=model, batch_size=args.batch_size, num_neighbors=args.num_neighbors, time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the saved best model has seen the training and validation edges, so the test edges are scored first
            model[0].memory_bank.node_raw_messages = torch.load(os.path.join(save_model_folder, f"{args.save_model_name}_nonparametric_data.pkl"),
                                                                map_location=args.device)
            test_logits = compute_edge_logits(data=test_data, **compute_logits_kwargs)
            # the memory after training is not saved, rebuild it by replaying the training edges with the saved best model
            model[0].memory_bank.__init_memory_bank__()
            compute_edge_logits(data=train_data, **compute_logits_kwargs)
            val_logits = compute_edge_logits(data=val_data, **compute_logits_kwargs)
        else:
            val_logits = compute_edge_logits(data=val_data, **compute_logits_kwargs)
            test_logits = compute_edge_logits(data=test_data, **compute_logits_kwargs)

        edge_ids, scores = [], []
        for split, data, logits in [('validate', val_data, val_logits), ('test', test_data, test_logits)]:
            # anomaly score 1 - p = 1 - sigmoid(logit) = sigmoid(-logit)
            split_scores = 1 / (1 + np.exp(logits))
            metrics = get_anomaly_detection_metrics(scores=split_scores, labels=data.labels)
            metric_all_runs[split].append(metrics)
            edge_ids.append(data.edge_ids)
            scores.append(split_scores)

            print(f"seed {args.seed}, {split}: {metrics['num_anomalies']} anomalies in {metrics['num_edges']} edges, "
                  f"chance average_precision {metrics['chance_average_precision']:.4f}")
            for metric_name in ['average_precision', 'roc_auc', 'recall_at_k', 'precision_at_100']:
                print(f"seed {args.seed}, {split} {metric_name}, {metrics[metric_name]:.4f}")
            print(f"seed {args.seed}, {split} average_precision 95% interval, "
                  f"[{metrics['average_precision_ci_low']:.4f}, {metrics['average_precision_ci_high']:.4f}]")

        # save the scores of the validation and test edges, and the metrics
        pd.DataFrame({'edge_id': np.concatenate(edge_ids), 'score': np.concatenate(scores)}).to_csv(os.path.join(score_folder, f"seed{args.seed}.csv"), index=False)
        with open(os.path.join(score_folder, f"seed{args.seed}_metrics.json"), 'w') as file:
            file.write(json.dumps({split: metric_all_runs[split][-1] for split in metric_all_runs}, indent=4))

    # store the average metrics at the log of the last run
    print(f'metrics over {args.num_runs} runs:')
    for split in metric_all_runs:
        for metric_name in ['average_precision', 'roc_auc', 'recall_at_k', 'precision_at_100']:
            metric_values = [metrics[metric_name] for metrics in metric_all_runs[split]]
            print(f'average {split} {metric_name}, {np.mean(metric_values):.4f} ± {np.std(metric_values, ddof=1) if len(metric_values) > 1 else 0:.4f}')
