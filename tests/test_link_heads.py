"""
Checks of the link prediction heads, run with: python tests/test_link_heads.py (from the repository root)

A contextual negative C keeps the source, destination and time of a real edge and swaps its message for the message of another edge.
The node embeddings of the two are the same, so only a head that reads the message of the queried edge can tell them apart.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.MemoryModel import MemoryModel
from models.modules import ContextualLinkPredictor, MergeLayer, build_link_predictor, compute_link_logits, get_edge_messages
from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler, set_random_seed

NODE_FEAT_DIM, EDGE_FEAT_DIM = 16, 8


def build_toy_tgn():
    """
    a small TGN on a toy stream of 6 edges between 4 nodes, edges 1-5 are history, edge 6 is the queried real edge
    :return: TGN backbone, Data
    """
    #                     edge:  1  2  3  4  5  6
    src_node_ids = np.array([1, 2, 1, 3, 2, 1])
    dst_node_ids = np.array([3, 4, 4, 4, 3, 3])
    times = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    edge_ids = np.arange(1, 7)
    data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=times, edge_ids=edge_ids, labels=np.zeros(6))
    node_raw_features = np.zeros((5, NODE_FEAT_DIM))
    # row 0 is the padding edge, as in the processed datasets
    edge_raw_features = np.vstack([np.zeros((1, EDGE_FEAT_DIM)), np.random.RandomState(0).randn(6, EDGE_FEAT_DIM)])
    backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                           neighbor_sampler=get_neighbor_sampler(data=data, sample_neighbor_strategy='recent'),
                           time_feat_dim=4, model_name='TGN', num_layers=1, num_heads=2, dropout=0.0, device='cpu')
    backbone.eval()
    return backbone, data


def score_real_and_contextual_negative(head: str):
    """
    score edge 6 (u=1, v=3, t=6, its own message) and its contextual negative (same u, v, t, the message of edge 2)
    :return: logit of the real edge, logit of the C negative, both Tensors with shape (1, )
    """
    set_random_seed(0)
    backbone, data = build_toy_tgn()
    link_predictor = build_link_predictor(head=head, node_feat_dim=NODE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM).eval()
    with torch.no_grad():
        # the history edges 1-5 go into the memory, as in the training stream
        backbone.compute_src_dst_node_temporal_embeddings(src_node_ids=data.src_node_ids[:5], dst_node_ids=data.dst_node_ids[:5],
                                                          node_interact_times=data.node_interact_times[:5], edge_ids=data.edge_ids[:5],
                                                          edges_are_positive=True, num_neighbors=2)
        query = dict(src_node_ids=data.src_node_ids[5:], dst_node_ids=data.dst_node_ids[5:], node_interact_times=data.node_interact_times[5:],
                     edge_ids=None, edges_are_positive=False, num_neighbors=2)
        # the negative is scored first and does not touch the memory, as for the negatives in train_link_prediction.py
        neg_src_embeddings, neg_dst_embeddings = backbone.compute_src_dst_node_temporal_embeddings(**query)
        src_embeddings, dst_embeddings = backbone.compute_src_dst_node_temporal_embeddings(**query)
        assert torch.equal(src_embeddings, neg_src_embeddings) and torch.equal(dst_embeddings, neg_dst_embeddings)

        real_messages = get_edge_messages(backbone.edge_raw_features, data.edge_ids[5:])
        swapped_messages = get_edge_messages(backbone.edge_raw_features, np.array([2]))
        assert not torch.equal(real_messages, swapped_messages)
        real_logit = compute_link_logits(link_predictor, src_embeddings, dst_embeddings, real_messages)
        negative_logit = compute_link_logits(link_predictor, neg_src_embeddings, neg_dst_embeddings, swapped_messages)
    return real_logit, negative_logit


def test_contextual_head_separates_contextual_negative():
    real_logit, negative_logit = score_real_and_contextual_negative(head='contextual')
    assert real_logit.shape == negative_logit.shape == (1, )
    assert not torch.isclose(real_logit, negative_logit).item(), (real_logit, negative_logit)


def test_standard_head_cannot_separate_contextual_negative():
    real_logit, negative_logit = score_real_and_contextual_negative(head='standard')
    assert torch.equal(real_logit, negative_logit), (real_logit, negative_logit)


def test_heads():
    assert isinstance(build_link_predictor('standard', NODE_FEAT_DIM, EDGE_FEAT_DIM), MergeLayer)
    contextual = build_link_predictor('contextual', NODE_FEAT_DIM, EDGE_FEAT_DIM)
    assert isinstance(contextual, ContextualLinkPredictor)
    # the message is projected to the node embedding dimension and concatenated with the two node embeddings
    assert contextual.fc_down.in_features == EDGE_FEAT_DIM and contextual.fc_down.out_features == NODE_FEAT_DIM
    assert contextual.fc1.in_features == 3 * NODE_FEAT_DIM
    # the standard head keeps the parameter names of the pilot's checkpoints
    assert set(build_link_predictor('standard', NODE_FEAT_DIM, EDGE_FEAT_DIM).state_dict()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}


if __name__ == '__main__':
    for test in [test_heads, test_contextual_head_separates_contextual_negative, test_standard_head_cannot_separate_contextual_negative]:
        test()
        print(f'{test.__name__} passed')
