import math
import numpy as np
import torch
from script.config import args


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def xavier_init(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return torch.Tensor(initial)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def prepare(data, t, detection=False):
    if not detection:
        # Edge indices
        edge_index = data['edge_index_list'][t].long().to(args.device)
        pos_index = data['pedges'][t].long().to(args.device)
        neg_index = data['nedges'][t].long().to(args.device)
        new_pos_index = data['new_pedges'][t].long().to(args.device)
        new_neg_index = data['new_nedges'][t].long().to(args.device)

        # Combine all edge types to extract involved node IDs
        all_edges = torch.cat([pos_index, neg_index, new_pos_index, new_neg_index, edge_index], dim=1)
        unique_nodes = torch.unique(all_edges).cpu().numpy()
        node_list = sorted(unique_nodes.tolist())
        node_id_map = {node: idx for idx, node in enumerate(node_list)}

        weights = None
        return edge_index, pos_index, neg_index, node_list, weights, new_pos_index, new_neg_index, node_id_map

    else:
        train_pos_edge_index = data['gdata'][t].train_pos_edge_index.long().to(args.device)
        val_pos_edge_index = data['gdata'][t].val_pos_edge_index.long().to(args.device)
        val_neg_edge_index = data['gdata'][t].val_neg_edge_index.long().to(args.device)
        test_pos_edge_index = data['gdata'][t].test_pos_edge_index.long().to(args.device)
        test_neg_edge_index = data['gdata'][t].test_neg_edge_index.long().to(args.device)

        return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index
