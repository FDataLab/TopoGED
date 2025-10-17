"""
PyTorch implementation of ROLAND

reference: https://github.com/manuel-dileo/dynamic-gnn
"""

import torch
import torch.nn.functional as F
from torch.nn import GRUCell
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear
import torch
from GraphGeneration.models.temporal_gnn.script.config import args


class ROLANDGNN(torch.nn.Module):
    def __init__(self, model_dim,
                 num_nodes, dropout=0.0, update='moving'):

        super(ROLANDGNN, self).__init__()

        self.conv1 = GCNConv(model_dim["input_dim"], model_dim["hidden_conv_1"])
        self.conv2 = GCNConv(model_dim["hidden_conv_1"], model_dim["hidden_conv_2"])

        self.dropout = dropout
        self.update = update
        if update == 'moving':
            self.tau = torch.Tensor([0]).to(args.device)
        elif update == 'learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0]).to(args.device)).to(args.device)
        elif update == 'gru':
            self.gru1 = GRUCell(model_dim["hidden_conv_1"], model_dim["hidden_conv_1"])
            self.gru2 = GRUCell(model_dim["hidden_conv_2"], model_dim["hidden_conv_2"])
        elif update == 'mlp':
            self.mlp1 = Linear(model_dim["hidden_conv_1"] * 2, model_dim["hidden_conv_1"])
            self.mlp2 = Linear(model_dim["hidden_conv_2"] * 2, model_dim["hidden_conv_2"])
        else:
            assert (update >= 0 and update <= 1)
            self.tau = torch.Tensor([update]).to(args.device)
        self.previous_embeddings = [
            torch.Tensor([[0 for i in range(model_dim["hidden_conv_1"])] for j in range(num_nodes)]).to(args.device), \
            torch.Tensor([[0 for i in range(model_dim["hidden_conv_2"])] for j in range(num_nodes)]).to(args.device)]

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, node_id_list, node_id_map, previous_embeddings=None, num_current_edges=None, num_previous_edges=None):

        # You do not need all the parameters to be different to None in test phase
        # You can just use the saved previous embeddings and tau
        if previous_embeddings is not None:  # None if test
            self.previous_embeddings = [previous_embeddings[0].clone(), previous_embeddings[1].clone()]
        if self.update == 'moving' and num_current_edges is not None and num_previous_edges is not None:  # None if test
            # compute moving average parameter
            self.tau = torch.Tensor(
                [num_previous_edges / (num_previous_edges + num_current_edges)]).to(args.device).clone()  # tau -- past weight

        current_embeddings = [torch.Tensor([]).to(args.device), torch.Tensor([]).to(args.device)]

        # GRAPHCONV
        # GraphConv1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Embedding Update after first layer
        if self.update == 'gru':
            h = torch.Tensor(self.gru1(h, self.previous_embeddings[0].clone().to(h.device)).to(args.device).detach())  # .numpy()
        elif self.update == 'mlp':
            hin = torch.cat((h, self.previous_embeddings[0].clone().to(h.device)), dim=1)
            h = torch.Tensor(self.mlp1(hin).detach())  # .numpy()
        else:
            h = torch.Tensor(
                (self.tau * self.previous_embeddings[0].clone() + (1 - self.tau) * h.clone()).detach()).to(args.device)  # .numpy()

        current_embeddings[0] = h.clone()

        # GraphConv2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Embedding Update after second layer
        if self.update == 'gru':
            h = torch.Tensor(self.gru2(h, self.previous_embeddings[1].clone().to(h.device)).detach()).to(args.device)  # .numpy()
        elif self.update == 'mlp':
            hin = torch.cat((h, self.previous_embeddings[1].clone().to(h.device)), dim=1)
            h = torch.Tensor(self.mlp2(hin).detach()).to(args.device)  # .numpy()
        else:
            h = torch.Tensor(
                (self.tau * self.previous_embeddings[1].clone() + (1 - self.tau) * h.clone()).detach()).to(args.device)  # .numpy()
        current_embeddings[1] = h.clone()

        # NOTE: last GCNConv layer is considered as the embeddings
        return {
            node_id: current_embeddings[0][node_id_map[node_id]] for node_id in node_id_list
        }, current_embeddings[0]

