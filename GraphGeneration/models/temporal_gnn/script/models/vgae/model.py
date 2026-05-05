"""
Taken from https://github.com/DaehanKim/vgae_pytorch/blob/master/model.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class VGAE(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim):
        """
        VGAE implementation using GCN encoder and inner product decoder[cite: 11, 18].
        """
        super(VGAE, self).__init__()
        self.adj = adj
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        # Inference model (Encoder) [cite: 18, 19]
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        """
        Maps node features and adjacency to latent variables Z[cite: 17, 19].
        """
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        
        # Reparameterization trick [cite: 27]
        # We sample z = mu + std * epsilon where epsilon ~ N(0, I) [cite: 19]
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim).to(X.device)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        """
        Generative model using an inner product between latent variables[cite: 20].
        """
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        """
        Graph Convolutional layer for sparse adjacency matrices[cite: 11, 19].
        """
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        # Propagation rule: f(X, A) = activation(A * X * W) [cite: 19]
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    """
    Decodes latent variables into an adjacency matrix via inner product[cite: 20, 30].
    """
    # p(A_ij = 1 | z_i, z_j) = sigmoid(z_i^T * z_j) [cite: 20, 21]
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    """
    Xavier/Glorot initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim):
        """
        Non-probabilistic Graph Auto-encoder variant[cite: 29].
        """
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        # GAE calculates embeddings Z directly [cite: 29, 30]
        hidden = self.base_gcn(X)
        z = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred
		

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out