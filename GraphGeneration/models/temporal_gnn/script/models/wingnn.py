#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv import GraphConv
from torch_scatter import scatter_add
from copy import deepcopy


class GCNLayer(nn.Module):

    def __init__(self, dim_in, dim_out, pos_isn, skip_connection = "affine"):
        super(GCNLayer, self).__init__()
        self.pos_isn = pos_isn
        self.skip_connection = skip_connection
        if self.skip_connection == 'affine':
            self.linear_skip_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
            self.linear_skip_bias = nn.Parameter(torch.ones(size=(dim_out, )))
        elif self.skip_connection == 'identity':
            assert self.dim_in == self.out_channels

        self.linear_msg_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
        self.linear_msg_bias = nn.Parameter(torch.ones(size=(dim_out, )))

        self.activate = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight, gain=gain)
        nn.init.xavier_normal_(self.linear_msg_weight, gain=gain)

        nn.init.constant_(self.linear_skip_bias, 0)
        nn.init.constant_(self.linear_msg_bias, 0)

    def norm(self, graph):
        edge_index = graph.edges()
        row = edge_index[0]
        edge_weight = torch.ones((row.size(0),),
                                 device=row.device)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=graph.num_nodes())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt

    def message_fun(self, edges):
        return {'m': edges.src['h'] * 0.1}

    def forward(self, g, feats, fast_weights=None):
        if fast_weights:
            linear_skip_weight = fast_weights[0]
            linear_skip_bias = fast_weights[1]
            linear_msg_weight = fast_weights[2]
            linear_msg_bias = fast_weights[3]

        else:
            linear_skip_weight = self.linear_skip_weight
            linear_skip_bias = self.linear_skip_bias
            linear_msg_weight = self.linear_msg_weight
            linear_msg_bias = self.linear_msg_bias



        feat_src, feat_dst = expand_as_pair(feats, g)
        norm_ = self.norm(g)
        feat_src = feat_src * norm_.view(-1, 1)
        g.srcdata['h'] = feat_src
        aggregate_fn = fn.copy_u('h', 'm')

        g.update_all(message_func=aggregate_fn, reduce_func=fn.sum(msg='m', out='h'))
        rst = g.dstdata['h']
        rst = rst * norm_.view(-1, 1)

        rst_ = F.linear(rst, linear_msg_weight, linear_msg_bias)

        if self.skip_connection == 'affine':
            skip_x = F.linear(feats, linear_skip_weight, linear_skip_bias)

        elif self.skip_connection == 'identity':
            skip_x = feats

        return rst_ + skip_x


class WinGNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=256, num_layers=2, dropout=0.16, edge_decoding = 'dot', skip_connection = 'affine'):
        #default hidden_dim= 256; out_features=64
        super(WinGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_decoding = edge_decoding
        self.skip_connection = skip_connection

        for i in range(num_layers):
            d_in = in_features if i == 0 else out_features
            pos_isn = True if i == 0 else False
            layer = GCNLayer(d_in, out_features, pos_isn, self.skip_connection)
            self.add_module('layer{}'.format(i), layer)

        self.weight1 = nn.Parameter(torch.ones(size=(hidden_dim, out_features)))
        self.weight2 = nn.Parameter(torch.ones(size=(1, hidden_dim)))

        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, g, x, fast_weights=None):
        count = 0
        for layer in self.children():
            if fast_weights != None:
                x = layer(g, x, fast_weights[2 + count * 4: 2 + (count + 1) * 4])
            else:
                x = layer(g, x, None)

            count += 1

        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2

        x = F.normalize(x)
        g.node_embedding = x

        pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(pred, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = F.sigmoid(F.linear(pred, weight2))

        # node_feat = pred[g.edge_label_index]
        # nodes_first = node_feat[0]
        # nodes_second = node_feat[1]

        # pred = self.decode_module(nodes_first, nodes_second)

        return x
    


if __name__ == "__main__":
    model = WinGNN(10,20,10,4,0.5)