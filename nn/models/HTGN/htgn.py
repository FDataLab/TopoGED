import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform
from nn.layers import HGCNConv, HypGRU, HGATConv
from .manifolds.poincare import PoincareBall
from utils.config import args

class BaseModel(nn.Module):
    def __init__(self, args=None):
        super(BaseModel, self).__init__()
        if args.use_gru:
            self.gru = nn.GRUCell(args.nhid, args.nhid)
        else:
            self.gru = lambda x, h: x

        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nhid)
        self.hidden_initial = torch.ones(args.num_nodes, args.nhid).to(args.device)

        self.model_type = args.model[:3]  # GRU or Dyn
        self.num_window = args.nb_window
        self.Q = Parameter(torch.ones((args.nhid, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.nhid = args.nhid
        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.hidden_initial] * self.num_window
        return self.hiddens

    def weighted_hiddens(self, hidden_window):
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window, self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nhid])
        s = torch.mean(a * hidden_window_new, dim=0)
        return s

    def save_hiddens(self, data_name, path):
        torch.save(self.hiddens, path + '{}_embeddings.pt'.format(data_name))

    def load_hiddens(self, data_name, path):
        self.hiddens = [torch.load(path + '{}_embeddings.pt'.format(data_name))[-1].to(args.device)]
        return self.hiddens[-1]

    def htc(self, x):
        h = self.hiddens[-1]
        return (x - h).pow(2).sum(-1).mean()

    # replace all nodes
    def update_hiddens_all_with(self, z_t):
        self.hiddens.pop(0)  # [element0, element1, element2] remove the first element0
        self.hiddens.append(z_t.clone().detach().requires_grad_(False))  # [element1, element2, z_t]
        return z_t

    # replace current nodes state
    def update_hiddens_with(self, z_t, nodes):
        last_z = self.hiddens[-1].detach_().clone().requires_grad_(False)
        last_z[nodes, :] = z_t[nodes, :].detach_().clone().requires_grad_(False)
        self.hiddens.pop(0)  # [element0, element1, element2] remove the first element0
        self.hiddens.append(last_z)  # [element1, element2, z_t]
        return last_z

    def continuous_encode(self, edge_index, x=None, weight=None):
        x = torch.cat([x, self.hiddens[-1]], dim=1)
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.layer1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, self.dropout2, training=self.training)
        x = self.layer2(x, edge_index)
        return x

    def gru_encode(self, edge_index, x=None, weight=None):
        x = torch.cat([x, self.hiddens[-1]], dim=1)
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.layer1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, self.dropout2, training=self.training)
        x = self.layer2(x, edge_index)
        h = self.weighted_hiddens(torch.cat(self.hiddens, dim=0))
        x = self.gru(x, h)
        return x

    def forward(self, edge_index, x=None, weight=None):
        if x is None:
            x = self.linear(self.feat)
        else:
            x = self.linear(x)
        if self.model_type == 'Dyn':
            x = self.continuous_encode(edge_index, x, weight)
        if self.model_type == 'GRU':
            x = self.gru_encode(edge_index, x, weight)
        return x


class HTGN(BaseModel):
    def __init__(self, args):
        super(HTGN, self).__init__(args)
        self.manifold_name = args.manifold
        self.manifold = PoincareBall()

        self.c = Parameter(torch.ones(3, 1).to(args.device) * args.curvature, requires_grad=not args.fixed_curvature)

        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nout)
        self.hidden_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        self.use_hta = args.use_hta
        if args.aggregation == 'deg':
            self.layer1 = HGCNConv(self.manifold, 2 * args.nout, 2 * args.nhid, self.c[0], self.c[1],
                                   dropout=args.dropout)
            self.layer2 = HGCNConv(self.manifold, 2 * args.nhid, args.nout, self.c[1], self.c[2], dropout=args.dropout)
        if args.aggregation == 'att':
            self.layer1 = HGATConv(self.manifold, 2 * args.nout, 2 * args.nhid, self.c[0], self.c[1],
                                   heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=True)
            self.layer2 = HGATConv(self.manifold, 2 * args.nhid * args.heads, args.nout, self.c[1], self.c[2],
                                   heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=False)
        self.gru = nn.GRUCell(args.nout, args.nout)

        self.nhid = args.nhid
        self.nout = args.nout
        self.cat = True
        self.Q = Parameter(torch.ones((args.nout, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.num_window = args.nb_window
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.initHyperX(self.hidden_initial)] * self.num_window
        return self.hiddens

    def weighted_hiddens(self, hidden_window):
        if self.use_hta == 0:
            return self.manifold.proj_tan0(self.manifold.logmap0(self.hiddens[-1], c=self.c[2]), c=self.c[2])
        # temporal self-attention
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window, self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nout])
        s = torch.mean(a * hidden_window_new, dim=0) # torch.sum is also applicable
        return s

    def initHyperX(self, x, c=1.0):
        if self.manifold_name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x

    def htc(self, x):
        x = self.manifold.proj(x, self.c[2])
        h = self.manifold.proj(self.hiddens[-1], self.c[2])

        return self.manifold.sqdist(x, h, self.c[2]).mean()

    def forward(self, edge_index, x=None, weight=None):
        if x is None:  # using trainable feat matrix
            x = self.initHyperX(self.linear(self.feat), self.c[0])
        else:
            x = self.initHyperX(self.linear(x), self.c[0])
        if self.cat:
            x = torch.cat([x, self.hiddens[-1]], dim=1)

        # layer 1
        x = self.manifold.proj(x, self.c[0])
        x = self.layer1(x, edge_index)

        # layer 2
        x = self.manifold.proj(x, self.c[1])
        x = self.layer2(x, edge_index)

        # GRU layer
        x = self.toTangentX(x, self.c[2])  # to tangent space
        hlist = self.manifold.proj_tan0(
            torch.cat([self.manifold.logmap0(hidden, c=self.c[2]) for hidden in self.hiddens], dim=0), c=self.c[2])
        h = self.weighted_hiddens(hlist)
        x = self.gru(x, h)  # can also utilize HypGRU
        x = self.toHyperX(x, self.c[2])  # to hyper space
        return x



