import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.gcn_i = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.gcn_f = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.gcn_o = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.gcn_c = GCNConv(in_channels + hidden_channels, hidden_channels)

    def forward(self, x, edge_index, h, c):
        combined = torch.cat([x, h], dim=1)

        i = torch.sigmoid(self.gcn_i(combined, edge_index))
        f = torch.sigmoid(self.gcn_f(combined, edge_index))
        o = torch.sigmoid(self.gcn_o(combined, edge_index))
        c_tilde = torch.tanh(self.gcn_c(combined, edge_index))

        c = f * c + i * c_tilde
        h = o * torch.tanh(c)

        return h, c

class GCLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super(GCLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            GCLSTMCell(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])

    def forward(self, x_list, edge_index_list, node_id_list, node_id_map):
        '''
        x_list: list of [N, F] node features per time step
        edge_index_list: list of [2, E_t] edge index tensors per time step
        node_id_list: list of node IDs in order of row index (length N)
        
        Returns:
            dict {node_id: final embedding (tensor of shape [H])}
        '''
        T = len(x_list)
        N = x_list[0].size(0)
        H = self.hidden_channels

        h = [torch.zeros(N, H, device=self.device) for _ in range(self.num_layers)]
        c = [torch.zeros(N, H, device=self.device) for _ in range(self.num_layers)]

        for t in range(T):
            x = x_list[t].to(self.device)
            edge_index = edge_index_list[t].to(self.device)
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](x, edge_index, h[l], c[l])
                x = h[l]

        # Final hidden state from last layer: shape [N, H]
        final_hidden = h[-1]

        # Map back to node IDs
        return {
            node_id: final_hidden[node_id_map[node_id]] for node_id in node_id_list
        }
