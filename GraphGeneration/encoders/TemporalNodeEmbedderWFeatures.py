import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class TGCN(torch.nn.Module):
    r"""An implementation of Temporal Graph Convolutional Gated Recurrent Cell.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False. If `improved = True`, the self-loops are added `A+2I` instead of `A+I` giving each node’s own features more influence during aggregation
        cached (bool): Caching the message weights. Default is False. The layer computes the normalized adjacency matrix only once. Speed up training but limit to transductive learning scenario (graph structure is assumed to be static)
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.

    Reference: https://arxiv.org/abs/1811.05320
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_candidate_state_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_update_gate_parameters_and_layers()

    def forward(  # Instead of doing my 2D matrix of embeddings, I will pass features and edges into here to get embeddings
        self,
        X: torch.Tensor,  # Static node feature 
        edge_index: torch.Tensor,  # The edges of the snapshot
        edge_weight: torch.Tensor | None = None,  # Leave None
        H: torch.Tensor | None = None,  # Past result (-1)
    ) -> torch.Tensor:  # Will return node embedding
        f"""Forward pass.

        Args:
            X (PyTorch Tensor): Node features.
            edge_index (PyTorch Long Tensor): Graph edge indices.
            edge_weight (PyTorch Long Tensor, optional): Edge weight vector.
            H (PyTorch Tensor, optional): Hidden state matrix for all nodes.
            lambda_max (PyTorch Tensor, optional but mandatory if normalization is not sym): Largest eigenvalue of Laplacian.

        Returns:
            H (PyTorch Tensor): Hidden state matrix for all nodes.

        Note: If edge weights are not present the forward pass defaults to an unweighted graph.
        """
        edge_index = edge_index.to(torch.int64)
        H = self._set_hidden_state(X, H)

        # Eq.3 (Section 3.3.3)
        U_t = self._calculate_update_gate(X, edge_index, edge_weight, H)

        # Eq.4 (Section 3.3.3)
        R_t = self._calculate_reset_gate(X, edge_index, edge_weight, H)

        # Eq.5 (Section 3.3.3)
        C_t = self._calculate_candidate_state(X, edge_index, edge_weight, H, R_t)

        # Eq.6 (Section 3.3.3)
        H_t = self._calculate_hidden_state(U_t, H, C_t)
        return H_t

    def _create_candidate_state_parameters_and_layers(self) -> None:
        self.conv_c = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_c = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_update_gate_parameters_and_layers(self) -> None:
        self.conv_u = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear_u = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self) -> None:
        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(
        self, X: torch.Tensor, H: torch.Tensor | None
    ) -> torch.Tensor:
        if H is None:
            H = zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
    ) -> torch.Tensor:
        U = torch.cat([self.conv_u(X, edge_index, edge_weight), H], 1)
        U = self.linear_u(U)
        U = torch.sigmoid(U)
        return U

    def _calculate_reset_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
    ) -> torch.Tensor:
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], 1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor | None,
        R: torch.Tensor,
    ) -> torch.Tensor:
        C = torch.cat([self.conv_c(X, edge_index, edge_weight), H * R], 1)
        C = self.linear_c(C)
        C = torch.tanh(C)
        return C

    def _calculate_hidden_state(
        self, U: torch.Tensor, H: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        H = U * H + (1 - U) * C
        return H
    

class TemporalNodeEmbedderWFeatures(nn.Module):
    def __init__(self, embedding_dim=128, num_snapshots=100, device='cpu',
                 initial_capacity=4096, init_scale=0.01, init_type='rand', use_degree=True):
        super().__init__()
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.init_scale = init_scale
        self.init = init_type
        self.use_degree = use_degree
        
        if use_degree:
            self.degree_dim = 1
        else:
            self.degree_dim = 0  
        
        
        # Node embeddings
        self.node_emb_dim = int(0.8 * (embedding_dim - self.degree_dim))
        self.time_emb_dim = embedding_dim - self.node_emb_dim - self.degree_dim

        # Node embeddings
        self.node_to_idx = {}
        self.idx_to_node = []
        self.capacity = max(int(initial_capacity), 55670)
        w = torch.randn(self.capacity, self.node_emb_dim, device=self.device) * self.init_scale
        self.register_parameter('weight', nn.Parameter(w))
        
        # Learnable time embeddings
        self.time_embeddings = nn.Embedding(400, self.time_emb_dim)  # Try to find a time encoding function TGNN model or smth
        nn.init.normal_(self.time_embeddings.weight, mean=0.0, std=0.01)
        
        self.last_seen = {}
        self.size = 0  # next free index

        
    def _modify_capacity(self, need):
        if need <= self.capacity:
            return 
        new_capacity = max(2 * self.capacity, need)
        new_weight = torch.randn(new_capacity, self.node_emb_dim, device=self.device) * self.init_scale
        with torch.no_grad():
            new_weight[:self.size] = self.weight.data[:self.size]
        self._replace_weight(new_weight)


    def _modify_time_embeddings(self, need):
        """Expand time embeddings if snapshot_num exceeds current capacity."""
        current_capacity = self.time_embeddings.num_embeddings
        if need < current_capacity:
            return
        new_capacity = max(2 * current_capacity, need + 1)  # +1 because index is 0-based
        new_weight = torch.randn(new_capacity, self.time_emb_dim, device=self.device) * self.init_scale
        with torch.no_grad():
            new_weight[:current_capacity] = self.time_embeddings.weight.data
        self.time_embeddings = nn.Embedding.from_pretrained(new_weight, freeze=False)


    def _replace_weight(self, new_weight):
        """
        Utility function to replace the tunable weights tensor with a new one.
        """
        del self._parameters['weight']  
        self.register_parameter('weight', nn.Parameter(new_weight))
        
    
    def add_nodes(self, nodes, snapshot_num):
        """ 
        Register new nodes weights
        
        params:
            nodes (list):
            snapshot_num (int): The snapshot number we are using
        """
        assigned = []
        for id in nodes:
            if id in self.node_to_idx:
                # node already exists, skip (never reassign expired IDs)
                assigned.append(self.node_to_idx[id])
                self.last_seen[id] = snapshot_num
                continue

            # allocate new row
            self._modify_capacity(self.size + 1)
            idx = self.size
            self.node_to_idx[id] = idx
            self.idx_to_node.append(id)
            self.size += 1

            # initialize embedding
            with torch.no_grad():
                if self.init == 'rand':
                    self.weight.data[idx].normal_(mean=0.0, std=0.01)
                elif self.init == 'zeros':
                    self.weight.data[idx].zero_()
                elif self.init == 'mean' and self.size > 1:
                    self.weight.data[idx] = self.weight.data[:idx].mean(dim=0)
                else:
                    raise ValueError(f"Unknown self.init mode: {self.init}\nPlease choose from ['rand', 'zero', 'mean']")  # TODO Fill this in
            
            self.last_seen[id] = snapshot_num
            assigned.append(idx)
            
        return assigned
    
    
    def get_embeddings(self, nodes, snapshot_num, node_degrees=None):
        """
        Returns embeddings concatenated with time embedding and optionally node degree,
        all summing to self.embedding_dim
        """
        idxs = []
        missing = []
        for node in nodes:  # TODO Vectorize this
            if node in self.node_to_idx:
                idxs.append(self.node_to_idx[node])
                self.last_seen[node] = snapshot_num
            else:
                missing.append(node)

        if missing:
            new_idxs = self.add_nodes(missing, snapshot_num)
            for node in missing:
                idxs.append(self.node_to_idx[node])

        # node embedding
        node_emb = self.weight[idxs]  # [num_nodes, node_emb_dim]
        # time embedding
        self._modify_time_embeddings(snapshot_num)
        time_emb = self.time_embeddings(torch.tensor(snapshot_num, device=self.device))  # [time_emb_dim]
        time_emb = time_emb.unsqueeze(0).expand(len(nodes), -1)  # [num_nodes, time_emb_dim]

        # concatenate
        if self.use_degree and node_degrees is not None:
            degree_feat = torch.tensor([node_degrees.get(n, 0.0) for n in nodes],
                                       device=self.device, dtype=torch.float32).unsqueeze(-1)  # [num_nodes, 1]
            final_emb = torch.cat([node_emb, time_emb, degree_feat], dim=-1)
        else:
            final_emb = torch.cat([node_emb, time_emb], dim=-1)

        return {node: final_emb[i] for i, node in enumerate(nodes)}
    
    
    def expire_nodes_older_than(self, current_snapshot_num, days_back=5):
        """
        Helps to save memory by expiring nodes that we have not seen in the past 'days_back'
        """
        expired = []
        thresh = current_snapshot_num - days_back
        for id, last in list(self.last_seen.items()):
            if last is None:
                continue
            if last <= thresh:
                expired.append(id)
                idx = self.node_to_idx.get(id)
                if idx is not None:
                    with torch.no_grad():
                        self.weight.data[idx].zero_()  # reset embedding
                        
                # remove all data; node will never reappear
                del self.node_to_idx[id]
                self.last_seen.pop(id, None)
                self.idx_to_node[idx] = None
                
        return expired


    def state_dict_for_save(self):
        return {
            'weight': self.weight.detach().cpu(),
            'node_to_idx': dict(self.node_to_idx),
            'idx_to_node': list(self.idx_to_node),
            'last_seen': dict(self.last_seen),
            'size': self.size,
            'capacity': self.capacity,
            'embedding_dim': self.embedding_dim
        }


    def load_state_dict_from_save(self, sd: dict, device=None):
        device = device or self.device
        w = sd['weight'].to(device)
        self._replace_weight(w)
        self.node_to_idx = dict(sd['node_to_idx'])
        self.idx_to_node = list(sd['idx_to_node'])
        self.last_seen = dict(sd['last_seen'])
        self.size = int(sd['size'])
        self.capacity = int(sd['capacity'])
        self.embedding_dim = int(sd['embedding_dim'])
        self.device = device
        
        
"""
Notes:
- Try a learnable temporal graph embedder
- New nodes can get an average of similar embeddings
- Increase embedding dim
- Add TGCN for embedding (do either random or onehotencoding for node features)
"""