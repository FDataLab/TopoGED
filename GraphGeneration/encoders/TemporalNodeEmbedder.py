import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalNodeEmbedder(nn.Module):
    def __init__(self, embedding_dim=128, device='cpu', initial_capacity=4096, init_scale=0.01, init_type='rand'):
        super().__init__()
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.init_scale = init_scale
        self.init = init_type
        
        # node_id -> row index in weight tensor
        self.node_to_idx = {}
        self.idx_to_node = []

        # weights tensor: [capacity, embedding_dim]
        self.capacity = max(int(initial_capacity), 128)
        w = torch.randn(self.capacity, embedding_dim, device=self.device) * self.init_scale
        self.register_parameter('weight', nn.Parameter(w))

        self.last_seen = {}  # track last seen snapshot index

        self.size = 0  # next free index
        
        
    def _modify_capacity(self, need):
        if need <= self.capacity:
            return 
        new_capacity = max(2 * self.capacity, need)
        new_weight = torch.randn(new_capacity, self.embedding_dim, device=self.device) * self.init_scale
        with torch.no_grad():
            new_weight[:self.size] = self.weight.data[:self.size]
        self._replace_weight(new_weight)


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
    
    
    def get_embeddings(self, nodes, snapshot_num):
        idxs = []
        missing = []
        for id in nodes:
            if id in self.node_to_idx:
                idxs.append(self.node_to_idx[id])
                self.last_seen[id] = snapshot_num
            else:
                missing.append(id)

        if missing:
            new_idxs = self.add_nodes(missing, snapshot_num=snapshot_num)
            for id in missing:
                idxs.append(self.node_to_idx[id])

        # final indexing
        emb = self.weight[idxs]
        return {node: emb[i] for i, node in enumerate(nodes)}
    
    
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