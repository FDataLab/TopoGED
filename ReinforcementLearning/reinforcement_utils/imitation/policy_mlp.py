import torch
import torch.nn as nn
import torch.nn.functional as F

class ImitationPolicyMLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    


'''
Check these now

[rbuck@evuser1 Topological_Temporal_GFM]$ squeue -u 'rbuck'
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            439207    normal GraphCon    rbuck PD       0:00      1 (Priority)
            439209    normal GraphCon    rbuck PD       0:00      1 (Priority)
            439210    normal GraphCon    rbuck PD       0:00      1 (Priority)
            439211    normal Temporal    rbuck PD       0:00      1 (Priority)
            439212    normal Temporal    rbuck PD       0:00      1 (Priority)

'''