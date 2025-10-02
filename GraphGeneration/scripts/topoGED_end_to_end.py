import math
import numpy as np 
import networkx as nx
import random
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GraphGeneration.utils.Evaluator import Evaluator
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached
from GraphGeneration.utils.casting_type import to_tensor
from GraphGeneration.utils.sampling_edges_utils import predict_edges
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, get_node_features, update_degrees
from create_sub_graphs import create_nn_graph, create_on_graph

# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model

# Import all node embedding methods
from compute_embedding import compute_embedding, load_cached_node2vec_embeddings_from_disk
from process_data import modifyGraphIds, build_edgebanks_from_start
from torch.utils.data import DataLoader

# Import Loss fn
from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss
from GraphGeneration.utils.estimate_graphlet import run_graphlet_estimate   

# Set up device
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
except Exception:
    device = torch.device("cpu")
  

# Load YAML config
with open("GraphGeneration/encoder.yaml", "r") as file:
    encoder_config = yaml.safe_load(file)

# Set seeds
random.seed(encoder_config["seed"])
np.random.seed(encoder_config["seed"]) 

class Runner(object):
    def __init__(self):      
        self.seed = encoder_config["seed"]
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}')
        common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_mlpEncoding{encoder_config["decoder_model"]["encode_links"]}_embeddingType{encoder_config["encoder_model"]["nodeEmbeddingType"]}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{common_suffix}'

        save_dir = os.path.join(self.file_visualization_path, encoder_config["dataset"], encoder_config["encoder_model"]["nodeEmbeddingType"])
        os.makedirs(save_dir, exist_ok=True)
        
        # Current target snapshot we want to predict
        self.current_target_snapshot = 2
        
        # All the edge types
        self.all_edge_types = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        self.best_validation_model_auc = 0
        
        # Load the global encoder & decoder model
        self.encoder_model, self.input_dim = load_encoder_model(encoder_config, device=device, node2vec_dimensions=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_dimensions"], 
                                                           hidden_dim=encoder_config["encoder_model"]["hidden_dim"])
        
        # Check if there is any add-on features we will plug at the end of encoder embedding
        if encoder_config["encoder_model"]["addOnFeature"] in ['NodeType', 'Position']:
            self.input_dim += 1
        
        self.link_prediction_decoder = setupMLP(embedding_dim=self.input_dim*2, mlpEncoding=encoder_config["decoder_model"]["encode_links"])
        # self.link_prediction_decoder =  MLP(in_channels=self.input_dim) 
        self.link_prediction_decoder.to(device)
        
        # Load all the snapshot true data 
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"])
        
        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds)

        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs, days=1e9)        

        # Reshape the graph description
        self.graph_descriptions = [list(zip(graph_description[0::3], graph_description[1::3], graph_description[2::3])) for graph_description in self.graph_descriptions]
        
        # Split training, validation, test graphs
        # Snapshots that we will use for traininng
        # Convert number of snapshots to integer
        self.num_snapshots = len(self.target_graphs)
        self.train_end = int(0.8 * self.num_snapshots)
        self.val_end = int(0.9 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(self.train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(self.train_end, self.val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(self.val_end, self.num_snapshots)]
        
        # Cached the node2vec embeddings of each snapshot
        cache_file_path = os.path.join(self.saved_input, rf"saved_data/node2vec_embeddings_{self.seed}.pkl")
        self.cached_node2vec_embeddings = load_cached_node2vec_embeddings_from_disk(cache_file_path, 
                                                                           self.target_graphs, device)
        
    # ======================= HELPER FUNCTIONS =======================
    def sample_old_nodes(
        self,
        prev_graphs,
        current_target_snapshot,
        *,
        strategy: str = "topk",      # "topk" | "weighted" | "softmax"
        temperature: float = 1.0,    # used only for "softmax"
        return_ranked: bool = False  # optional: also return ranked list for debugging
    ):
        """
        Rank & pick old nodes based on reappearance probabilities.

        Parameters
        ----------
        prev_graphs : your structure (passed through to compute_reappearance_probabilities)
        current_target_snapshot : int
        strategy : 
            - "topk"     -> deterministic top-K by probability
            - "weighted" -> sample without replacement with probabilities ∝ p
            - "softmax"  -> sample without replacement with probabilities ∝ softmax(p / T)
        temperature : float
            Softmax temperature (>0). Lower -> peakier. Used only if strategy="softmax".
        return_ranked : bool
            If True, also return the full ranked list [(node, prob), ...].

        Returns
        -------
        selected : set[int]
        ranked   : list[tuple[int, float]] (optional if return_ranked=True)
        """
        # seeded RNG for reproducibility
        rng = np.random.default_rng(self.seed)

        # 1) get raw reappearance probabilities
        probs = compute_reappearance_probabilities(
            graphs=prev_graphs, 
            t_curr=current_target_snapshot
        )  # expected: dict[node_id] -> float (>=0)

        if not probs:
            return (set(), []) if return_ranked else set()

        node_ids = np.array(list(probs.keys()), dtype=np.int64)
        w        = np.array([float(probs[n]) for n in node_ids], dtype=np.float64)

        # 2) clean probs: clip negatives, handle NaNs/Inf
        w[~np.isfinite(w)] = 0.0
        w = np.clip(w, 0.0, None)

        # 3) if all zeros, fall back to uniform over the observed nodes
        if w.sum() <= 0.0:
            w = np.ones_like(w, dtype=np.float64)

        # 4) compute ranked order (used by topk, and handy to return)
        #    stable deterministic tie-break: (-prob, node_id)
        order = np.lexsort((node_ids, -w))   # lexsort sorts by last key fastest -> use (-w) then node_ids
        ranked_nodes = node_ids[order]
        ranked_probs = w[order]
        ranked_list  = list(zip(ranked_nodes.tolist(), ranked_probs.tolist()))

        K = min(len(node_ids), getattr(self, "current_target_count_old_nodes", len(node_ids)))

        if strategy == "topk":
            chosen = ranked_nodes[:K]

        else:
            # build a sampling distribution
            if strategy == "softmax":
                # temperature-scaled softmax over **probabilities** (not logits)
                t = max(1e-8, float(temperature))
                z = w / t
                z = z - z.max()        # stabilize
                p = np.exp(z)
            elif strategy == "weighted":
                p = w.copy()
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Use 'topk', 'weighted', or 'softmax'.")

            # normalize
            s = p.sum()
            if s <= 0:
                p = np.ones_like(p) / len(p)
            else:
                p = p / s

            # sample without replacement according to p
            # rng.choice supports replace=False with probabilities
            K = min(K, len(node_ids))
            chosen = rng.choice(node_ids, size=K, replace=False, p=p)

        selected = set(map(int, chosen.tolist()))
        return (selected, ranked_list) if return_ranked else selected

    
    # ======================= TRAIN MODEL =======================
    def run_validation(self, validation_samples, batch_size, epoch):
        # For computing AUC Scores
        train_preds = []
        train_labels = []
        dl_num_workers = 0
        use_cuda = (device.type == "cuda")
        epoch_aucs   = {k: [] for k in self.all_edge_types}
        for i in range(self.val_end - self.train_end):
            snapshot = self.train_end + i
            self.encoder_model.eval()
            self.link_prediction_decoder.eval()
            
            current_cached_node2vec = [self.cached_node2vec_embeddings[j] for j in range(max(snapshot - encoder_config["training"]["day"], 0), snapshot)]

            window_graphs = [graph[-1] for graph in self.target_graphs[max(snapshot - encoder_config["training"]["day"], 0):snapshot]]
        

            with torch.no_grad():
                print("INFO: Validation on snapshot", snapshot)
                
                # Prepare current target graph count
                self.current_target_count_old_nodes = self.probabilities[snapshot][0]
                self.current_target_count_new_nodes = self.probabilities[snapshot][1]
                self.current_target_count = {
                    'o-o-bank': self.probabilities[snapshot][2], 
                    'o-o-nobank': self.probabilities[snapshot][5], 
                    'o-n': self.probabilities[snapshot][4], 
                    'n-n': self.probabilities[snapshot][3]
                }

                node_types = { 
                    "old_nodes": self.sample_old_nodes(self.training_graphs[:snapshot], snapshot),
                    "new_nodes": set()
                } 
                
                constructing_graph = nx.DiGraph() # Graph we try to predict
                    
                # Adding old nodes to constructing_graph
                constructing_graph.add_nodes_from(node_types['old_nodes'])
                for flag in self.all_edge_types[1:]:
                    X_np = np.array([
                        (x.cpu().numpy() if torch.is_tensor(x) else x)
                        for x in validation_samples[flag]['X'][i]
                    ])
                    y_np = np.array(validation_samples[flag]['y'][i])
                    if len(X_np) == 0:
                        continue

                    X_np, y_np = shuffle(X_np, y_np, random_state=self.seed)

                    # keep on CPU for DataLoader workers
                    X = torch.tensor(X_np, dtype=torch.float32)
                    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

                    # --------- CONDITIONAL DATALOADER ARGS ----------
                    dl_kwargs = dict(
                        batch_size=len(y),
                        shuffle=True,
                        num_workers=dl_num_workers,
                        pin_memory=use_cuda,
                        drop_last=True,
                    )
                    if dl_num_workers > 0:
                        # only valid when multiprocessing is enabled
                        dl_kwargs.update(
                            persistent_workers=False,  # set True later if stable
                            prefetch_factor=2
                        )
                    loader = DataLoader(TensorDataset(X, y), **dl_kwargs)
                    # ------------------------------------------------

                    flag_logits = []
                    flag_targets = []
                    base_embeddings = None
                    for xb, yb in loader:
                        # move to GPU here (main process)
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)

                        src_nodes = xb[:, 0].long().tolist()
                        dst_nodes = xb[:, 1].long().tolist()

                        base_embeddings = compute_embedding(
                                embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"],
                                graphs=window_graphs,
                                encoder_model=self.encoder_model,
                                device=device,
                                cached_node2vec_embeddings=current_cached_node2vec,
                                snapshots=list(range(max(snapshot - encoder_config["training"]["day"], 0), snapshot))
                            )
                        
                        for n in src_nodes + dst_nodes:
                            n_int = int(n)
                            if n_int not in base_embeddings:
                                node_types["new_nodes"].add(n_int)
                                base_embeddings[n_int] = torch.zeros(self.input_dim, device=device)

                        src_embed = torch.stack([base_embeddings[int(n)] for n in src_nodes])
                        dst_embed = torch.stack([base_embeddings[int(n)] for n in dst_nodes])

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            logits = self.link_prediction_decoder(
                                src_embed=src_embed, dst_embed=dst_embed, edge_type=flag
                            ).view(-1, 1)

                        flag_logits.append(logits.detach().cpu())
                        flag_targets.append(yb.detach().cpu())

                    if len(flag_logits) == 0:
                        # No batches produced (tiny dataset). Avoid cat() on empty list.
                        epoch_aucs[flag].append(0.0)
                    else:
                        L = torch.cat(flag_logits, dim=0).sigmoid().numpy()
                        T = torch.cat(flag_targets, dim=0).numpy()
                        epoch_aucs[flag].append(roc_auc_score(T, L) if len(np.unique(T)) > 1 else 0.0)

                    if base_embeddings is None:
                        break
        # Record the Training Loss, AUC 
        current_model_auc = 0 #we take average of all edge types
        
        for flag in [self.all_edge_types[1]]:
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {flag}  | Validation AUCROC {np.mean(epoch_aucs[flag]):.4f} | GPU: {gpu_mem_alloc:.1f}MiB"
            current_model_auc += np.mean(epoch_aucs[flag])
            print(epochMessage)
            with open(rf'{self.file_visualization_path}\{encoder_config["dataset"]}\{encoder_config["encoder_model"]["nodeEmbeddingType"]}\multiheadMLP_performance_{self.seed}.txt', "a") as f:
                f.write(epochMessage + "\n")
                f.flush()
                
        # We check and cache if it has the best auc
        if current_model_auc/4 >= self.best_validation_model_auc:
            self.best_validation_model_auc = current_model_auc/4
            
            print("INFO: Saving the model...")
            torch.save(self.link_prediction_decoder.state_dict(), self.decoder_model_path)
            torch.save(self.encoder_model.state_dict(), self.encoder_model_path)
            print("INFO: The model is saved. Done.")

    # Same for the above but run faster with autocast and mixed precision and freeze encoder
    def train_multi_head(self, training_samples, validation_samples):
        lr = encoder_config["training"]["lr"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = (device.type == "cuda")

        # choose workers; start safe at 0 on cluster, bump later to 2–4
        dl_num_workers = 0

        self.link_prediction_decoder.train()
        optimizer = torch.optim.Adam(
            list(self.encoder_model.parameters()) +
            list(self.link_prediction_decoder.parameters()),
            lr=lr
        )
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

        for epoch in range(encoder_config["training"]["epochs"]):
            epoch_losses = {k: [] for k in self.all_edge_types}
            epoch_aucs   = {k: [] for k in self.all_edge_types}

            for snapshot in range(2, self.train_end):
                self.current_target_count_old_nodes = self.probabilities[snapshot][0]
                self.current_target_count_new_nodes = self.probabilities[snapshot][1]
                self.current_target_count = {
                    'o-o-bank': self.probabilities[snapshot][2], 
                    'o-o-nobank': self.probabilities[snapshot][5], 
                    'o-n': self.probabilities[snapshot][4], 
                    'n-n': self.probabilities[snapshot][3]
                }
                
                current_cached_node2vec = [self.cached_node2vec_embeddings[i] for i in range(max(snapshot - encoder_config["training"]["day"], 0), snapshot)]

                node_types = {
                    "old_nodes": self.sample_old_nodes(
                        self.training_graphs[max(0, snapshot - encoder_config["training"]["day"]): snapshot],
                        snapshot
                    ),
                    "new_nodes": set()
                }

                window_graphs = self.training_graphs[max(0, snapshot - encoder_config["training"]["day"]): snapshot]

                # PyTorch <=1.9: no device_type kw

                constructing_graph = nx.DiGraph()
                constructing_graph.add_nodes_from(node_types["old_nodes"])

                for flag in self.all_edge_types[1:]:
                    X_np = np.array([
                        (x.cpu().numpy() if torch.is_tensor(x) else x)
                        for x in training_samples[flag]['X'][snapshot]
                    ])
                    y_np = np.array(training_samples[flag]['y'][snapshot])
                    if len(X_np) == 0:
                        continue

                    X_np, y_np = shuffle(X_np, y_np, random_state=self.seed)

                    # keep on CPU for DataLoader workers
                    X = torch.tensor(X_np, dtype=torch.float32)
                    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

                    # --------- CONDITIONAL DATALOADER ARGS ----------
                    dl_kwargs = dict(
                        batch_size=len(y),
                        shuffle=True,
                        num_workers=dl_num_workers,
                        pin_memory=use_cuda,
                        drop_last=True,
                    )
                    if dl_num_workers > 0:
                        # only valid when multiprocessing is enabled
                        dl_kwargs.update(
                            persistent_workers=False,  # set True later if stable
                            prefetch_factor=2
                        )
                    loader = DataLoader(TensorDataset(X, y), **dl_kwargs)
                    # ------------------------------------------------

                    flag_logits = []
                    flag_targets = []
                    base_embeddings = None
                    for xb, yb in loader:
                        optimizer.zero_grad(set_to_none=True)

                        # move to GPU here (main process)
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)

                        src_nodes = xb[:, 0].long().tolist()
                        dst_nodes = xb[:, 1].long().tolist()

                        base_embeddings = compute_embedding(
                                embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"],
                                graphs=window_graphs,
                                encoder_model=self.encoder_model,
                                device=device,
                                cached_node2vec_embeddings=current_cached_node2vec,
                                snapshots=list(range(max(snapshot - encoder_config["training"]["day"], 0), snapshot))
                            )
                        
                        for n in src_nodes + dst_nodes:
                            n_int = int(n)
                            if n_int not in base_embeddings:
                                node_types["new_nodes"].add(n_int)
                                base_embeddings[n_int] = torch.zeros(self.input_dim, device=device)

                        src_embed = torch.stack([base_embeddings[int(n)] for n in src_nodes])
                        dst_embed = torch.stack([base_embeddings[int(n)] for n in dst_nodes])

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            logits = self.link_prediction_decoder(
                                src_embed=src_embed, dst_embed=dst_embed, edge_type=flag
                            ).view(-1, 1)
                            bce = loss_fn(logits, yb)
                            loss = bce

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        flag_logits.append(logits.detach().cpu())
                        flag_targets.append(yb.detach().cpu())
                        epoch_losses[flag].append(loss.item())
                    
                    if len(flag_logits) == 0:
                        # No batches produced (tiny dataset). Avoid cat() on empty list.
                        epoch_aucs[flag].append(0.0)
                        if not epoch_losses[flag]:
                            epoch_losses[flag].append(0.0)   # keep logging sane
                    else:
                        L = torch.cat(flag_logits, dim=0).sigmoid().numpy()
                        T = torch.cat(flag_targets, dim=0).numpy()
                        epoch_aucs[flag].append(roc_auc_score(T, L) if len(np.unique(T)) > 1 else 0.0)

                    if base_embeddings is None:
                        break
                    # curr_embeds = base_embeddings
                    # constructing_graph = get_node_features(
                    #     constructing_graph.copy(), self.training_graphs[:snapshot],
                    #     self.thresholds, self.graph_descriptions[snapshot],
                    #     node_types["old_nodes"], node_types["new_nodes"]
                    # )
                    # sampled_edges = predict_edges(
                    #     constructing_graph, edge_type=flag, node_types=node_types,
                    #     edgebank=self.all_edgebanks[snapshot],
                    #     link_prediction_decoder=self.link_prediction_decoder,
                    #     old_node_embeddings=curr_embeds,
                    #     top_k=self.current_target_count[flag], graph_num=snapshot, device=device
                    # )
                    # constructing_graph.add_edges_from(list(sampled_edges))
                    # update_degrees(constructing_graph)

            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else 0
            for flag in [self.all_edge_types[1]]:
                msg = (
                    f"Epoch: {epoch+1:02d} | Edge Type: {flag} | "
                    f"Train Loss: {np.mean(epoch_losses[flag]):.4f} | "
                    f"Train AUCROC: {np.mean(epoch_aucs[flag]):.4f} | "
                    f"GPU: {gpu_mem_alloc:.1f}MiB"
                )
                print(msg, flush=True)
                with open(rf'{self.file_visualization_path}\{encoder_config["dataset"]}\{encoder_config["encoder_model"]["nodeEmbeddingType"]}\multiheadMLP_performance_{self.seed}.txt', "a") as f:
                    f.write(msg + "\n")
                    f.flush()
            
            # Run validation at the end of each epoch
            self.run_validation(validation_samples, batch_size=len(y), epoch=epoch)

        return self.link_prediction_decoder, self.encoder_model

    def train_models(self):
        """
        Create and train the models used for graph construction, these will be used for later graph construction
        
        Args:
            lr (float): The learning rate to use for the model
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            link_prediction_decoder (MLP NN): The trained MLP, either single or multiheaded
        """
        MAX_SAMPLES = 1000000  # 1 Million

        old_training_nodes = set().union(*[g.nodes() for g in self.training_graphs]) 
        
        # Prepare training data
        training_sorted_samples, training_new_edges_count = generate_training_data_cached(training_graphs=self.training_graphs,
                                                all_edgebanks=self.all_edgebanks, MAX_SAMPLES=MAX_SAMPLES, dataset=encoder_config["dataset"], seed=self.seed, saved_data_file_path=self.saved_input + "/saved_data")

        # Prepare validation data
        # We pass all_edgebanks of the training snapshots edgebanks
        validation_sorted_samples, training_new_edges_count = generate_validation_data_cached(training_graphs=self.validation_graphs, old_training_nodes=old_training_nodes, 
                                                all_edgebanks=self.all_edgebanks[self.train_end], MAX_SAMPLES=MAX_SAMPLES, dataset=encoder_config["dataset"], seed=self.seed, type_data="validation", saved_data_file_path=self.saved_input + "/saved_data")
        # Prepare test data
        # We pass all_edgebanks of the training snapshots edgebanks
        test_sorted_samples, training_new_edges_count = generate_validation_data_cached(training_graphs=self.test_graphs, old_training_nodes=old_training_nodes, 
                                                all_edgebanks=self.all_edgebanks[self.val_end], MAX_SAMPLES=MAX_SAMPLES, dataset=encoder_config["dataset"], seed=self.seed, type_data="test", saved_data_file_path=self.saved_input + "/saved_data")
        
        print('Training') 
    
        self.link_prediction_decoder, self.encoder_model = self.train_multi_head(training_samples=training_sorted_samples, validation_samples=validation_sorted_samples)
        
        return self.link_prediction_decoder, self.encoder_model
            
    # ======================= BUILD GRAPH =======================
    def _norm_edge(self, u, v, undirected=True):
        return (u, v) if (not undirected or u <= v) else (v, u)

    def sample_oobank_edges(self, edgebank, top_k, 
                            mode="topk",         # "topk", "proportional", or "softmax"
                            undirected=True, 
                            temperature=1.0,     # used only for mode="softmax"
                            exclude_edges=None,  # set of edges to exclude (e.g., current positives)
                            rng=None):
        """
        Rank & pick edges from edgebank by reappearance.

        edgebank: dict[u] -> list[v, v, ...]   # v repeated each time (u,v) appeared in past days
        top_k   : how many to return
        mode    : 
            - "topk"         -> take the top_k highest reappearance counts
            - "proportional" -> sample without replacement, p ∝ count
            - "softmax"      -> sample without replacement, p ∝ softmax(count / temperature)
        undirected : if True, normalize edges to (min, max)
        temperature: softmax temperature; lower = peakier
        exclude_edges: set of normalized edges to skip
        rng     : optional random.Random instance (else use module RNG)
        """
        if rng is None:
            rng = random

        # 1) flatten -> counts
        #    each repeated (u,v) in the lists contributes +1 to its reappearance count
        cnt = Counter()
        for u, nbrs in edgebank.items():
            u = int(u)
            for v in nbrs:
                v = int(v)
                if u == v: 
                    continue
                e = self._norm_edge(u, v, undirected)
                cnt[e] += 1

        if not cnt:
            return []

        # optional exclusions (e.g., current snapshot positives or known forbidden)
        if exclude_edges:
            for e in list(cnt.keys()):
                if e in exclude_edges:
                    del cnt[e]

        if not cnt:
            return []

        edges, counts = zip(*cnt.items())
        counts = np.asarray(counts, dtype=np.float64)
        k = min(top_k, len(edges))

        if mode == "topk":
            # deterministic: highest reappearance first; tie-break by node ids for stability
            ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            return [e for e, _c in ranked[:k]]

        elif mode == "proportional":
            # sample without replacement, p ∝ count
            p = counts / counts.sum()
            idx = np.random.choice(len(edges), size=k, replace=False, p=p)
            return [edges[i] for i in idx]

        elif mode == "softmax":
            # smoother than proportional if counts vary wildly
            z = counts / max(1e-8, float(temperature))
            z = z - z.max()             # stabilize
            p = np.exp(z); p /= p.sum()
            idx = np.random.choice(len(edges), size=k, replace=False, p=p)
            return [edges[i] for i in idx]

        else:
            raise ValueError(f"Unknown mode '{mode}' (use 'topk', 'proportional', or 'softmax').")
        
    
    
    def build_accumulating_filtration_sequence_with_edgebank(self, current_target_snapshot):
        """
        Our main driver function to build graphs, takes in various arguments to guide the graph construction
        Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
        But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
        
        Args:
            current_target_graph_description (list): The TopER current_target_graph_description to guide construction of the graph, stores the number of nodes and edges to add to the graph
            thresholds (list): The thresholds for node degrees 'maxDegree' as dicted by TopER
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER current_target_graph_description
            node_types (dict): A dictionary that stores 'old_nodes' and 'new_nodes' organized into lists
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Get the edgebank up to the current target snapshot
        edgebank = self.all_edgebanks[current_target_snapshot]
        current_target_graph_description = self.graph_descriptions[current_target_snapshot]
        prev_graphs = [graph[-1] for graph in self.target_graphs[max(current_target_snapshot - encoder_config["training"]["day"], 0):current_target_snapshot]]
        
        # Prepare the graphs we have known
        known_graphs = self.training_graphs[:self.current_target_snapshot]
        
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])
        W_total = current_target_graph_description[-1][2] 

        # Sample old nodes
        probs = compute_reappearance_probabilities(graphs=prev_graphs,
                                                   t_curr=current_target_snapshot)
        node_ids = list(probs.keys())
        weights = list(probs.values())
        
        old_nodes = list(np.random.choice(node_ids, size=min(len(node_ids), self.current_target_count_old_nodes), replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time

        # Create new node IDs
        current_target_old_nodes = set().union(*[g[-1].nodes() for g in self.target_graphs[:current_target_snapshot]])
        if current_target_old_nodes:
            max_id = max(current_target_old_nodes)
        else:
            max_id = 0

        new_nodes = list(range(max_id + 1, max_id + 1 + self.current_target_count_new_nodes))
        all_nodes = old_nodes + new_nodes

        constructing_graph = nx.DiGraph()  # A graph for computing node embeddings easily
        
        node_types = {
            "old_nodes": old_nodes,
            "new_nodes": new_nodes
        } 
        
        # Assign maximum degrees
        constructing_graph = get_node_features(constructing_graph, prev_graphs, self.thresholds, current_target_graph_description, old_nodes, new_nodes)  
        current_cached_node2vec = [self.cached_node2vec_embeddings[i] for i in range(max(current_target_snapshot - encoder_config["training"]["day"], 0), current_target_snapshot)]

        # Assign embeddings for all the training_nodes
        curr_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], 
                                            graphs=prev_graphs, 
                                            encoder_model=self.encoder_model, device=device, 
                                            cached_node2vec_embeddings=current_cached_node2vec,
                                            snapshots=list(range(max(current_target_snapshot - encoder_config["training"]["day"], 0), current_target_snapshot)))

        # Assign zero vector for new nodes
        for new_node in new_nodes:
            curr_embeddings[new_node] = np.zeros(self.input_dim)
            
        # SAMPLE EDGES STEP
        # Get edges of each type
        edge_pool = []
        
        
        # Construct oobank with edgeBank + Reappearance score
        constructing_graph.add_edges_from(self.sample_oobank_edges(edgebank, self.current_target_count[self.all_edge_types[0]]))
        print(f"Sampled {len(constructing_graph.edges())} o-obank edges")
        
        # Sample edges 4 phases
        for flag in self.all_edge_types[1:]:  # Skip o-o-bank as we will sample directly from the edgebank
            sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=edgebank, link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=current_target_snapshot, device=device)
        
            constructing_graph.add_edges_from(sampled_edges)
            update_degrees(constructing_graph)
            # new_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], graphs=prev_graphs + [constructing_graph], encoder_model=self.encoder_model, device=device)
            new_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], 
                                            graphs=prev_graphs, 
                                            encoder_model=self.encoder_model, device=device, 
                                            cached_node2vec_embeddings=current_cached_node2vec,
                                            snapshots=list(range(max(current_target_snapshot - encoder_config["training"]["day"], 0), current_target_snapshot)))
            curr_embeddings.update(new_embeddings)  # Recompute old node embeddings
        
            edge_pool = edge_pool + sampled_edges
            
        weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
        edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

        # ======== START GRAPH CONSTRUCTION ========
        G = nx.DiGraph()
        used_edges = set()
        filtration_graphs = []

        for i, (v_target, e_target, w_target) in enumerate(current_target_graph_description):
            v_target = int(v_target)
            e_target = int(e_target)

            current_nodes = set(all_nodes[:v_target])
            G.add_nodes_from(current_nodes)

            available_edges = [
                (u, v) for (u, v) in edge_pool
                if u in current_nodes and v in current_nodes and (u, v) not in used_edges
            ]

            needed = e_target - G.number_of_edges()
            selected_edges = available_edges[:needed]

            for (u, v) in selected_edges:
                G.add_edge(u, v, weight=edge_weight_map[(u, v)])
                used_edges.add((u, v))

            filtration_graphs.append(G.copy())

        return filtration_graphs, node_types

    # ======================= Evaluate =======================
    def evaluate(self, pred_graph, true_graph, node_types):
        # Evaluate the graph of o-n 
        pred_on_graph = create_on_graph(node_types["new_nodes"], self.current_target_old_nodes, pred_graph.copy())
        true_on_graph = create_on_graph(node_types["new_nodes"], self.current_target_old_nodes, true_graph.copy())
        
        on_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_on_graph, true_on_graph, mode="total")

        with open(rf'{self.file_visualization_path}\{encoder_config["dataset"]}\{encoder_config["encoder_model"]["nodeEmbeddingType"]}\kl_results_on.txt', "a") as f:
            f.write(f"{self.current_target_snapshot + 1}, {on_kl_divergence_results:.6f}\n")
            f.flush()
            
        # Evaluate the graph of n-n 
        pred_nn_graph = create_nn_graph(node_types["new_nodes"], pred_graph.copy())
        true_nn_graph = create_nn_graph(node_types["new_nodes"], true_graph.copy())

        nn_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_nn_graph, true_nn_graph, mode="total")

        with open(rf'{self.file_visualization_path}\{encoder_config["dataset"]}\{encoder_config["encoder_model"]["nodeEmbeddingType"]}\kl_results_nn.txt', "a") as f:
            f.write(f"{self.current_target_snapshot + 1}, {nn_kl_divergence_results:.6f}\n")
            f.flush()
            
        # Evaluate the graph of old nodes
        oldG = pred_graph.subgraph(self.current_target_old_nodes).copy()
        target_oldG = true_graph.subgraph(self.current_target_old_nodes).copy()
        print(f"Number of nodes in oldG: {len(oldG.nodes())}, Number of edges in oldG: {len(oldG.edges())}")
        print(f"Number of nodes in target_oldG: {len(target_oldG.nodes())}, Number of edges in target_oldG: {len(target_oldG.edges())}")
        
        # results_edges = self.evaluator.evaluateEdges(pred_graph, true_graph, curr_edgebank_pred, all_edgebanks[i], graph_num=i)
        results_true_structure = self.evaluator.evaluateSingleStructure(target_oldG, graph_num=self.current_target_snapshot)
        results_pred_structure = self.evaluator.evaluateSingleStructure(oldG, graph_num=self.current_target_snapshot)
        pred_kernel, true_kernel = None, None
        if len(oldG.nodes()) >= 5:
            pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(oldG, target_oldG)
        
        # Store all results
        pd.DataFrame([results_true_structure]).to_csv(f"{self.structure_dir}/structure_true.csv", mode='a', header=False, index=False)
        pd.DataFrame([results_pred_structure]).to_csv(f"{self.structure_dir}/structure_pred.csv", mode='a', header=False, index=False)
        
        if pred_kernel is not None and true_kernel is not None:
            pd.DataFrame([pred_kernel]).to_csv(f"{self.kernel_dir}/kernel_pred.csv", mode='a', header=False, index=False)
            pd.DataFrame([true_kernel]).to_csv(f"{self.kernel_dir}/kernel_true.csv", mode='a', header=False, index=False)
        else:
            pd.DataFrame([0]).to_csv(f"{self.kernel_dir}/kernel_pred.csv", mode='a', header=False, index=False)
            pd.DataFrame([0]).to_csv(f"{self.kernel_dir}/kernel_true.csv", mode='a', header=False, index=False)
    
    def run(self):             
        print("INFO: Dataset: {}".format(encoder_config["dataset"]))
        self.encoder_model_path = os.path.join(self.saved_input, rf'saved_models/encoder_{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.seed}')
        self.decoder_model_path = os.path.join(self.saved_input, rf'saved_models/decoder_{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.seed}')

        if os.path.exists(self.encoder_model_path) and os.path.exists(self.decoder_model_path):
            self.link_prediction_decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))
            self.encoder_model.load_state_dict(torch.load(self.encoder_model_path, map_location=device))

            self.link_prediction_decoder.to(device)
            self.encoder_model.to(device)
            
            self.link_prediction_decoder.eval()
            self.encoder_model.eval()
            print(f"Link Prediction Decoder loaded from: {self.decoder_model_path}")
            print(f"Encoder loaded from: {self.encoder_model_path}")
        else:
            # Train the Decoder and Encoder model
            print('Training the Link Prediction Decoder and Encoder')
            self.link_prediction_decoder, self.encoder_model = self.train_models()
            torch.save(self.encoder_model.state_dict(), os.path.join(self.saved_input, rf'saved_models/encoder_{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.seed}'))
            torch.save(self.link_prediction_decoder.state_dict(), os.path.join(self.saved_input, rf'saved_models/decoder_{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.seed}'))
            print('Finished training the Link Prediction Decoder and Encoder; Start Graph Construction')
       
        # Old graphs that we know up to now
        self.old_graphs = [self.target_graphs[0], self.target_graphs[1]]
        
        # To predict snapshot i, we use snapshot 0,...,i-1 to train
        for i in range(2, len(self.target_graphs)): 
            print("INFO: >>> Temporal Graph Construction <<<")
            print("INFO: Predict snapshot: ", i)
            print("======================================")

            self.current_target_snapshot = i
            
            # Get all old nodes
            self.current_target_old_nodes = set().union(*[g[-1].nodes() for g in self.old_graphs])
            
            # Get the true count of 4 edges type and number of new, old nodes of the target snapshot
            self.current_target_count_old_nodes = self.probabilities[i][0]
            self.current_target_count_new_nodes = self.probabilities[i][1]
            self.current_target_count = {
                    'o-o-bank': self.probabilities[i][2], 
                    'o-o-nobank': self.probabilities[i][5], 
                    'o-n': self.probabilities[i][4], 
                    'n-n': self.probabilities[i][3]
                }

            print(f"Current target count: {self.current_target_count}")
            # Build the filtration sequence using the current parameters
            filtration_sequence, node_types = self.build_accumulating_filtration_sequence_with_edgebank(current_target_snapshot=i)
            
            # Evaluate generated graph
            self.evaluate(pred_graph=filtration_sequence[-1], true_graph=self.target_graphs[i][-1], node_types=node_types)
            
            # Add to the old graphs
            self.old_graphs.append(self.target_graphs[i])
            
if __name__ == '__main__':
    from GraphGeneration.models.temporal_gnn.script.config import args
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 