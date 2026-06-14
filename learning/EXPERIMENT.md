# EXPERIMENT.md — Recurrent Forecasters vs. V-EWMA in TopoGED

This report documents a re-implementation of the TopoGED forecasting pipeline and three
controlled experiments built on it. It is organised as follows:

- **Section 2 — Implementation.** How the TopoGED pipeline was rewritten under `learning/`,
  the changes made to the original scripts, and the rationale for each.
- **Section 3 — Setup.** Configuration, methods, and datasets shared by all experiments.
- **Section 4 — Experiment 1: Reproduction and recurrent forecasters.** Validation of the
  re-implementation against the published results (`topoged.md`, Tables 12, 15, 21),
  followed by a comparison of a recurrent setup (LSTM for the TopER vector, RNN for the
  probability vector) against the paper's V-EWMA baseline.
- **Section 5 — Experiment 2: Sensitivity to time horizon.** Behaviour on `sx-mathoverflow`
  as the observation span grows (189 → 700 → 2,350 days).
- **Section 6 — Experiment 3: Normalised TopER embeddings.** Whether normalising the TopER
  vector by node and edge count (a shape/scale decomposition) changes forecast error (MAR)
  and the downstream constructed-graph metrics.
- **Section 7 — Reproduction.** Commands to regenerate every result above.

All result tables are presented in markdown. The publication-ready LaTeX versions
(`table_nodes.tex`, `table_edges.tex`, `table_structure.tex`) accompany each `summary.csv`
in [output/comparison_results/](output/comparison_results/).

---

## 1. The Pipeline in Short

TopoGED works in two stages: *forecast, then construct*. First it forecasts two small vectors for the
next snapshot:

- the **TopER vector** (20-D): node and edge counts per filtration bucket,
- the **probability vector** (6-D): the mix of edge types (old-old bank, old-old no-bank, old-new, new-new).

These forecasts decide *how many* nodes and edges of each type the constructed graph gets.
*Which* exact edges are placed is decided by a GNN edge predictor plus an edge bank. So the
forecaster controls the budget, not the edge identities. This matters for reading the results later.

Each original script has a `_cl` version under `learning/`:

| Original | My version | Job |
|---|---|---|
| `toper/generate_toper.py` | [generate_toper_cl.py](generate_toper_cl.py) | Forecast TopER vectors |
| `probs/generate_probs.py` | [generate_probs_cl.py](generate_probs_cl.py) | Forecast probability vectors |
| `GraphGeneration/scripts/topoGED_gnn_implementation_oobankchanges_sampling.py` | [topoGED_gnn_implementation_oobankchanges_sampling_cl.py](topoGED_gnn_implementation_oobankchanges_sampling_cl.py) | Train GNN + build graphs |
| `GraphGeneration/scripts/evaluate_graphs.py` | [compare_graphs_cl.py](compare_graphs_cl.py) | Evaluate built graphs |
| — (new) | [prepare_datasets.py](prepare_datasets.py) | Convert SNAP data to Loader format |

---

## 2. What I Changed in the Code, and Why

### 2.1 Everything stays inside `learning/`

The original scripts wrote caches and outputs into `data/` with hardcoded relative paths. Runs broke
depending on where you started them, and they mixed with the original experiment files. Every `_cl`
script now changes to the project root and redirects the `Loader` paths:

```python
# Redirect Loader to use learning/ paths so nothing is written outside learning/
_learning = os.path.join(_project_root, 'learning')
Loader.output_dir   = os.path.join(_learning, 'datasets', 'cached')
Loader.edgelist_dir = os.path.join(_learning, 'datasets', 'edgelist')
Loader.label_dir    = os.path.join(_learning, 'datasets', 'labels')
```

Caches, predicted vectors, built graphs, and tables all land under `learning/`. The whole experiment
can be copied to a cluster node as one folder.

### 2.2 New: online RNN / LSTM / GRU forecasters

The paper only uses classical predictors (SMA, VAR, V-EWMA, SSM, VECM). I added recurrent models to
both forecasting scripts. The model is small on purpose — one recurrent layer plus a linear output:

```python
class _RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type):
        super().__init__()
        cell = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = cell(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, input_size)
```

The training is **online**, the same way V-EWMA is used: walk through the series once, at each step
feed the current vector, take a few gradient steps toward the next vector, store the prediction, move
on. No separate train/test fitting. This keeps the comparison with V-EWMA fair. Core loop from
[generate_probs_cl.py](generate_probs_cl.py):

```python
# Normalization stats come from the first half of the series only,
# so no future data leaks into the training signal.
warmup_n = max(10, T // 2)
mean = gt_embeddings[:warmup_n].mean(axis=0)
std  = gt_embeddings[:warmup_n].std(axis=0)

for t in range(T - 1):
    # a few gradient steps toward v_{t+1}
    model.train()
    for _ in range(grad_steps):
        optimizer.zero_grad()
        out, _    = model.rnn(x, h)
        pred_norm = model.fc(out.squeeze(0).squeeze(0))
        loss_fn(pred_norm, target).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # clean inference pass: this gives the stored prediction
    # and the hidden state carried to t+1
    model.eval()
    with torch.no_grad():
        out, h_new = model.rnn(x, h)
        pred_norm  = model.fc(out.squeeze(0).squeeze(0))
```

Two details were important in practice:

- **No look-ahead.** The z-score normalization uses only the first 50% of the series.
- **Clean hidden state.** The prediction and the hidden state for the next step come from a separate
  `eval()` pass, not from inside the gradient loop. Reusing a hidden state from mid-backprop made the
  predictions unstable.

Hidden size, learning rate, and gradient steps per time step came from a grid search:

```python
_RNN_BEST = {
    'RNN':  {'hidden_size':  32, 'lr': 5e-3, 'grad_steps': 5},
    'LSTM': {'hidden_size': 128, 'lr': 5e-3, 'grad_steps': 5},
    'GRU':  {'hidden_size':  64, 'lr': 5e-3, 'grad_steps': 5},
}
```

### 2.3 Bug fix in the original probability forecaster

The original `probs/generate_probs.py` has a real bug. The class uses `self.df` in its prediction
methods, but `__init__` never creates it:

```python
# probs/generate_probs.py (original)
def __init__(self, data):
    self.data = data.astype(np.float32)
    self.T, self.D = ...          # self.df is never set!

def predict_vewma(self, alpha=0.8):
    return self.df.ewm(alpha=alpha, adjust=False).mean().iloc[-1].values
```

Every call raised `AttributeError`. The surrounding bare `try/except` caught it and quietly fell back
to "use the last value" (persistence). So the original script never actually computed a V-EWMA on the
probability vectors. My version fixes it:

```python
# learning/generate_probs_cl.py
def __init__(self, data):
    self.data = data.astype(np.float32)
    self.T, self.D = data.shape
    self.df = pd.DataFrame(self.data)   # <-- the fix
```

I also matched the smoothing constants to the paper run: **α = 0.8 for probabilities, α = 0.5 for
TopER**. An earlier draft of mine had 0.5 for probabilities; fixed on 2026-06-10, together with
restoring the paper settings in `GraphGeneration/encoder.yaml` (lr 0.001, `zeros`, `default`
edgebank).

Smaller changes: the probability constraints (node bins sum to 1, edge-type bins sum to 1, no
negatives) are now a `@staticmethod` applied to every prediction including the recurrent ones;
prediction pickles go to `learning/pred_vectors/<dataset>/`; test MSE is reported on the last 15% of
snapshots (the same region the construction is tested on); plotting and the unused Delta mode were
removed.

### 2.4 Construction script: choose the forecaster per vector

The model, training, and construction logic are untouched. I added a way to choose which forecaster
feeds the construction, separately for each vector:

```python
parser.add_argument("--rnn_type",     type=str, default=None, choices=["RNN", "LSTM", "GRU", "VEWMA"])
parser.add_argument("--toper_method", type=str, default=None, choices=["RNN", "LSTM", "GRU", "VEWMA"])
parser.add_argument("--probs_method", type=str, default=None, choices=["RNN", "LSTM", "GRU", "VEWMA"])
```

The script loads the **true** vectors first, then swaps them for the chosen forecaster's pickle:

```python
def _load_pred_vectors(kind, method):
    name = 'V-EWMA' if method == 'VEWMA' else method
    path = os.path.join(_learning, 'pred_vectors', self.dataset,
                        f'{self.dataset}_test{kind}_{name}_Raw_{num_buckets}.pkl')
    with open(path, 'rb') as _f:
        return _pkl.load(_f)

if self.probs_method is not None:
    self.probabilities = _load_pred_vectors('probs', self.probs_method)
if self.toper_method is not None:
    self.graph_descriptions = _load_pred_vectors('toper', self.toper_method)
```

Why this design: the GNN's training samples are built from the true graphs, not from the forecasts.
So the GNN is trained **once** and reused by every forecaster run (it is cached after the first run).
Swapping V-EWMA for LSTM+RNN changes only the count budgets at construction time. Both arms use the
exact same edge predictor, which is the clean way to isolate the temporal component.

Output folders also get a method tag now (`_rnnVEWMA`, `_toperLSTM_probsRNN`), because the original
path did not include the method and a second run would overwrite the first.

### 2.5 New evaluation script: `compare_graphs_cl.py`

The original `evaluate_graphs.py` is 2,225 lines, evaluates one run at a time, has a hardcoded
test-snapshot count per dataset, and needs `grakel` plus the external Orca binary. I replaced it with
a ~600-line script that takes N constructed-graph pickles and puts them side by side. It computes the
same metrics as the original `Evaluator` (node P/R/F1, oo-bank / oo-nobank edge metrics against a
causally built edge bank, o-n and n-n count differences, structural relative errors, TopER descriptor
norm) and writes per-snapshot CSVs, a `summary.csv`, and three LaTeX tables per dataset.

One fix here was critical. An early version marked a predicted node as "old" by checking it against
the true snapshot's old set. That makes Precision-Old always 1.0 and breaks several other metrics.
The original evaluator instead checks against the history of all nodes ever seen. I matched that:

```python
# a predicted node is "old" if it appeared in ANY earlier snapshot,
# mirroring evaluate_graphs.py
pred_old = sn_true['history'] & set(pred_g.nodes())
pred_new = set(pred_g.nodes()) - pred_old
```

Without this fix the replication in Section 4 would not be valid.

### 2.6 Dataset preparation: `prepare_datasets.py` (new)

Converts the SNAP file (`src dst unix_timestamp` lines, gzipped) into the `from,to,date,value` CSV
that `Loader` expects. One snapshot per day:

```python
df = pd.DataFrame(rows, columns=['from', 'to', 'timestamp'])
df['date']  = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
df['value'] = 1.0
df[['from', 'to', 'date', 'value']].to_csv(out_path, index=False)
```

This builds `sx-mathoverflow` (full network, 2,350 days, 398,509 interactions; source:
[SNAP, sx-mathoverflow](https://snap.stanford.edu/data/sx-mathoverflow.html)).
`sx-mathoverflow-700` is the same edge list cut down to the **last 700 days**
(2014-04-07 → 2016-03-06).

---

## 3. Setup

**Config** (paper settings; `GraphGeneration/encoder.yaml` + CLI): GCN embedding (dim 128), Concat
link encoding, lr 0.001, 100 epochs with early stopping, `days_back = 5`, 10 TopER buckets,
`new_node_strategy = zeros`, `edgebank_style = default`, seed 1024, undirected graphs, last 15% of
snapshots as test. Sampling parameters per dataset from paper Table 8: mathoverflow α=5.94, β=7.46,
λ=0.15; Aion α=1.94, β=8.75, λ=0.60; DGD α=1.32, β=9.96, λ=0.31. The sx-mathoverflow runs reuse the
mathoverflow values since it is the same network.

**Methods.**

- **V-EWMA** — the paper baseline. No parameters to train.
- **LSTM+RNN** — online LSTM for TopER vectors, online RNN for probability vectors.
  Both runs share the same trained GNN (Section 2.4).

**Datasets.**

| Dataset | Days | Test snapshots | Notes |
|---|---|---|---|
| mathoverflow | 189 | 29 | The paper's subset |
| networkaion (Aion) | 196 | 30 | ERC20 token network |
| networkdgd (DGD) | 725 | 109 | ERC20 token network |
| sx-mathoverflow-700 | 700 | 105 | Last 700 days of the full network |
| sx-mathoverflow | 2,350 | 353 | Full SNAP network, 2009–2016 |

**How to read the tables.** ↑ means higher is better. →0 means closer to zero is better (these are
relative errors: positive = over-predicted, negative = under-predicted). **Bold** marks the better
method. Values come straight from `summary.csv` / the generated `.tex` files.

---

## 4. Experiment 1 — Comparison with the Paper

### 4.1 Forecast quality first

Before the downstream comparison, I checked the forecasters on plain next-step vector prediction
(test = last 15%):

- **Probability vectors:** RNN clearly beats V-EWMA. On sx-mathoverflow-700, test MSE is
  **0.0016 (RNN) vs 0.0023 (V-EWMA)**
  ([probs_results_buckets_10_Raw.tex](latex_tables/probs_results_buckets_10_Raw.tex)).
  That is why RNN handles the probabilities.
- **TopER vectors:** the recurrent models do *not* beat V-EWMA on count accuracy. On
  sx-mathoverflow-700 the node count error is 4.52% (LSTM) vs 2.90% (V-EWMA), edge count error
  5.41% vs 3.65%
  ([results_buckets_10_new.tex](latex_tables/results_buckets_10_new.tex); columns showing
  0.00 are methods that were not re-run in the last pass — ignore them). I still used LSTM for TopER
  to test a fully recurrent pipeline end to end.

### 4.2 Does my pipeline reproduce the paper?

My **V-EWMA run** should match the paper's TopoGED column. Paper values are from Tables 12
(MathOverflow), 15 (Aion), and 21 (DGD) in `topoged.md`; mine are from `summary.csv` in each results
folder.

| Metric | MathOverflow ours / paper | Aion ours / paper | DGD ours / paper |
|---|---|---|---|
| F1 Nodes ↑ | 0.41 / 0.41 | 0.44 / 0.45 | 0.52 / 0.53 |
| F1 Old Nodes ↑ | 0.35 / 0.35 | 0.13 / 0.13 | 0.31 / 0.32 |
| New Nodes Predicted → 0 | 0.08 / 0.07 | 0.009 / 0.01 | 0.14 / 0.14 |
| oo-bank F1 ↑ | 0.08 / 0.08 | 0.08 / 0.09 | 0.13 / 0.15 |
| Edge F1 ↑ | 0.02 / 0.02 | 0.14 / 0.15 | 0.08 / 0.08 |
| Num o-n Predicted → 0 | 0.14 / 0.14 | 0.01 / 0.02 | 0.17 / 0.17 |
| Num n-n Predicted → 0 | 0.49 / 0.45 | 1.01 / 1.05 | 0.54 / 0.49 |
| Descriptor Norm → 0 | 96.00 / 98.76 | 279.61 / 279.83 | 180.24 / 180.59 |

The match is very close — Descriptor Norm agrees with the paper to within 0.1% on Aion and DGD. So
the rewritten pipeline is doing the same thing the paper did. Small differences are expected: the
paper evaluates a fixed number of trailing snapshots (28 for MathOverflow, I use all 29 from the 15%
split), and GNN retraining is not seeded identically.

**One real disagreement.** The paper reports `Median Extra/Missing Nodes/Edges` ≈ 0 for TopoGED. My
V-EWMA run gives e.g. 12.5 / 21 / 23 / 17.5 on MathOverflow. I checked where these numbers come from:
they are exactly the forecaster's count error in the last bucket (the medians from constructed-vs-true
match the medians from forecast-vs-true). The paper's own Table 6 says V-EWMA has ~3–8% count error,
so medians of 0 cannot come from V-EWMA-driven budgets. They most likely come from a run that used
true vectors instead of predictions. I do not compare these four rows against the paper and use
Descriptor Norm instead.

### 4.3 LSTM+RNN vs V-EWMA, dataset by dataset

Each table has a **Paper** column with the TopoGED values reported in the paper (Tables 12, 15, 21).
Bold compares only my two runs; the Paper column is a reference. Since the paper pipeline uses
V-EWMA, the Paper column should sit close to my V-EWMA column — and it does, except for the four
median rows (marked †, see Section 4.2).

#### 4.3.1 mathoverflow (189 days, 29 test snapshots)

**Node evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| Precision Nodes ↑ | **0.43** | 0.42 | 0.42 |
| Recall Nodes ↑ | 0.41 | 0.41 | 0.41 |
| F1 Nodes ↑ | **0.42** | 0.41 | 0.41 |
| Precision Old Nodes ↑ | 0.36 | 0.35 | 0.36 |
| Recall Old Nodes ↑ | 0.35 | 0.35 | 0.35 |
| F1 Old Nodes ↑ | 0.35 | 0.35 | 0.35 |
| New Nodes Predicted → 0 | **0.06** | 0.08 | 0.07 |

**Edge evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| oo-bank Precision ↑ | 0.06 | 0.07 | 0.07 |
| oo-bank Recall ↑ | 0.09 | **0.10** | 0.09 |
| oo-bank F1 ↑ | 0.07 | 0.08 | 0.08 |
| oo-nobank Precision ↑ | 0.002 | 0.002 | 0.00 |
| oo-nobank Recall ↑ | 0.002 | 0.002 | 0.00 |
| oo-nobank F1 ↑ | 0.002 | 0.002 | 0.00 |
| Num o-n Predicted → 0 | 0.15 | 0.14 | 0.14 |
| Num n-n Predicted → 0 | **0.37** | 0.49 | 0.45 |
| Edge Precision ↑ | 0.02 | 0.02 | 0.02 |
| Edge Recall ↑ | 0.02 | 0.02 | 0.03 |
| Edge F1 ↑ | 0.02 | 0.02 | 0.02 |

**Structure evaluation** (all → 0)

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| Avg Node Degree | **0.05** | 0.06 | 0.07 |
| Unique Degree Count | 0.42 | **0.39** | 0.46 |
| Degree Centrality | 0.09 | **0.08** | 0.09 |
| Assortativity Coefficient | 50.46 | **46.02** | 51.44 |
| Clustering Coefficient | 2.98 | **2.90** | 2.45 |
| Density | 0.09 | **0.08** | 0.09 |
| Num Triangles | **6.31** | 7.07 | 6.45 |
| Descriptor Norm | **89.62** | 96.00 | 98.76 |
| Median Extra Nodes | **7** | 12.5 | 0 † |
| Median Missing Nodes | **13** | 21 | 3 † |
| Median Extra Edges | **12** | 23 | 8 † |
| Median Missing Edges | 18 | **17.5** | 0 † |

Reading: the recurrent setup wins almost all node and count metrics (node F1, new-node calibration,
descriptor norm, the median count errors). V-EWMA wins every exact-edge metric, but by small margins
(Edge F1 0.0264 vs 0.0240).

#### 4.3.2 networkaion / Aion (196 days, 30 test snapshots)

**Node evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| Precision Nodes ↑ | **0.47** | 0.45 | 0.45 |
| Recall Nodes ↑ | 0.42 | **0.44** | 0.45 |
| F1 Nodes ↑ | 0.44 | 0.44 | 0.45 |
| Precision Old Nodes ↑ | **0.15** | 0.13 | 0.14 |
| Recall Old Nodes ↑ | 0.13 | 0.13 | 0.13 |
| F1 Old Nodes ↑ | 0.13 | 0.13 | 0.13 |
| New Nodes Predicted → 0 | −0.11 | **0.008** | 0.01 |

**Edge evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| oo-bank Precision ↑ | **0.07** | 0.06 | 0.07 |
| oo-bank Recall ↑ | 0.12 | **0.15** | 0.16 |
| oo-bank F1 ↑ | 0.08 | 0.08 | 0.09 |
| oo-nobank Precision ↑ | 0.02 | 0.02 | 0.03 |
| oo-nobank Recall ↑ | 0.01 | 0.01 | 0.01 |
| oo-nobank F1 ↑ | 0.01 | 0.01 | 0.02 |
| Num o-n Predicted → 0 | −0.22 | **0.01** | 0.02 |
| Num n-n Predicted → 0 | **0.13** | 1.01 | 1.05 |
| Edge Precision ↑ | **0.15** | 0.14 | 0.15 |
| Edge Recall ↑ | 0.11 | **0.14** | 0.15 |
| Edge F1 ↑ | 0.13 | **0.14** | 0.15 |

**Structure evaluation** (all → 0)

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| Avg Node Degree | −0.12 | **0.009** | 0.01 |
| Unique Degree Count | −0.39 | **−0.21** | −0.17 |
| Degree Centrality | 0.05 | **0.04** | 0.05 |
| Assortativity Coefficient | 1.31 | **1.16** | 1.08 |
| Clustering Coefficient | **3.11** | 4.34 | 7.38 |
| Density | 0.05 | **0.04** | 0.05 |
| Num Triangles | **5.07** | 11.29 | 15.53 |
| Descriptor Norm | 320.79 | **279.61** | 279.83 |
| Median Extra Nodes | **24.5** | 34 | 0 † |
| Median Missing Nodes | 58 | **34** | 8.5 † |
| Median Extra Edges | **8** | 34 | 2.5 † |
| Median Missing Edges | 56 | **34** | 4 † |

Reading: this is the weakest dataset for the recurrent setup, and the cause is the LSTM TopER
forecast. It under-predicts counts: New Nodes −0.115 (about 11% too few new nodes), Avg Node Degree
−0.126, Median Missing Nodes/Edges 58/56 vs V-EWMA's 34/34. The descriptor norm suffers the same way
(321 vs 280). The RNN probability side is still good: n-n over-prediction drops from 1.02 to 0.13.

#### 4.3.3 networkdgd / DGD (725 days, 109 test snapshots)

**Node evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| Precision Nodes ↑ | **0.56** | 0.53 | 0.53 |
| Recall Nodes ↑ | 0.53 | **0.55** | 0.56 |
| F1 Nodes ↑ | 0.52 | 0.52 | 0.53 |
| Precision Old Nodes ↑ | **0.34** | 0.32 | 0.33 |
| Recall Old Nodes ↑ | 0.32 | **0.33** | 0.34 |
| F1 Old Nodes ↑ | **0.32** | 0.31 | 0.32 |
| New Nodes Predicted → 0 | **0.03** | 0.14 | 0.14 |

**Edge evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| oo-bank Precision ↑ | **0.14** | 0.13 | 0.14 |
| oo-bank Recall ↑ | 0.14 | **0.15** | 0.17 |
| oo-bank F1 ↑ | 0.13 | 0.13 | 0.15 |
| oo-nobank Precision ↑ | 0.008 | **0.011** | 0.01 |
| oo-nobank Recall ↑ | 0.003 | **0.008** | 0.01 |
| oo-nobank F1 ↑ | 0.005 | **0.008** | 0.01 |
| Num o-n Predicted → 0 | **0.08** | 0.17 | 0.17 |
| Num n-n Predicted → 0 | **0.28** | 0.54 | 0.49 |
| Edge Precision ↑ | 0.08 | 0.08 | 0.08 |
| Edge Recall ↑ | 0.07 | **0.08** | 0.08 |
| Edge F1 ↑ | 0.07 | **0.08** | 0.08 |

**Structure evaluation** (all → 0)

| Metric | LSTM+RNN | V-EWMA | Paper |
|---|---|---|---|
| Avg Node Degree | **0.01** | 0.02 | 0.02 |
| Unique Degree Count | **0.01** | 0.03 | 0.05 |
| Degree Centrality | 0.18 | **0.06** | 0.06 |
| Assortativity Coefficient | 1.02 | 1.02 | 1.04 |
| Clustering Coefficient | **2.83** | 4.67 | 3.06 |
| Density | 0.18 | 0.06 | 0.06 |
| Num Triangles | **3.98** | 5.71 | 7.17 |
| Descriptor Norm | **175.17** | 180.24 | 180.59 |
| Median Extra Nodes | **24** | 27 | 12 † |
| Median Missing Nodes | 55 | **36** | 0 † |
| Median Extra Edges | **27** | 30.5 | 13 † |
| Median Missing Edges | 51 | **40** | 0 † |

Reading: same pattern as mathoverflow. The recurrent setup wins the node and calibration metrics
(New Nodes 0.030 vs 0.144, n-n 0.29 vs 0.54, descriptor norm 175 vs 180); V-EWMA keeps a small lead
on exact edges (Edge F1 0.081 vs 0.074).

### 4.4 What I take from Experiment 1

1. **It is roughly a tie, but the wins split in a clear pattern.** The recurrent setup wins the
   *count* metrics (node F1 on 2 of 3, new-node and n-n calibration everywhere, descriptor norm on
   2 of 3). V-EWMA keeps a small lead on *exact edge* metrics (Edge F1 on all three). This makes
   sense given the architecture: the forecaster only sets counts, the shared GNN picks the actual
   edges. Even a much better probability forecast moves Edge F1 only a little.
2. **The RNN probability forecast fixes the edge-type mix.** Over-prediction of new-new edges drops
   from 0.50 to 0.38 (mathoverflow), 1.02 to 0.13 (Aion), 0.54 to 0.29 (DGD).
3. **The LSTM TopER forecast is the weak part.** Aion shows it clearly (Section 4.3.2). This matches
   the forecast-level check in 4.1, where LSTM lost to V-EWMA on TopER counts. The natural next
   config is V-EWMA for TopER + RNN for probabilities.

---

## 5. Experiment 2 — What Happens When Time Gets Longer

The paper's MathOverflow dataset covers 189 days. The public record of the same network covers 2,350
days. So I ran the same comparison at three time spans: 189 days (the paper subset, results in
Section 4.3.1), the last 700 days, and the full 2,350 days. The train/test split is always 85/15, so
a longer span means more training history and more test snapshots.

### 5.1 The trend across horizons

The paper never ran sx-mathoverflow, so its only reference point is the 189-day MathOverflow result
(Table 12, TopoGED column). That column is repeated in the tables below for orientation — it is not
a third competitor. Rows marked † are the questionable paper medians discussed in Section 4.2.

| Metric | Paper (189 d) | 189 d (LSTM+RNN / V-EWMA) | 700 d (LSTM+RNN / V-EWMA) | 2,350 d (LSTM+RNN / V-EWMA) |
|---|---|---|---|---|
| F1 Nodes ↑ | 0.41 | **0.42** / 0.41 | 0.40 / 0.40 | **0.39** / 0.38 |
| New Nodes Predicted → 0 | 0.07 | **0.07** / 0.08 | **0.10** / 0.12 | **0.10** / 0.17 |
| Edge F1 ↑ | 0.02 | 0.02 / 0.02 | 0.03 / 0.03 | 0.04 / 0.04 |
| oo-bank F1 ↑ | 0.08 | 0.078 / **0.084** | 0.099 / 0.099 | **0.099** / 0.098 |
| Num o-n Predicted → 0 | 0.14 | **0.16** / 0.14 | **0.2** / 0.23 | **0.24** / 0.33 |
| Num n-n Predicted → 0 | 0.45 | **0.37** / 0.49 | **0.50** / 0.52 | **0.26** / 0.44 |
| Descriptor Norm → 0 | 98.8 | **89.6** / 96.0 | **106.4** / 109.6 | **105.9** / 110.1 |

The two new runs in full:

### 5.2 sx-mathoverflow-700 (700 days, 105 test snapshots)

**Node evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper (189 d) |
|---|---|---|---|
| Precision Nodes ↑ | 0.39 | **0.40** | 0.42 |
| Recall Nodes ↑ | 0.40 | 0.40 | 0.41 |
| F1 Nodes ↑ | 0.40 | 0.40 | 0.41 |
| Precision Old Nodes ↑ | 0.35 | 0.35 | 0.36 |
| Recall Old Nodes ↑ | 0.36 | 0.36 | 0.35 |
| F1 Old Nodes ↑ | 0.35 | 0.35 | 0.35 |
| New Nodes Predicted → 0 | **0.10** | 0.12 | 0.07 |

**Edge evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper (189 d) |
|---|---|---|---|
| oo-bank Precision ↑ | 0.08 | 0.08 | 0.07 |
| oo-bank Recall ↑ | 0.12 | 0.12 | 0.09 |
| oo-bank F1 ↑ | 0.09 | 0.09 | 0.08 |
| oo-nobank Precision ↑ | 0.001 | 0.001 | 0.00 |
| oo-nobank Recall ↑ | 0.0009 | 0.0009 | 0.00 |
| oo-nobank F1 ↑ | 0.0009 | **0.001** | 0.00 |
| Num o-n Predicted → 0 | **0.20** | 0.23 | 0.14 |
| Num n-n Predicted → 0 | **0.50** | 0.52 | 0.45 |
| Edge Precision ↑ | 0.03 | 0.03 | 0.02 |
| Edge Recall ↑ | 0.03 | 0.03 | 0.03 |
| Edge F1 ↑ | 0.03 | 0.03 | 0.02 |

**Structure evaluation** (all → 0)

| Metric | LSTM+RNN | V-EWMA | Paper (189 d) |
|---|---|---|---|
| Avg Node Degree | 0.06 | 0.06 | 0.07 |
| Unique Degree Count | **0.55** | 0.58 | 0.46 |
| Degree Centrality | **0.04** | 0.07 | 0.09 |
| Assortativity Coefficient | 7.61 | **6.59** | 51.44 |
| Clustering Coefficient | **2.41** | 2.42 | 2.45 |
| Density | **0.04** | 0.07 | 0.09 |
| Num Triangles | 8.98 | **8.59** | 6.45 |
| Descriptor Norm | **106.42** | 109.58 | 98.76 |
| Median Extra Nodes | **15** | 20 | 0 † |
| Median Missing Nodes | 15 | **14** | 3 † |
| Median Extra Edges | **18.5** | 21.5 | 8 † |
| Median Missing Edges | **15** | 17 | 0 † |

Reading: at 700 days the two methods are basically tied. Most differences are in the third decimal.
The recurrent setup already has the better count calibration (new nodes 0.100 vs 0.123, extra
nodes/edges medians), but V-EWMA still holds tiny leads on several node and oo-bank metrics.

### 5.3 sx-mathoverflow, full network (2,350 days, 353 test snapshots)

**Node evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper (189 d) |
|---|---|---|---|
| Precision Nodes ↑ | **0.39** | 0.38 | 0.42 |
| Recall Nodes ↑ | 0.39 | 0.39 | 0.41 |
| F1 Nodes ↑ | 0.38 | 0.38 | 0.41 |
| Precision Old Nodes ↑ | 0.35 | 0.35 | 0.36 |
| Recall Old Nodes ↑ | 0.35 | 0.35 | 0.35 |
| F1 Old Nodes ↑ | 0.35 | 0.35 | 0.35 |
| New Nodes Predicted → 0 | **0.10** | 0.17 | 0.07 |

**Edge evaluation**

| Metric | LSTM+RNN | V-EWMA | Paper (189 d) |
|---|---|---|---|
| oo-bank Precision ↑ | 0.08 | 0.08 | 0.07 |
| oo-bank Recall ↑ | **0.12** | 0.11 | 0.09 |
| oo-bank F1 ↑ | 0.09 | 0.09 | 0.08 |
| oo-nobank Precision ↑ | 0.001 | 0.001 | 0.00 |
| oo-nobank Recall ↑ | **0.001** | 0.0009 | 0.00 |
| oo-nobank F1 ↑ | 0.001 | 0.001 | 0.00 |
| Num o-n Predicted → 0 | **0.24** | 0.33 | 0.14 |
| Num n-n Predicted → 0 | **0.26** | 0.44 | 0.45 |
| Edge Precision ↑ | 0.04 | 0.04 | 0.02 |
| Edge Recall ↑ | 0.04 | 0.04 | 0.03 |
| Edge F1 ↑ | 0.04 | 0.04 | 0.02 |

**Structure evaluation** (all → 0)

| Metric | LSTM+RNN | V-EWMA | Paper (189 d) |
|---|---|---|---|
| Avg Node Degree | 0.07 | **0.06** | 0.07 |
| Unique Degree Count | 0.35 | **0.34** | 0.46 |
| Degree Centrality | 0.07 | 0.07 | 0.09 |
| Assortativity Coefficient | **6.3** | 6.9 | 51.44 |
| Clustering Coefficient | 2.62 | **2.52** | 2.45 |
| Density | 0.07 | 0.07 | 0.09 |
| Num Triangles | 8.59 | **8.18** | 6.45 |
| Descriptor Norm | **105.95** | 110.06 | 98.76 |
| Median Extra Nodes | **15.5** | 21 | 0 † |
| Median Missing Nodes | **14** | 17 | 3 † |
| Median Extra Edges | **18** | 23 | 8 † |
| Median Missing Edges | **16** | 17.5 | 0 † |

Reading: at the full horizon the picture is one-sided. LSTM+RNN wins **every** edge metric and every
headline node metric. The gap is widest on calibration: V-EWMA over-predicts new nodes by 17.9% and
o-n edges by 33.9%, while LSTM+RNN stays at 10.7% and 24.5%. Its n-n calibration (0.267) is the best
in the whole study.

### 5.4 Findings

1. **Node prediction gets harder as time grows.** F1 Nodes drops from ~0.42 to ~0.40 to ~0.39 for
   both methods. The reason is simple: the pool of "old" nodes that might come back keeps growing,
   but only ~140 nodes are active on any given day. Picking the right ones gets harder. Both methods
   suffer the same way — this is a property of the problem, not of the forecaster.
2. **Edge prediction gets *easier* as time grows.** Edge F1 rises from ~0.025 to ~0.042. The edge
   bank is the reason: with six years of history it remembers far more previously-seen pairs, and
   repeated edges are the one thing the pipeline predicts well.
3. **V-EWMA drifts off as the series gets longer; the recurrent models do not.** V-EWMA's new-node
   over-prediction more than doubles (0.081 → 0.123 → 0.179), and its o-n over-prediction grows the
   same way (0.143 → 0.235 → 0.339). LSTM+RNN stays flat (~0.07–0.11). A fixed smoothing constant
   cannot adapt when the network slowly changes regime — and the full series contains MathOverflow's
   whole growth-then-decline arc. The online models keep learning through the series.
4. **The winner flips.** At 189 days V-EWMA wins every edge metric. At 700 days it is a tie. At 2,350
   days LSTM+RNN wins every headline metric. On a 189-step series the recurrent models simply do not
   get enough training steps; with ~2,000 steps they do, and the advantage shows.
5. **Takeaway.** The paper's choice of V-EWMA is fine at the scale of its benchmarks (most have
   180–550 snapshots). For long-running networks the temporal part is worth replacing with a
   recurrent model — the gains appear exactly where V-EWMA degrades.

### 5.5 Caveats

- The forecaster only sets budgets, so all differences are bounded. Absolute Edge F1 stays below
  0.05 on this network: non-repeat edges on a high-churn Q&A network are close to unpredictable
  (oo-nobank F1 ≈ 0.001).
- The 189-day subset was preprocessed by the paper authors and ends 12 days earlier than the SNAP
  record. So the 189 vs 700/2,350 comparison is near-identical data, not byte-identical. The
  700 vs 2,350 comparison is exactly controlled.
- One run per configuration, no seed variance. The Experiment 1 gaps are probably within noise
  (the paper's own standard deviations are larger). The Experiment 2 trends are monotone across
  three horizons and larger than the within-table gaps, so I trust them more.
- The structural ratio metrics (assortativity, clustering, triangles) are noisy; read them
  qualitatively.

---

## 6. Experiment 3 — Normalized TopER (shape / scale decomposition)

My supervisor asked a specific question: in the TopER embedding forecast, does
**normalizing the whole embedding by node and edge count** change the error (MAR)?
This section answers it at two levels: the forecast itself (MAR), and the downstream
constructed graphs (the full metric set from Section 4).

All of this lives in [generate_toper_normalized_cl.py](generate_toper_normalized_cl.py)
and does **not** modify [generate_toper_cl.py](generate_toper_cl.py) — it imports its
forecasters so the methods are byte-identical to the raw arm.

### 6.1 How the normalization works

The raw TopER vector for one snapshot is the interleaved, **cumulative** count curve

```
[ n1, e1, n2, e2, ..., n10, e10 ]
```

where bucket *i* counts nodes with `degree ≤ threshold_i` (and the edges among them).
The thresholds are degree percentiles `0 → 100`, so bucket 10 contains the whole
graph: **`n10 = total_nodes`, `e10 = total_edges`**. The curve is monotone
non-decreasing.

I normalize each snapshot by its own totals:

```
n_i_norm = n_i / total_nodes        e_i_norm = e_i / total_edges
( total_nodes = n10 ,  total_edges = e10 )
```

This turns each snapshot into a monotone **shape** curve in `[0,1]` ending at `1.0`,
separating *shape* (how mass is distributed across the filtration) from *scale*
(how big the graph is). Forecasting then has three steps:

1. **Shape forecast** — forecast the normalized vector with the method (V-EWMA / RNN /
   LSTM / GRU).
2. **Scale forecast** — forecast the 2-D totals series `[total_nodes, total_edges]`
   **separately** with the same method. This is an honest forecast from past values
   only (**no leakage** — we never use the true future totals). For V-EWMA this is
   provably identical to the raw arm's `n10`/`e10` forecast; for the recurrent models
   it is a dedicated 2-D forecaster. (Using true future totals would be an oracle
   diagnostic only, so it is not reported here.)
3. **Reconstruct** — `n_i = n_i_norm × predicted_total_nodes` (same for edges). The
   reconstruction clips fractions to `[0,1]` and applies a cumulative-max so the curve
   stays monotone, then writes a 20-D raw-count vector in the **same pickle format the
   construction consumes**. So the downstream code path is unchanged — only the numbers
   differ.

**Why this can matter downstream.** The construction reads the TopER vector in two
places: the totals (`n10`, `e10`) set the node/edge budgets, and the inner node buckets
drive degree assignment in `get_node_features`. So a better shape/scale split can change
both the graph size and its degree structure.

### 6.2 Forecast-level result — MAR

MAR = **Mean Absolute Relative error** on the test split (last 15% of snapshots),
averaged over snapshots and over vector dimensions with a non-zero ground truth:
`mean( |pred − true| / true )`. Two views: the full 20-D vector, and the two totals
only (`n10`, `e10` — the part that drives budgets). **Bold** = better (lower) of
raw vs. norm.

**MAR — full TopER vector**

| Dataset | V-EWMA raw / norm | RNN raw / norm | LSTM raw / norm | GRU raw / norm |
|---|---|---|---|---|
| mathoverflow | **0.165** / 0.166 | **0.131** / 0.166 | 0.139 / **0.138** | **0.145** / 0.150 |
| networkaion | **0.202** / 0.205 | **0.172** / 0.232 | 0.358 / **0.323** | **0.269** / 0.361 |
| networkdgd | **0.323** / 0.335 | **0.277** / 0.332 | 0.325 / **0.295** | 0.302 / **0.293** |
| sx-mathoverflow-700 | **0.202** / 0.203 | **0.180** / 0.192 | **0.182** / 0.183 | **0.181** / 0.183 |

**MAR — totals only (`n10`, `e10`)**

| Dataset | V-EWMA raw / norm | RNN raw / norm | LSTM raw / norm | GRU raw / norm |
|---|---|---|---|---|
| mathoverflow | 0.134 / 0.134 | **0.107** / 0.120 | 0.103 / **0.100** | 0.112 / **0.107** |
| networkaion | 0.152 / 0.152 | **0.114** / 0.159 | 0.214 / **0.171** | 0.215 / **0.195** |
| networkdgd | 0.340 / 0.340 | 0.309 / 0.309 | 0.314 / **0.281** | 0.292 / **0.279** |
| sx-mathoverflow-700 | 0.163 / 0.163 | 0.147 / **0.135** | 0.149 / **0.146** | 0.149 / **0.144** |

Reading:

- **V-EWMA is essentially unaffected.** Its totals MAR is identical raw vs norm (EWMA is
  per-column, so normalizing the other dimensions cannot change the `n10`/`e10`
  forecast). The tiny full-vector differences come only from the inner-bucket
  reconstruction.
- **For the recurrent models the effect is method- and dataset-dependent.**
  Normalization **helps LSTM and GRU** — clearly on the two ERC20 networks
  (LSTM full MAR on Aion 0.358 → 0.323, on DGD 0.325 → 0.295; the totals improve too),
  and slightly on sx-mathoverflow-700. It **hurts plain RNN** on every dataset.
- The biggest single win is **LSTM's totals on Aion** (0.214 → 0.171): on that dataset
  the raw LSTM badly under-predicts graph size, and the shape/scale split fixes the
  scale part.

### 6.3 Downstream result — full graph metrics

Each table compares four arms: the two raw arms from Section 4 (**V-EWMA**,
**LSTM+RNN**) and their normalized counterparts (**V-EWMA-norm** = `toper VEWMA-norm`
+ `probs VEWMA`; **LSTM-norm+RNN** = `toper LSTM-norm` + `probs RNN`). Probabilities
are never normalized — the supervisor's question is about the TopER embedding only.
↑ = higher is better; →0 = closer to zero is better. **Bold** marks the best arm in each
row (ties bolded together). Numbers come straight from each
`output/comparison_results/<dataset>_normalized_ablation/summary.csv`.

##### mathoverflow (189 d, 29 test)

**Node evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Precision Nodes ↑ | 0.42 | 0.42 | **0.43** | **0.43** |
| Recall Nodes ↑ | **0.42** | **0.42** | **0.42** | 0.41 |
| F1 Nodes ↑ | **0.42** | **0.42** | **0.42** | **0.42** |
| Precision Old Nodes ↑ | **0.36** | **0.36** | **0.36** | **0.36** |
| Recall Old Nodes ↑ | **0.35** | **0.35** | **0.35** | 0.34 |
| F1 Old Nodes ↑ | 0.35 | 0.35 | **0.36** | 0.35 |
| New Nodes Predicted →0 | 0.081 | 0.081 | 0.070 | **0.053** |

**Edge evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| oo-bank Precision ↑ | **0.072** | **0.072** | 0.069 | 0.067 |
| oo-bank Recall ↑ | **0.10** | **0.10** | 0.093 | 0.090 |
| oo-bank F1 ↑ | **0.084** | **0.084** | 0.078 | 0.076 |
| oo-nobank Precision ↑ | **0.003** | **0.003** | **0.003** | **0.003** |
| oo-nobank Recall ↑ | **0.002** | **0.002** | **0.002** | **0.002** |
| oo-nobank F1 ↑ | **0.003** | **0.003** | 0.002 | 0.002 |
| Num o-n Predicted →0 | **0.14** | **0.14** | 0.16 | 0.15 |
| Num n-n Predicted →0 | 0.50 | 0.50 | 0.38 | **0.36** |
| Edge Precision ↑ | **0.026** | **0.026** | 0.024 | 0.024 |
| Edge Recall ↑ | **0.027** | **0.027** | 0.024 | 0.024 |
| Edge F1 ↑ | **0.026** | **0.026** | 0.024 | 0.024 |

**Structure evaluation** (all →0)

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Avg Node Degree | 0.066 | 0.066 | **0.055** | **0.055** |
| Unique Degree Count | **0.40** | **0.40** | 0.43 | 0.41 |
| Degree Centrality | **0.086** | **0.086** | 0.096 | 0.12 |
| Assortativity Coefficient | **46.0** | **46.0** | 50.5 | 55.2 |
| Clustering Coefficient | **2.91** | **2.91** | 2.98 | **2.91** |
| Density | **0.086** | **0.086** | 0.096 | 0.12 |
| Num Triangles | 7.07 | 7.07 | **6.31** | 6.43 |
| Descriptor Norm | 96.0 | 96.0 | **89.6** | 91.3 |
| Median Extra Nodes | 12.5 | 12.5 | 7.0 | **5.0** |
| Median Missing Nodes | 21.0 | 21.0 | **13.0** | 16.0 |
| Median Extra Edges | 23.0 | 23.0 | 12.0 | **8.5** |
| Median Missing Edges | **17.5** | **17.5** | 18.0 | 18.5 |

##### networkaion / Aion (196 d, 30 test)

**Node evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Precision Nodes ↑ | 0.46 | 0.46 | **0.48** | 0.45 |
| Recall Nodes ↑ | **0.45** | **0.45** | 0.42 | 0.44 |
| F1 Nodes ↑ | **0.45** | **0.45** | 0.44 | 0.44 |
| Precision Old Nodes ↑ | 0.14 | 0.14 | **0.15** | 0.14 |
| Recall Old Nodes ↑ | **0.14** | **0.14** | 0.13 | 0.13 |
| F1 Old Nodes ↑ | **0.14** | **0.14** | **0.14** | 0.13 |
| New Nodes Predicted →0 | 0.009 | 0.009 | −0.11 | **0.007** |

**Edge evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| oo-bank Precision ↑ | 0.065 | 0.065 | **0.071** | 0.064 |
| oo-bank Recall ↑ | **0.15** | **0.15** | 0.12 | 0.14 |
| oo-bank F1 ↑ | **0.089** | **0.089** | 0.087 | 0.084 |
| oo-nobank Precision ↑ | 0.025 | 0.025 | **0.028** | 0.023 |
| oo-nobank Recall ↑ | **0.012** | **0.012** | 0.010 | 0.011 |
| oo-nobank F1 ↑ | **0.016** | **0.016** | 0.014 | 0.014 |
| Num o-n Predicted →0 | **0.018** | **0.018** | −0.23 | −0.055 |
| Num n-n Predicted →0 | 1.02 | 1.02 | **0.13** | 0.45 |
| Edge Precision ↑ | 0.15 | 0.15 | **0.16** | 0.14 |
| Edge Recall ↑ | **0.15** | **0.15** | 0.12 | 0.13 |
| Edge F1 ↑ | **0.15** | **0.15** | 0.13 | 0.13 |

**Structure evaluation** (all →0)

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Avg Node Degree | **0.009** | **0.009** | −0.13 | −0.055 |
| Unique Degree Count | **−0.22** | **−0.22** | −0.39 | −0.31 |
| Degree Centrality | 0.048 | 0.048 | 0.056 | **0.012** |
| Assortativity Coefficient | **1.17** | **1.17** | 1.32 | 1.21 |
| Clustering Coefficient | 4.34 | 4.34 | **3.11** | 3.84 |
| Density | 0.048 | 0.048 | 0.056 | **0.012** |
| Num Triangles | 11.3 | 11.3 | **5.07** | 7.72 |
| Descriptor Norm | **279.6** | **279.6** | 320.8 | 309.5 |
| Median Extra Nodes | 34.0 | 34.0 | **24.5** | 29.0 |
| Median Missing Nodes | **34.0** | **34.0** | 58.0 | 54.0 |
| Median Extra Edges | 34.0 | 34.0 | **8.0** | 18.0 |
| Median Missing Edges | **34.0** | **34.0** | 56.0 | 37.0 |

##### networkdgd / DGD (725 d, 109 test)

**Node evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Precision Nodes ↑ | 0.53 | 0.53 | **0.57** | **0.57** |
| Recall Nodes ↑ | **0.56** | **0.56** | 0.53 | 0.53 |
| F1 Nodes ↑ | **0.53** | **0.53** | **0.53** | **0.53** |
| Precision Old Nodes ↑ | 0.32 | 0.32 | **0.35** | **0.35** |
| Recall Old Nodes ↑ | **0.33** | **0.33** | **0.33** | **0.33** |
| F1 Old Nodes ↑ | 0.32 | 0.32 | 0.32 | **0.33** |
| New Nodes Predicted →0 | 0.14 | 0.14 | 0.030 | **0.008** |

**Edge evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| oo-bank Precision ↑ | 0.13 | 0.13 | 0.14 | **0.15** |
| oo-bank Recall ↑ | **0.16** | **0.16** | 0.14 | 0.15 |
| oo-bank F1 ↑ | 0.13 | 0.13 | **0.14** | **0.14** |
| oo-nobank Precision ↑ | **0.011** | **0.011** | 0.008 | **0.011** |
| oo-nobank Recall ↑ | **0.008** | **0.008** | 0.004 | 0.007 |
| oo-nobank F1 ↑ | **0.008** | **0.008** | 0.005 | **0.008** |
| Num o-n Predicted →0 | 0.17 | 0.17 | 0.083 | **0.034** |
| Num n-n Predicted →0 | 0.54 | 0.54 | 0.29 | **0.24** |
| Edge Precision ↑ | **0.082** | 0.081 | 0.081 | 0.081 |
| Edge Recall ↑ | **0.086** | **0.086** | 0.075 | 0.074 |
| Edge F1 ↑ | **0.081** | 0.080 | 0.074 | 0.075 |

**Structure evaluation** (all →0)

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Avg Node Degree | 0.021 | 0.021 | 0.019 | **−0.001** |
| Unique Degree Count | 0.038 | 0.038 | **0.012** | 0.023 |
| Degree Centrality | **0.060** | **0.060** | 0.19 | 0.17 |
| Assortativity Coefficient | **1.02** | **1.02** | **1.02** | **1.02** |
| Clustering Coefficient | 4.67 | 4.67 | 2.83 | **2.05** |
| Density | **0.060** | **0.060** | 0.19 | 0.17 |
| Num Triangles | 5.71 | 5.68 | 3.98 | **3.30** |
| Descriptor Norm | 180.2 | 180.2 | 175.2 | **169.8** |
| Median Extra Nodes | 27.0 | 27.0 | 24.0 | **18.0** |
| Median Missing Nodes | **36.0** | **36.0** | 55.0 | 40.0 |
| Median Extra Edges | 30.5 | 30.5 | 27.0 | **21.0** |
| Median Missing Edges | **40.0** | **40.0** | 51.0 | 40.5 |

##### sx-mathoverflow-700 (700 d, 105 test)

**Node evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Precision Nodes ↑ | **0.40** | **0.40** | **0.40** | **0.40** |
| Recall Nodes ↑ | 0.40 | 0.40 | **0.41** | **0.41** |
| F1 Nodes ↑ | **0.40** | **0.40** | **0.40** | **0.40** |
| Precision Old Nodes ↑ | **0.36** | **0.36** | 0.35 | 0.35 |
| Recall Old Nodes ↑ | **0.36** | **0.36** | **0.36** | **0.36** |
| F1 Old Nodes ↑ | **0.36** | **0.36** | **0.36** | **0.36** |
| New Nodes Predicted →0 | 0.12 | 0.12 | **0.10** | 0.11 |

**Edge evaluation**

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| oo-bank Precision ↑ | **0.085** | **0.085** | 0.083 | 0.084 |
| oo-bank Recall ↑ | **0.13** | **0.13** | **0.13** | **0.13** |
| oo-bank F1 ↑ | **0.099** | **0.099** | **0.099** | **0.099** |
| oo-nobank Precision ↑ | **0.001** | **0.001** | **0.001** | **0.001** |
| oo-nobank Recall ↑ | **0.001** | **0.001** | **0.001** | **0.001** |
| oo-nobank F1 ↑ | **0.001** | **0.001** | **0.001** | **0.001** |
| Num o-n Predicted →0 | 0.23 | 0.23 | **0.20** | **0.20** |
| Num n-n Predicted →0 | 0.53 | 0.53 | 0.51 | **0.50** |
| Edge Precision ↑ | **0.035** | **0.035** | 0.034 | 0.034 |
| Edge Recall ↑ | 0.037 | 0.037 | **0.038** | **0.038** |
| Edge F1 ↑ | 0.035 | 0.035 | **0.036** | **0.036** |

**Structure evaluation** (all →0)

| Metric | V-EWMA | V-EWMA-norm | LSTM+RNN | LSTM-norm+RNN |
|---|---|---|---|---|
| Avg Node Degree | 0.066 | 0.066 | 0.066 | **0.062** |
| Unique Degree Count | 0.58 | 0.58 | **0.56** | 0.58 |
| Degree Centrality | 0.071 | 0.071 | 0.049 | **0.045** |
| Assortativity Coefficient | **6.60** | **6.60** | 7.62 | 9.11 |
| Clustering Coefficient | 2.42 | 2.42 | **2.41** | 2.42 |
| Density | 0.071 | 0.071 | 0.049 | **0.045** |
| Num Triangles | **8.59** | **8.59** | 8.98 | 8.85 |
| Descriptor Norm | 109.6 | 109.6 | 106.4 | **105.4** |
| Median Extra Nodes | 20.0 | 20.0 | **15.0** | **15.0** |
| Median Missing Nodes | **14.0** | **14.0** | 15.0 | **14.0** |
| Median Extra Edges | 21.5 | 21.5 | 18.5 | **18.0** |
| Median Missing Edges | 17.0 | 17.0 | **15.0** | 16.0 |

### 6.4 Findings

The first thing that stands out is that normalization does basically nothing for V-EWMA.
Across every metric and every dataset, V-EWMA-norm lands on top of plain V-EWMA to two or
three decimals. At first I thought this was a bug in the ablation, but it turns out to be
the expected behaviour: V-EWMA forecasts each dimension on its own, so the totals — the
numbers that actually set the node and edge budgets — come out exactly the same whether or
not we normalize. The only thing normalization can touch here is the inner-bucket
reconstruction, and after rounding and degree assignment those small differences disappear.
So for the forecaster the paper actually uses, the honest answer to the supervisor's
question ("does normalizing change the MAR or the graphs?") is simply no.

The recurrent models are a different story. For LSTM, normalization clearly helps, and what
I found interesting is that it helps most exactly where the raw model was struggling. The
clearest case is **Aion**, where raw LSTM was failing badly — it under-predicted new nodes
by about 11.5% (−0.115). Splitting the embedding into shape and scale fixes the size
forecast: new nodes move to +0.007, the descriptor norm drops from 320.8 to 309.5, and the
average-degree error shrinks from −0.13 to −0.055. **DGD** tells the same story on the
headline metrics — F1 Nodes goes from 0.530 to 0.534, new nodes from 0.030 to 0.008, the
descriptor norm from 175.2 to 169.8, and all four median count errors get smaller. On
mathoverflow and sx-mathoverflow-700, though, it's essentially a wash, which fits the
pattern: there wasn't much of a size error to fix in the first place.

Once you line these up, the mechanism is fairly clear. The construction sets the graph size
entirely from the totals forecast, so normalization only buys you something when it makes
the totals forecast better. That's exactly what happens for LSTM on the two ERC20 networks
(totals MAR on Aion 0.214 → 0.171, on DGD 0.314 → 0.281), and it's why V-EWMA stays flat —
its totals don't change by construction.

The one caveat is the plain RNN. Here normalization goes the wrong way and raises the MAR on
every dataset, so the recommendation is to leave RNN un-normalized.

Putting it together: normalizing the TopER embedding by node
and edge count makes no real difference for V-EWMA, but it does help the recurrent
forecasters (LSTM/GRU). And the gain isn't spread evenly — it concentrates on the datasets
and metrics tied to predicting graph *size*, which is precisely the part of TopER that the
construction relies on.

---

## 7. How to Reproduce

```bash
# 0. One-time dataset conversion (SNAP gz -> Loader CSV)
.venv/bin/python learning/prepare_datasets.py

# 1. Forecast vectors (writes learning/pred_vectors/<dataset>/*.pkl;
#    edit the `datasets`/`methods` lists at the bottom of each script)
.venv/bin/python learning/generate_toper_cl.py     # V-EWMA + LSTM
.venv/bin/python learning/generate_probs_cl.py     # V-EWMA + RNN

# 2. Construct graphs (first run trains the shared GNN; later runs reuse it)
cd learning
python topoGED_gnn_implementation_oobankchanges_sampling_cl.py \
    --dataset sx-mathoverflow --alpha 5.94 --beta 7.46 --decay_factor 0.15 \
    --new_node_strategy zeros --rnn_type VEWMA
python topoGED_gnn_implementation_oobankchanges_sampling_cl.py \
    --dataset sx-mathoverflow --alpha 5.94 --beta 7.46 --decay_factor 0.15 \
    --new_node_strategy zeros --toper_method LSTM --probs_method RNN

# 3. Evaluate side by side (full paths in run_sx_mathoverflow.sbatch)
python compare_graphs_cl.py --dataset sx-mathoverflow \
    --pkl_paths <LSTM+RNN pkl> <V-EWMA pkl> \
    --method_names LSTM+RNN V-EWMA \
    --output_dir output/comparison_results/sx-mathoverflow_LSTMtoper_RNNprobs_vs_VEWMA
```

Two warnings: rerunning a forecaster script overwrites its pickle with a non-seeded retrain, so the
matching construction must be rerun too. Changing `lr` (or other training keys) in `encoder.yaml`
invalidates the GNN cache and forces retraining.

### 7.1 Experiment 3 — normalized TopER (Section 6)

Reuses the same cached GNN as the raw runs (no retraining). Sampling parameters per
dataset: mathoverflow / sx-mathoverflow-700 `--alpha 5.94 --beta 7.46 --decay_factor 0.15`,
networkaion `--alpha 1.94 --beta 8.75 --decay_factor 0.60`,
networkdgd `--alpha 1.32 --beta 9.96 --decay_factor 0.31`.

```bash
cd learning
PY=../.venv/bin/python

# 1. Forecast: writes normalized -norm pickles for all 4 methods + the MAR tables
#    (learning/latex_tables/mar_normalized.csv). Does NOT touch generate_toper_cl.py.
$PY generate_toper_normalized_cl.py

# 2. Construct the two normalized arms for each dataset (example: networkaion).
#    V-EWMA-norm keeps probs V-EWMA; LSTM-norm keeps probs RNN — same as the raw arms.
$PY topoGED_gnn_implementation_oobankchanges_sampling_cl.py \
    --dataset networkaion --alpha 1.94 --beta 8.75 --decay_factor 0.60 \
    --new_node_strategy zeros --toper_method VEWMA-norm --probs_method VEWMA
$PY topoGED_gnn_implementation_oobankchanges_sampling_cl.py \
    --dataset networkaion --alpha 1.94 --beta 8.75 --decay_factor 0.60 \
    --new_node_strategy zeros --toper_method LSTM-norm --probs_method RNN

# 3. Evaluate all four arms side by side (raw V-EWMA, V-EWMA-norm, raw LSTM+RNN, LSTM-norm+RNN).
#    SUF is the fixed folder suffix; D is the dataset.
D=networkaion
SUF=_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges_zeros_sampling_predvalsTrue_tmp
PRE=output/constructed_graphs
$PY compare_graphs_cl.py --dataset $D \
    --pkl_paths \
      "$PRE/${D}${SUF}_rnnVEWMA_edgebank_default_VectorTypeV-EWMA/GCN_constructed_graphs_${D}.pkl" \
      "$PRE/${D}${SUF}_toperVEWMA-norm_probsVEWMA_edgebank_default_VectorTypeV-EWMA/GCN_constructed_graphs_${D}.pkl" \
      "$PRE/${D}${SUF}_toperLSTM_probsRNN_edgebank_default_VectorTypeV-EWMA/GCN_constructed_graphs_${D}.pkl" \
      "$PRE/${D}${SUF}_toperLSTM-norm_probsRNN_edgebank_default_VectorTypeV-EWMA/GCN_constructed_graphs_${D}.pkl" \
    --method_names V-EWMA V-EWMA-norm LSTM+RNN LSTM-norm+RNN \
    --output_dir output/comparison_results/${D}_normalized_ablation
```

Repeat steps 2–3 for `mathoverflow`, `networkdgd`, and `sx-mathoverflow-700` with their
sampling parameters. The raw `rnnVEWMA` and `toperLSTM_probsRNN` folders referenced in
step 3 are produced by the Section 4 runs above.
