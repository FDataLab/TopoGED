"""
Compare two sets of constructed graphs (e.g. LSTM vs V-EWMA) using the exact
same metrics as evaluate_graphs_cl.py / Evaluator.py.

Usage (from ~/TopoGED/learning/):
    python compare_graphs_cl.py \
        --dataset CollegeMsg \
        --pkl_paths \
            "output/constructed_graphs/<LSTM_folder>/GCN_constructed_graphs_CollegeMsg.pkl" \
            "output/constructed_graphs/<VEWMA_folder>/GCN_constructed_graphs_CollegeMsg.pkl" \
        --method_names LSTM V-EWMA \
        --output_dir "output/comparison_results/CollegeMsg_LSTM_vs_VEWMA"
"""

import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(_project_root)
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, 'GraphGeneration', 'scripts'))

from utils.loader import Loader
_learning = os.path.join(_project_root, 'learning')
Loader.output_dir   = os.path.join(_learning, 'datasets', 'cached')
Loader.edgelist_dir = os.path.join(_learning, 'datasets', 'edgelist')
Loader.label_dir    = os.path.join(_learning, 'datasets', 'labels')

from GraphGeneration.scripts.load_data import load_data
from GraphGeneration.scripts.process_data import modifyGraphIds
from utils.embedding_methods.degree import EmbedDegree


# ---------------------------------------------------------------------------
# Metric helpers (mirror of Evaluator.py — no grakel dependency)
# ---------------------------------------------------------------------------

def _prf(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0, 0.0, 0.0
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return float(p), float(r), float(f)


def evaluate_node_selection(sorted_nodes_pred, sorted_nodes_true):
    """Mirrors Evaluator.evaluate_node_selection."""
    pred_old = set(sorted_nodes_pred['old_nodes'])
    pred_new = set(sorted_nodes_pred['new_nodes'])
    true_old = set(sorted_nodes_true['old_nodes'])
    true_new = set(sorted_nodes_true['new_nodes'])

    def prf_sets(pred_set, true_set):
        correct = pred_set & true_set
        prec = len(correct) / len(pred_set) if pred_set else 0.0
        rec  = len(correct) / len(true_set) if true_set else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    p_old, r_old, f_old = prf_sets(pred_old, true_old)
    p_new, r_new, f_new = prf_sets(pred_new, true_new)
    p_all, r_all, f_all = prf_sets(pred_old | pred_new, true_old | true_new)

    num_new_true = len(true_new) if true_new else 1.0
    new_node_pct_diff = (len(pred_new) - len(true_new)) / num_new_true

    return {
        'Precision_Old': p_old, 'Recall_Old': r_old, 'F1_Old': f_old,
        'Precision_New': p_new, 'Recall_New': r_new, 'F1_New': f_new,
        'Precision_All': p_all, 'Recall_All': r_all, 'F1_All': f_all,
        'Num_Old_Predicted': len(pred_old), 'Num_New_Predicted': len(pred_new),
        'Num_Old_True': len(true_old),       'Num_New_True': len(true_new),
        'New_Node_Pct_Diff': new_node_pct_diff,
    }


def evaluate_graph_edges(pred_graph, true_graph, is_directed=False, edgebank=None):
    """Mirrors Evaluator.evaluate_graph_edges."""
    if is_directed:
        pred_edges = set(pred_graph.edges())
        true_edges = set(true_graph.edges())
    else:
        pred_edges = {tuple(sorted(e)) for e in pred_graph.edges()}
        true_edges = {tuple(sorted(e)) for e in true_graph.edges()}

    if edgebank is not None:
        pred_bank   = {e for e in pred_edges if e[1] in edgebank.get(e[0], set())}
        pred_nobank = pred_edges - pred_bank
        true_bank   = {e for e in true_edges if e[1] in edgebank.get(e[0], set())}
        true_nobank = true_edges - true_bank

        def _metrics(pred_s, true_s):
            tp = len(pred_s & true_s)
            fp = len(pred_s - true_s)
            fn = len(true_s - pred_s)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return prec, rec, f1, tp, fp, fn

        p_b,  r_b,  f_b,  tp_b,  fp_b,  fn_b  = _metrics(pred_bank,   true_bank)
        p_nb, r_nb, f_nb, tp_nb, fp_nb, fn_nb  = _metrics(pred_nobank, true_nobank)

        return {
            'Precision_bank': p_b,  'Recall_bank': r_b,  'F1_bank': f_b,
            'TP_bank': tp_b, 'FP_bank': fp_b, 'FN_bank': fn_b,
            'Precision_nobank': p_nb, 'Recall_nobank': r_nb, 'F1_nobank': f_nb,
            'TP_nobank': tp_nb, 'FP_nobank': fp_nb, 'FN_nobank': fn_nb,
        }
    else:
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {'Precision': prec, 'Recall': rec, 'F1': f1, 'TP': tp, 'FP': fp, 'FN': fn}


def evaluate_edge_types(pred_graph, true_graph, old_nodes_true, old_nodes_pred, is_directed=False):
    """
    Evaluate o-n and n-n edge counts (TP/FP/FN) — mirrors evaluateEdgesNew + on/nn CSV logic.
    Returns raw TP/FP/FN for on and nn subgraphs.
    """
    new_nodes_pred = set(pred_graph.nodes()) - set(old_nodes_pred)
    new_nodes_true = set(true_graph.nodes()) - set(old_nodes_true)

    def _edge_set(g, a_nodes, b_nodes, directed):
        if directed:
            return {(u, v) for u, v in g.edges() if u in a_nodes and v in b_nodes}
        else:
            edges = set()
            for u, v in g.edges():
                if (u in a_nodes and v in b_nodes) or (u in b_nodes and v in a_nodes):
                    edges.add(tuple(sorted((u, v))))
            return edges

    pred_on = _edge_set(pred_graph, old_nodes_pred, new_nodes_pred, is_directed)
    true_on = _edge_set(true_graph, old_nodes_true, new_nodes_true, is_directed)
    pred_nn = _edge_set(pred_graph, new_nodes_pred, new_nodes_pred, is_directed)
    true_nn = _edge_set(true_graph, new_nodes_true, new_nodes_true, is_directed)

    def counts(pred_s, true_s):
        tp = len(pred_s & true_s)
        fp = len(pred_s - true_s)
        fn = len(true_s - pred_s)
        return tp, fp, fn

    tp_on, fp_on, fn_on = counts(pred_on, true_on)
    tp_nn, fp_nn, fn_nn = counts(pred_nn, true_nn)

    return {
        'TP_on': tp_on, 'FP_on': fp_on, 'FN_on': fn_on,
        'TP_nn': tp_nn, 'FP_nn': fp_nn, 'FN_nn': fn_nn,
    }


def evaluate_structure(graph):
    """Mirrors Evaluator.evaluateSingleStructure (subset used in table building)."""
    if graph.number_of_nodes() == 0:
        return {k: 0.0 for k in ['Average Node Degree', 'Unique Degree Count', 'Degree Centrality',
                                   'Assortivity Coefficient', 'Clustering Coefficient', 'Density',
                                   'Number of Triangles']}
    ug = graph.to_undirected() if graph.is_directed() else graph
    try:
        assortativity = nx.degree_assortativity_coefficient(ug)
    except Exception:
        assortativity = float('nan')
    try:
        triangles = sum(nx.triangles(ug).values()) // 3
    except Exception:
        triangles = 0
    degrees = [d for _, d in ug.degree()]
    return {
        'Average Node Degree':   float(np.mean(degrees)) if degrees else 0.0,
        'Unique Degree Count':   float(len(set(degrees))),
        'Degree Centrality':     float(np.mean(list(nx.degree_centrality(ug).values()))),
        'Assortivity Coefficient': float(assortativity),
        'Clustering Coefficient':  float(nx.average_clustering(ug)),
        'Density':               float(nx.density(ug)),
        'Number of Triangles':   float(triangles),
    }


def evaluate_toper(pred_graph, true_graph, embedder):
    """Mirrors Evaluator.evaluateTopER applied to a single pair."""
    if pred_graph.number_of_nodes() == 0:
        pred_vec = np.zeros(20)
    else:
        emb, _, _ = embedder.process_graphs_for_embeddings([pred_graph])
        pred_vec = np.array(emb[0])
    true_emb, _, _ = embedder.process_graphs_for_embeddings([true_graph])
    true_vec = np.array(true_emb[0])

    l2 = float(np.linalg.norm(pred_vec - true_vec))
    diff = pred_vec - true_vec
    result = {'l2_norm': l2}
    for i in range(10):
        result[f'node_diff_{i+1}'] = float(diff[2 * i])
        result[f'edge_diff_{i+1}'] = float(diff[2 * i + 1])
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        pred_graphs, _, node_types_list, _ = data
        return pred_graphs, node_types_list
    raise TypeError(f"Unexpected pkl format: {type(data)}")


def load_true_graphs(dataset, num_pred):
    _, _, thresholds, target_graphs = load_data(
        dataset, '', '', '', 'all',
        use_predicted=False, num_buckets=10, use_test_style=None
    )
    target_graphs, _ = modifyGraphIds(target_graphs, thresholds, 10000)
    target_graphs_last = [inner[-1] for inner in target_graphs if inner]
    true_graphs = target_graphs_last[-num_pred:]
    base_graphs = target_graphs_last[: len(target_graphs_last) - num_pred]
    return true_graphs, base_graphs


def build_old_nodes_info(true_graphs, base_graphs):
    """
    Mirrors evaluate_graphs_cl.py __init__ logic for sorted_nodes_true.
    Returns list of dicts with 'old_nodes' and 'new_nodes' per snapshot.
    """
    old_nodes = set()
    for g in base_graphs:
        old_nodes.update(g.nodes())

    sorted_nodes = []
    for g_true in true_graphs:
        curr = set(g_true.nodes())
        sorted_nodes.append({
            'old_nodes': old_nodes & curr,
            'new_nodes': curr - old_nodes,
            'history': set(old_nodes),
        })
        old_nodes.update(curr)
    return sorted_nodes


# ---------------------------------------------------------------------------
# Per-snapshot evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(pred_graphs, true_graphs, sorted_nodes_true, is_directed=False):
    """
    Run all metrics per snapshot. Returns a dict of lists (one entry per snapshot).
    """
    embedder = EmbedDegree(include_weights=False)
    edgebank = {}

    records = {
        'node': [], 'structure_true': [], 'structure_pred': [],
        'toper': [], 'all_edges': [], 'old_edges': [], 'on_nn': [],
    }

    for i, (pred_g, true_g, sn_true) in enumerate(zip(pred_graphs, true_graphs, sorted_nodes_true)):
        pred_g = pred_g.to_undirected() if (not is_directed and pred_g.is_directed()) else pred_g
        true_g = true_g.to_undirected() if (not is_directed and true_g.is_directed()) else true_g

        # --- classify pred node types against the running history of all
        # nodes ever seen, mirroring evaluate_graphs.py (a predicted node is
        # "old" if it appeared in any earlier snapshot, regardless of whether
        # it reappears in the true graph) ---
        pred_old = sn_true['history'] & set(pred_g.nodes())
        pred_new = set(pred_g.nodes()) - pred_old
        sn_pred = {'old_nodes': pred_old, 'new_nodes': pred_new}

        records['node'].append(evaluate_node_selection(sn_pred, sn_true))
        records['structure_true'].append(evaluate_structure(true_g))
        records['structure_pred'].append(evaluate_structure(pred_g))
        records['toper'].append(evaluate_toper(pred_g, true_g, embedder))

        # All edges (no edgebank split)
        records['all_edges'].append(evaluate_graph_edges(pred_g, true_g, is_directed=is_directed))

        # Old-node subgraph edges (bank / nobank split)
        pred_old_g = pred_g.subgraph(sn_pred['old_nodes']).copy()
        true_old_g = true_g.subgraph(sn_true['old_nodes']).copy()
        records['old_edges'].append(evaluate_graph_edges(
            pred_old_g, true_old_g, is_directed=is_directed, edgebank=edgebank
        ))

        # o-n and n-n edge counts
        records['on_nn'].append(evaluate_edge_types(
            pred_g, true_g, sn_true['old_nodes'], sn_pred['old_nodes'], is_directed=is_directed
        ))

        # Update edgebank with true edges
        for u, v in true_g.edges():
            edgebank.setdefault(u, set()).add(v)
            if not is_directed:
                edgebank.setdefault(v, set()).add(u)

    return {k: pd.DataFrame(v) for k, v in records.items()}


# ---------------------------------------------------------------------------
# Aggregate into summary row (mirrors formTables / construct_ablation_tables)
# ---------------------------------------------------------------------------

def aggregate(dfs):
    node       = dfs['node']
    str_true   = dfs['structure_true']
    str_pred   = dfs['structure_pred']
    toper      = dfs['toper']
    all_edges  = dfs['all_edges']
    old_edges  = dfs['old_edges']
    on_nn      = dfs['on_nn']

    def mean(df, col):
        return df[col].replace([np.inf, -np.inf], np.nan).mean()

    def rel_err(t, p):
        denom = t.copy().astype(float)
        denom[denom == 0] = 1.0
        return ((p - t) / denom).mean()

    structure_map = {
        'Avg Node Degree':         'Average Node Degree',
        'Unique Degree Count':     'Unique Degree Count',
        'Degree Centrality':       'Degree Centrality',
        'Assortativity Coefficient': 'Assortivity Coefficient',
        'Clustering Coefficient':  'Clustering Coefficient',
        'Density':                 'Density',
        'Num Triangles':           'Number of Triangles',
    }

    row = {}

    # Node metrics
    row['Precision Nodes']     = mean(node, 'Precision_All')
    row['Recall Nodes']        = mean(node, 'Recall_All')
    row['F1 Nodes']            = mean(node, 'F1_All')
    row['Precision Old Nodes'] = mean(node, 'Precision_Old')
    row['Recall Old Nodes']    = mean(node, 'Recall_Old')
    row['F1 Old Nodes']        = mean(node, 'F1_Old')
    denom_new = node['Num_New_True'].astype(float).copy()
    denom_new[denom_new == 0] = 1.0
    row['New Nodes Predicted'] = ((node['Num_New_Predicted'] - node['Num_New_True']) / denom_new).mean()

    # Structure metrics (relative error)
    for col, csv_col in structure_map.items():
        row[col] = rel_err(str_true[csv_col], str_pred[csv_col])

    # TopER metrics
    extra_n = toper[toper['node_diff_10'] > 0]['node_diff_10']
    miss_n  = toper[toper['node_diff_10'] < 0]['node_diff_10']
    extra_e = toper[toper['edge_diff_10'] > 0]['edge_diff_10']
    miss_e  = toper[toper['edge_diff_10'] < 0]['edge_diff_10']
    row['Median Extra Nodes']   = extra_n.median() if not extra_n.empty else 0
    row['Median Missing Nodes'] = abs(miss_n.median()) if not miss_n.empty else 0
    row['Median Extra Edges']   = extra_e.median() if not extra_e.empty else 0
    row['Median Missing Edges'] = abs(miss_e.median()) if not miss_e.empty else 0
    row['Descriptor Norm']      = mean(toper, 'l2_norm')

    # Edge metrics
    row['Edge Precision'] = mean(all_edges, 'Precision')
    row['Edge Recall']    = mean(all_edges, 'Recall')
    row['Edge F1']        = mean(all_edges, 'F1')

    row['oo-bank Precision']   = mean(old_edges, 'Precision_bank')
    row['oo-bank Recall']      = mean(old_edges, 'Recall_bank')
    row['oo-bank F1']          = mean(old_edges, 'F1_bank')
    row['oo-nobank Precision'] = mean(old_edges, 'Precision_nobank')
    row['oo-nobank Recall']    = mean(old_edges, 'Recall_nobank')
    row['oo-nobank F1']        = mean(old_edges, 'F1_nobank')

    # o-n and n-n percent diff (positive = over-predicted, negative = under-predicted)
    num_true_on  = on_nn['TP_on'] + on_nn['FN_on']
    num_pred_on  = on_nn['TP_on'] + on_nn['FP_on']
    denom_on     = num_true_on.astype(float).copy(); denom_on[denom_on == 0] = 1.0
    row['Num o-n Predicted'] = ((num_pred_on - num_true_on) / denom_on).mean()

    num_true_nn  = on_nn['TP_nn'] + on_nn['FN_nn']
    num_pred_nn  = on_nn['TP_nn'] + on_nn['FP_nn']
    denom_nn     = num_true_nn.astype(float).copy(); denom_nn[denom_nn == 0] = 1.0
    row['Num n-n Predicted'] = ((num_pred_nn - num_true_nn) / denom_nn).mean()

    return row


# ---------------------------------------------------------------------------
# LaTeX table builder (mirrors makeTable in evaluate_graphs_cl.py)
# ---------------------------------------------------------------------------

MIN_BEST_COLS = {
    'Avg Node Degree', 'Unique Degree Count', 'Degree Centrality',
    'Assortativity Coefficient', 'Clustering Coefficient', 'Density',
    'Num Triangles', 'Descriptor Norm',
    'Median Extra Nodes', 'Median Missing Nodes',
    'Median Extra Edges', 'Median Missing Edges',
    'Num o-n Predicted', 'Num n-n Predicted', 'New Nodes Predicted',
}

INT_COLS = {'Median Extra Nodes', 'Median Missing Nodes', 'Median Extra Edges', 'Median Missing Edges'}


def make_latex_table(summary_df, caption, label, table_type=None):
    methods = list(summary_df.index)
    metrics = list(summary_df.columns)

    col_fmt = 'l ' + ' '.join(['c'] * len(methods))
    lines = [
        r'\begin{tabular}{' + col_fmt + '}',
        r'\toprule',
        r'\toprule',
        r'\textbf{Metric} & ' + ' & '.join(fr'\textbf{{{m}}}' for m in methods) + r' \\ \midrule',
    ]

    def fmt_val(v, col):
        if pd.isna(v):
            return '$-$'
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if col in INT_COLS:
            return f'{int(round(fv))}'
        return f'{fv:.4f}'

    for metric in metrics:
        vals = {}
        for m in methods:
            try:
                vals[m] = float(summary_df.at[m, metric])
            except Exception:
                vals[m] = float('nan')

        valid_vals = {m: v for m, v in vals.items() if not np.isnan(v)}
        if valid_vals:
            if metric in MIN_BEST_COLS:
                sorted_keys = sorted(valid_vals, key=lambda m: abs(valid_vals[m]))
            else:
                sorted_keys = sorted(valid_vals, key=lambda m: valid_vals[m], reverse=True)
            best   = sorted_keys[0] if len(sorted_keys) > 0 else None
            second = sorted_keys[1] if len(sorted_keys) > 1 else None
        else:
            best, second = None, None

        cells = [metric.replace('_', r'\_')]
        for m in methods:
            v = vals[m]
            s = fmt_val(v, metric)
            if m == best and s not in ('0', '0.0000'):
                cells.append(r'$\mathbf{' + s + '}$')
            elif m == second and s not in ('0', '0.0000'):
                cells.append(r'$\underline{' + s + '}$')
            else:
                cells.append(f'${s}$')
        lines.append(' & '.join(cells) + r' \\')

        if table_type == 'structure' and metric == 'Num Triangles':
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\bottomrule', r'\end{tabular}']
    lines += [
        r'\begin{center}',
        r'\vspace{-2pt}',
    ]
    if table_type == 'nodes':
        lines.append(r'{\small For all metrics higher is better. \par}')
    elif table_type == 'structure':
        lines.append(r'{\small For all metrics closer to 0 is better. \par}')
    elif table_type == 'edges':
        lines.append(r'{\small For \texttt{Num o-n Predicted} and \texttt{Num n-n Predicted} closer to 0 is better; for others, higher is better. \par}')
    lines += [
        r'{\small \textbf{Bold} indicates best, the second-best is \underline{underlined}. \par}',
        r'\end{center}',
    ]

    full = [
        r'\begin{table}[ht]',
        fr'\caption{{{caption}}}',
        fr'\label{{{label}}}',
        r'\centering',
    ] + lines + [r'\end{table}']

    return '\n'.join(full)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

NODE_COLS = [
    'Precision Nodes', 'Recall Nodes', 'F1 Nodes',
    'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes',
    'New Nodes Predicted',
]
STRUCTURE_COLS = [
    'Avg Node Degree', 'Unique Degree Count', 'Degree Centrality',
    'Assortativity Coefficient', 'Clustering Coefficient', 'Density',
    'Num Triangles', 'Descriptor Norm',
    'Median Extra Nodes', 'Median Missing Nodes',
    'Median Extra Edges', 'Median Missing Edges',
]
EDGE_COLS = [
    'oo-bank Precision', 'oo-bank Recall', 'oo-bank F1',
    'oo-nobank Precision', 'oo-nobank Recall', 'oo-nobank F1',
    'Num o-n Predicted', 'Num n-n Predicted',
    'Edge Precision', 'Edge Recall', 'Edge F1',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pkl_paths', nargs='+', required=True,
                        help='One or more paths to GCN_constructed_graphs_*.pkl (relative to learning/)')
    parser.add_argument('--method_names', nargs='+', required=True,
                        help='Display name for each pkl (same order as --pkl_paths)')
    parser.add_argument('--output_dir', type=str, default='output/comparison_results/default',
                        help='Directory to save CSV files and LaTeX tables (relative to learning/)')
    args = parser.parse_args()

    assert len(args.pkl_paths) == len(args.method_names), \
        '--pkl_paths and --method_names must have the same number of entries'

    out_dir = os.path.join(_learning, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load first pkl to know how many test snapshots we have
    first_pkl = os.path.join(_learning, args.pkl_paths[0])
    pred0, _ = load_pkl(first_pkl)
    num_pred = len(pred0)

    print(f'Loading true graphs for {args.dataset} ({num_pred} test snapshots)...')
    true_graphs, base_graphs = load_true_graphs(args.dataset, num_pred)
    sorted_nodes_true = build_old_nodes_info(true_graphs, base_graphs)

    # Summary rows indexed by method name
    summary_rows = {}

    for pkl_rel, name in zip(args.pkl_paths, args.method_names):
        pkl_path = os.path.join(_learning, pkl_rel)
        print(f'\nEvaluating {name} from {pkl_path} ...')
        pred_graphs, node_types_list = load_pkl(pkl_path)
        assert len(pred_graphs) == num_pred, \
            f'{name}: expected {num_pred} graphs, got {len(pred_graphs)}'

        dfs = run_evaluation(pred_graphs, true_graphs, sorted_nodes_true, is_directed=False)

        # Save per-snapshot CSVs
        method_dir = os.path.join(out_dir, name)
        os.makedirs(method_dir, exist_ok=True)
        for key, df in dfs.items():
            df.to_csv(os.path.join(method_dir, f'{key}.csv'), index=False)

        summary_rows[name] = aggregate(dfs)
        print(f'  done. Edge F1={summary_rows[name]["Edge F1"]:.4f}  '
              f'F1 Nodes={summary_rows[name]["F1 Nodes"]:.4f}  '
              f'Descriptor Norm={summary_rows[name]["Descriptor Norm"]:.4f}')

    # Build summary DataFrames
    all_cols = NODE_COLS + STRUCTURE_COLS + EDGE_COLS
    summary = pd.DataFrame(summary_rows).T[all_cols]
    summary.to_csv(os.path.join(out_dir, 'summary.csv'))
    print(f'\nSummary saved to {os.path.join(out_dir, "summary.csv")}')
    print(summary.to_string())

    # Split into three tables
    node_df      = summary[NODE_COLS]
    structure_df = summary[STRUCTURE_COLS]
    edge_df      = summary[EDGE_COLS]

    ds = args.dataset
    tables = [
        (node_df,      'nodes',     f'Node Evaluation: {ds}',      f'tab:{ds}_nodes'),
        (structure_df, 'structure', f'Structure Evaluation: {ds}',  f'tab:{ds}_structure'),
        (edge_df,      'edges',     f'Edge Evaluation: {ds}',       f'tab:{ds}_edges'),
    ]
    for df, ttype, caption, label in tables:
        tex = make_latex_table(df, caption, label, table_type=ttype)
        tex_path = os.path.join(out_dir, f'table_{ttype}.tex')
        with open(tex_path, 'w') as f:
            f.write(tex)
        print(f'LaTeX {ttype} table → {tex_path}')


if __name__ == '__main__':
    main()
