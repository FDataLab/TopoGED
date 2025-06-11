import pandas as pd
import networkx as nx

def load_oo_graph(graph_name):
    # Load data
    df = pd.read_csv(rf'C:\Users\duykh\OneDrive\Documents\GNN Research\Topological-Temporal-GFM\Topological-Temporal-GFM\data\input\raw\edgelist\{graph_name}.txt')

    # Ensure Snapshot column is sorted
    snapshots = sorted(df["Snapshot"].unique())

    # Store graphs
    oo_graphs = {}
    known_nodes = set()

    for snapshot in snapshots:
        df_snap = df[df["Snapshot"] == snapshot]

        # Filter edges where both nodes are known
        old_edges = df_snap[
            df_snap["from"].isin(known_nodes) &
            df_snap["to"].isin(known_nodes)
        ]

        # Create graph for this snapshot
        G = nx.from_pandas_edgelist(old_edges, source="from", target="to", edge_attr=True)
        oo_graphs[snapshot] = G

        # Update known nodes set with all nodes in this snapshot (regardless of old/new)
        known_nodes.update(df_snap["from"].tolist())
        known_nodes.update(df_snap["to"].tolist())
        
    return oo_graphs

# load_oo_graph("CollegeMsg")