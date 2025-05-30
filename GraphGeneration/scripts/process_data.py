import networkx as nx
import numpy as np

def modifyGraphIds(graphs, thresholds):
    '''
    For the target graphs, modify their ids to start at 0 for an instance of a node, then increment throughout the graphs
    
    Args:
        graphs (list(nx.Graph)): A list of graphs to modify
        
    Returns:
        graphs (list(nx.Graph)): The modified graphs (operations performed in-place)       
    '''
    # This dictionary will store the mapping of original node IDs to new node IDs
    node_mapping = {}
    new_id = 0
    updated_graphs = []

    # Iterate over all graphs in the list of lists (where each graph is a subgraph in the list)
    # First pass: assign a new ID to every unique node
    for graph_list in graphs:
        updated_sublist = []
        for graph in graph_list:
            curr_mapping = {}  # Mapping applies to this specific graph

            for node in graph.nodes:
                # Ensure that 'feat' exists and is properly initialized
                if 'feat' not in graph.nodes[node]:
                    graph.nodes[node]['feat'] = {}

                # Mark the node as new or old
                if node not in node_mapping:
                    node_mapping[node] = new_id
                    new_id += 1
                    graph.nodes[node]['feat']['type'] = 1  # Node is new
                else:
                    graph.nodes[node]['feat']['type'] = 0  # Node is old

                # Map the node and update the ID in the feature dictionary
                curr_mapping[node] = node_mapping[node]
                graph.nodes[node]['feat']['id'] = node_mapping[node]

                node_degree = graph.degree(node)  # Current node's degree

                # If thresholds are available, calculate the max degree based on thresholds
                if np.any(thresholds):
                    graph.nodes[node]['feat']['currDegree'] = node_degree
                    graph.nodes[node]['feat']['maxDegree'] = next((t for t in thresholds if node_degree <= t), thresholds[-1])
                else:
                    # If no thresholds, use degree as maxDegree
                    graph.nodes[node]['feat']['currDegree'] = node_degree
                    graph.nodes[node]['feat']['maxDegree'] = node_degree

            # Relabel the graph nodes according to the new IDs
            relabeled_graph = nx.relabel_nodes(graph, curr_mapping, copy=True)

            # Preserve features for relabeled nodes
            for old_node, new_node in curr_mapping.items():
                relabeled_graph.nodes[new_node]['feat'] = graph.nodes[old_node]['feat'].copy()
                # print(f"Old Node: {old_node}, Features: {graph.nodes[old_node]['feat']}")
                # print(f"New Node: {new_node}, Features: {relabeled_graph.nodes[new_node]['feat']}")

            updated_sublist.append(relabeled_graph)
        updated_graphs.append(updated_sublist)

    return updated_graphs, len(node_mapping)

def build_edgebanks_from_start(graphs):
    """
    Build the edgebanks for each graph in graphs, stores all edges from graph i-1 in each index i
    
    Args:
        graphs (list(nx.Graph)): A list of nx Graphs that we will build our edgebanks from
        
    Returns:
        edgebanks (list(dict)): A list of dictionary edgebanks that store all edges from the previous graphs in each index
    """
    edgebanks = [{}]  # Initialize an empty list for edgebanks

    # Loop over all graphs (starting from the second graph)
    for i in range(1, len(graphs)):
        curr_edgebank = {}

        # Add edges from all previous graphs (not the current graph)
        for j in range(i):  # Loop through all previous graphs (graphs 0 to i-1)
            for u, v in graphs[j][-1].edges():  # Accessing the graph directly
                u_key = u
                v_key = v
                curr_edgebank.setdefault(u_key, []).append(v_key)  # Add edge from u to v

        edgebanks.append(curr_edgebank)  # Append the current edgebank to the list

    return edgebanks

