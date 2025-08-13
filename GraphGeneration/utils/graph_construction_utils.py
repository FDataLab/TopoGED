import random
import numpy as np 
import networkx as nx

def compute_reappearance_probabilities(graphs, t_curr, decay_factor=3.0, alpha=1.0, epsilon=1e-8):
    """
    Compute the probability for each node to reappear given how long ago it was seen and its latest degree
    Nodes of higher degree, and nodes seen more recently are preferred
    
    Args:
        graphs: The snapshots we have observed up to the t_curr
        t_curr (int): The current graph number we are on, used to compute probabilities
        decay_factor (float): How quickly the recency of a node decays. Higher means that the nodes seen long ago decay slower
        alpha (float): Our decay constant, controls how influential degree is (alpha > 1 means that it prefers degree, alpha < 1 means that it matters less)
        epsilon (float): Prevents having 0 probabilities for a node, and thus prevents numpy errors later on
    
    Returns:
        probs (dict):  A dictionary of {node_id: percent probability} probabilities for each node in nodes
    """
    nodes = dict()
    # Create the nodes dict degree history
    # nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
    for t, G in enumerate(graphs):
        for node in G.nodes():
            degree = G.degree(node)
            nodes[node] = (t, degree) 
    
    if not nodes:
        return {}
    
    max_degree = max(degree for _, (_, degree) in nodes.items())

    probs = {}
    for node_id, (last_seen, degree) in nodes.items():
        recency_score = np.exp(-max(0, t_curr - last_seen) / decay_factor)
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else epsilon
        raw_score = recency_score * degree_score
        probs[node_id] = max(raw_score, epsilon)  # Apply epsilon floor to avoid exact 0

    # Normalize to make a valid probability distribution
    total = sum(probs.values())
    for node in probs:
        probs[node] /= total

    return probs
       
def get_node_features(constructing_graph, previous_graphs, thresholds, embedding, old_nodes, new_nodes):
    """
    Assign the maximum degree of a node, either using its last seen degree (if args.oldDegree == True) or randomly giving it one
    
    Args:
        constructing_graph (nx.DiGraph): The Graph we are attempting to construct, we assign node features here
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
        embedding (list): The current TopER graph embedding vector used to figure out how many nodes have a degree
        old_nodes (list): The list of old nodes that are in the graph
        new_nodes (list): The list of new nodes that are in the graph
        
    Returns:
        None
    """
    
    existing_nodes = dict()
    
    # Add the nodes to the graph
    for node in old_nodes:
        constructing_graph.add_node(node)
        feature_dict_old = {'id': node}  
        constructing_graph.nodes[node]['feat'] = feature_dict_old
    for node in new_nodes:
        constructing_graph.add_node(node)
        feature_dict_new = {'id': node}  
        constructing_graph.nodes[node]['feat'] = feature_dict_new
        
    # Create the nodes dict degree history
    # nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
    for t, G in enumerate(previous_graphs):
        for node in G.nodes():
            degree = G.degree(node)
            existing_nodes[node] = (t, degree)
    
    degree_counts = [embedding[i][0] for i in range(0, len(embedding))]
    
    degree_dict = {thresholds[i]: degree_counts[i] for i in range(len(thresholds))}
    
    degree_assignment = []  # This will store the assigned degrees
    
    for degree, count in degree_dict.items():
        degree_assignment.extend([degree] * count)
        
    random.shuffle(degree_assignment)
    

    for node in old_nodes:
        old_degree = existing_nodes[node][1]

        # Find the smallest degree in degree_assignment ≥ old_degree
        suitable_degrees = [d for d in degree_assignment if d >= old_degree]
        if suitable_degrees:
            assigned_degree = min(suitable_degrees)
        else:
            assigned_degree = degree_assignment.pop()

        if not degree_assignment:
            pass
        else:
            degree_assignment.remove(assigned_degree)
        
        constructing_graph.nodes[node]['feat']['currDegree'] = 0
        constructing_graph.nodes[node]['feat']['maxDegree'] = assigned_degree
    
    # Give the node a random new degree    
    for node in new_nodes:
        assigned_degree = degree_assignment.pop()

        constructing_graph.nodes[node]['feat']['currDegree'] = 0
        constructing_graph.nodes[node]['feat']['maxDegree'] = assigned_degree
 
    return constructing_graph
    
def update_degrees(graph: nx.DiGraph):
    """
    After updating the graph, between edge types, update the nodes current degree feature
    
    Args:
        graph (nx.DiGraph): The current graph in construction
        
    Returns:
        None
    """
    for node in graph.nodes(data=False):
        if 'feat' not in graph.nodes[node]:
            graph.nodes[node]['feat'] = {'id': node}  
            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
        else:
            graph.nodes[node]['feat']['currDegree'] = graph.degree(node)
     