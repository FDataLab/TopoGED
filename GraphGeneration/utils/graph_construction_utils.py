def get_node_features(graph, thresholds, embedding, old_nodes, new_nodes):
    """
    Assign the maximum degree of a node, either using its last seen degree (if args.oldDegree == True) or randomly giving it one
    
    Args:
        graph (nx.DiGraph): The Graph we are attempting to construct, we assign node features here
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
        embedding (list): The current TopER graph embedding vector used to figure out how many nodes have a degree
        old_nodes (list): The list of old nodes that are in the graph
        new_nodes (list): The list of new nodes that are in the graph
        
    Returns:
        None
    """
    degree_counts = [embedding[i][0] for i in range(0, len(embedding))]
    
    degree_dict = {thresholds[i]: degree_counts[i] for i in range(len(thresholds))}
    
    degree_assignment = []  # This will store the assigned degrees
    
    for degree, count in degree_dict.items():
        degree_assignment.extend([degree] * count)
        
    random.shuffle(degree_assignment)
    
    if args.oldDegree == 'True':
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
            
            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
        
        # Give the node a random new degree    
        for node in new_nodes:
            assigned_degree = degree_assignment.pop()

            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
            
    else:
        for i, node in enumerate(graph.nodes):        
            # Assign features to the node as an attribute
            graph.nodes[node]['feat']['currDegree'] = 0  # Starts at 0
            graph.nodes[node]['feat']['maxDegree'] = degree_assignment[i]
 