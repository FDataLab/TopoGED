import networkx as nx


class Probs():
    def __init__(self):
        pass 


    def gen_probs(self, num_graphs_back: int, graphs: list):
        probabilities = []

        for i in range(num_graphs_back, len(graphs)):
            curr_graphs = graphs[i - num_graphs_back : i]  # Get groups of size num_graphs_back
            target_graph = graphs[i]

            # Generate our probabilities
            probs = [
                self.prob_old_nodes(target_graph, curr_graphs),
                self.prob_new_nodes(target_graph, curr_graphs),
                self.prob_oo(target_graph, curr_graphs),
                self.prob_2(target_graph, curr_graphs),
                self.prob_3(target_graph, curr_graphs),
                self.prob_4(target_graph, curr_graphs)
            ]
            
            probabilities.append(probs)

        return probabilities


    # Edges that already existed in previous graphs
    def prob_oo(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances an edge reappearing

        prev_edges = set().intersection(*(graph.edges() for graph in prev_graphs))

        for edge in target_graph.edges():
            if edge in prev_edges:
                count += 1

        return count 
    

    # New edge because of one new node
    def prob_on(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances of a new edge because of one new node

        # Not sure if i want to use intersection or union here when considering multiple graphs
        prev_nodes = set().intersection(*(graph.nodes() for graph in prev_graphs))
        nodes_in_curr_graph = set(target_graph.nodes())
        new_nodes = nodes_in_curr_graph - prev_nodes


        for edge in target_graph.edges():
            # Get nodes
            node_1 = edge[0]
            node_2 = edge[1]

            if((node_1 in prev_nodes and node_2 in new_nodes) or (node_1 in new_nodes and node_2 in prev_nodes)):
                count += 1

        return count


    # New edge between already existing nodes that did not previously have an edge
    def prob_oon(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances of a new edge between existing nodes

        # Get all nodes and edges that previously existed
        prev_nodes = set().intersection(*(graph.nodes() for graph in prev_graphs))
        prev_edges = set().union(*(graph.edges() for graph in prev_graphs))

        for edge in target_graph.edges():
            # Get nodes
            node_1 = edge[0]
            node_2 = edge[1]

            # If the nodes previously existed, but did not make an edge, add to the count
            if(node_1 in prev_nodes and node_2 in prev_nodes) and (edge not in prev_edges):
                count += 1
                
        return count


    # New edge between two new nodes
    def prob_nn(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0

        # Get all nodes that previously existed
        prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])

        for node_1, node_2 in target_graph.edges():
            if(node_1 not in prev_nodes and node_2 not in prev_nodes):
                count += 1

        return count 