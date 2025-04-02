import networkx as nx


class Probs():
    def __init__(self):
        pass 


    def gen_probs(self, num_graphs_back: int, graphs: list):
        probabilities = []

        for i in range(num_graphs_back, len(graphs), 1):
            curr_probs = []
            curr_graphs = graphs[i - num_graphs_back : i]  # Get groups of size num_graphs_back

            # Generate our probabilities
            curr_probs.append(self.prob_1(graphs[i], curr_graphs))
            curr_probs.append(self.prob_2(graphs[i], curr_graphs))
            curr_probs.append(self.prob_3(graphs[i], curr_graphs))
            curr_probs.append(self.prob_4(graphs[i], curr_graphs))
            
            probabilities.append(curr_probs)

        return probabilities


    # Number of edges that already existed
    def prob_1(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances an edge reappeared


        for edge in target_graph.edges():
            for graph in prev_graphs:
                if(edge in graph.edges()):
                    count += 1

        return count / num_edges_in_target


    # New edge because of one new node
    def prob_2(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances of a new edge because of one new node


        for edge in target_graph.edges():
            # Get nodes
            node_1 = edge[0]
            node_2 = edge[1]

            for graph in prev_graphs:
                curr_graph_nodes = graph.nodes()
                if((node_1 in curr_graph_nodes and node_2 not in curr_graph_nodes) or (node_1 not in curr_graph_nodes and node_2 in curr_graph_nodes)):
                    count += 1

        return count / num_edges_in_target



    # New edge between already existing nodes that did not previously have an edge
    def prob_3(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances of a new edge between existing nodes

        for edge in target_graph.edges():
            # Get nodes
            node_1 = edge[0]
            node_2 = edge[1]

            # Need to check this, might need to change to a break statement if it existed at any point in the previous graphs, works for single graph case
            for graph in prev_graphs:
                curr_graph_nodes = graph.nodes()

                # If the nodes previously existed, but did not make an edge, add to the count
                if(node_1 in curr_graph_nodes and node_2 in curr_graph_nodes) and (edge not in graph.edges()):
                    count += 1
                
        return count / num_edges_in_target


    # New edge between two new nodes
    def prob_4(self, target_graph, prev_graphs: list):
        num_edges_in_target = target_graph.number_of_edges()
        count = 0

        for edge in target_graph.edges():
            node_1 = edge[0]
            node_2 = edge[1]

            for graph in prev_graphs:
                curr_graph_nodes = graph.nodes()
                
                if(node_1 not in curr_graph_nodes and node_2 not in curr_graph_nodes):
                    count += 1
        
        return count / num_edges_in_target