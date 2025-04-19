# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
import pandas as pd


class Probs():
    def gen_probs(self, num_graphs_back: int, graphs: list, from_start=True):
        """
        Generate the probabilities for each graph snapshot, for a number of snapshots back
        
        inputs:
            num_graphs_back (int): The number of graphs to look back when computing probabilities
            graphs (list(nx.DiGraph())): The graphs to generate probabilities for
        
        returns:
            probabilities(list[list[float]]): A list of 4 (or 5) probabilities for each graph
        """
        probabilities = []

        # All new nodes and new edges
        for i in range(num_graphs_back):
            target_graph = graphs[i]
            probs = [
                0,
                target_graph.number_of_nodes(),
                0,
                target_graph.number_of_edges(),
                0,
                0
            ]
            probabilities.append(probs)
            

        for i in range(num_graphs_back, len(graphs)):
            if(from_start):
                num_graphs_back = i
            curr_graphs = graphs[i - num_graphs_back : i]  # Get groups of size num_graphs_back
            target_graph = graphs[i]

            # Generate our probabilities
            probs = [
                self.prob_old_nodes(target_graph, curr_graphs),
                self.prob_new_nodes(target_graph, curr_graphs),
                self.prob_oo(target_graph, curr_graphs),
                self.prob_nn(target_graph, curr_graphs),
                self.prob_on(target_graph, curr_graphs),
                self.prob_oon(target_graph, curr_graphs)
            ]
            
            probabilities.append(probs)

        return probabilities


    # Edges that already existed in previous graphs
    def prob_oo(self, target_graph, prev_graphs: list):
        """
        Compute the percentage of edges that exist in this graph, that existed in the previous graphs
        
        inputs:
            target_graph (nx.DiGraph()): The graph we are computing probabilities for
            prev_graphs(list[nx.DiGraph()]): The previous graphs to look at for their edges
        
        returns:
            prob_1 (float): The number of edges that exist in this graph that existed in previous graphs
        """
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances an edge reappearing

        prev_edges = set().union(*(graph.edges() for graph in prev_graphs))

        for edge in target_graph.edges():
            if edge in prev_edges:
                count += 1

        prob_1 = count / num_edges_in_target  # Compute probability

        return count  # No longer using a probability, now a discrete count


    # New edge between already existing nodes that did not previously have an edge
    def prob_oon(self, target_graph, prev_graphs: list):
        """
        Compute the percentage of edges that exist in this graph, that had two previously seen nodes that did not previously have an edge
        
        inputs:
            target_graph (nx.DiGraph()): The graph we are computing probabilities for
            prev_graphs(list[nx.DiGraph()]): The previous graphs to look at for their edges
        
        returns:
            prob_2 (float): The number of edges that formed between two, previously existing nodes that did not have an edge
        """
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances of a new edge between existing nodes

        # Get all nodes and edges that previously existed
        prev_nodes = set().union(*(graph.nodes() for graph in prev_graphs))
        prev_edges = set().union(*(graph.edges() for graph in prev_graphs))


        for edge in target_graph.edges():
            # Get nodes
            node_1 = edge[0]
            node_2 = edge[1]

            # If the nodes previously existed, but did not make an edge, add to the count
            if(node_1 in prev_nodes and node_2 in prev_nodes) and (edge not in prev_edges):
                count += 1
                
        prob_2 = count / num_edges_in_target  # Compute probability 
        
        return count  # No longer using a probability, now a discrete count


    # New edge because of one new node
    def prob_on(self, target_graph, prev_graphs: list):
        """
        Compute the percentage of edges that exist in this graph, that have one old node and one new node
        
        inputs:
            target_graph (nx.DiGraph()): The graph we are computing probabilities for
            prev_graphs(list[nx.DiGraph()]): The previous graphs to look at for their edges
        
        returns:
            prob_3 (float): The number of edges that exist in this graph between a previously existing node and a new node
        """
        num_edges_in_target = target_graph.number_of_edges()
        count = 0  # Our numerator, the number of instances of a new edge because of one new node

        # Not sure if i want to use intersection or union here when considering multiple graphs
        prev_nodes = set().union(*(graph.nodes() for graph in prev_graphs))
        nodes_in_curr_graph = set(target_graph.nodes())
        new_nodes = nodes_in_curr_graph - prev_nodes

        for edge in target_graph.edges():
            # Get nodes
            node_1 = edge[0]
            node_2 = edge[1]

            if((node_1 in prev_nodes and node_2 in new_nodes) or (node_1 in new_nodes and node_2 in prev_nodes)):
                count += 1
        
        prob_3 = count / num_edges_in_target  # Compute probability 
        
        return count  # No longer using a probability, now a discrete count


    # New edge between two new nodes
    def prob_nn(self, target_graph, prev_graphs: list):
        """
        Compute the percentage of edges that exist in this graph, that formed because of two entirely new nodes
        
        inputs:
            target_graph (nx.DiGraph()): The graph we are computing probabilities for
            prev_graphs(list[nx.DiGraph()]): The previous graphs to look at for their edges
        
        returns:
            prob_4 (float): The number of edges that exist in this graph that formed because of two entirely new nodes
        """
        num_edges_in_target = target_graph.number_of_edges()
        count = 0

        # Get all nodes that previously existed
        prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])

        for node_1, node_2 in target_graph.edges():
            if(node_1 not in prev_nodes and node_2 not in prev_nodes):
                count += 1

        prob_4 = count / num_edges_in_target  # Compute probability
        
        return count  # No longer using a probability, now a discrete count

    
    # Number of reappearing nodes
    def prob_old_nodes(self, target_graph, prev_graphs: list):
        """
        Compute the percentage of nodes that exist in this graph that were previously seen
        
        inputs:
            target_graph (nx.DiGraph()): The graph we are computing probabilities for
            prev_graphs(list[nx.DiGraph()]): The previous graphs to look at for their nodes
        
        returns:
            prob_5 (float): The number of nodes that exist in this graph, that were seen in previous graphs
        """
        num_nodes_in_target = target_graph.number_of_nodes()
        
        prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])
        curr_nodes = set(target_graph.nodes())
        
        count = len(list(prev_nodes.intersection(curr_nodes)))
        
        prob_5 = count / num_nodes_in_target  # Compute probability
        
        return count  # No longer using a probability, now a discrete count
    

    def prob_new_nodes(self, target_graph, prev_graphs: list):
        """
        Compute the percentage of nodes that exist in this graph that were not previously seen
        
        inputs:
            target_graph (nx.DiGraph()): The graph we are computing probabilities for
            prev_graphs(list[nx.DiGraph()]): The previous graphs to look at for their nodes
        
        returns:
            prob_5 (float): The number of nodes that exist in this graph, that were seen in previous graphs
        """
        num_nodes_in_target = target_graph.number_of_nodes()
        
        prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])
        curr_nodes = set(target_graph.nodes())
        
        count = len(curr_nodes - prev_nodes)
        
        prob_5 = count / num_nodes_in_target  # Compute probability
        
        return count  # No longer using a probability, now a discrete count
    
    
# Generate the probability files based on number of graphs to look back
if __name__ == "__main__":
    my_loader = Loader()
    my_probs = Probs()
    num_graphs_back = 1  # Number of graphs to look back
    from_start = True
    
    datasets = ['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B']

    for dataset in datasets:
        graphs = my_loader.read_edges_directed(dataset)
        probs = my_probs.gen_probs(num_graphs_back=num_graphs_back, graphs=graphs, from_start=from_start)
        df = pd.DataFrame(probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
        if from_start:
            df.to_csv(f'ReinforcementLearning/output/probabilities/all_back/{dataset}_from_start.csv')
        else:
            df.to_csv(f'ReinforcementLearning/output/probabilities/{dataset}_{num_graphs_back}back.csv')