import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression 


# A class with general utility functions
class Utils:
    def linear_fit(embeddings):
        """
        Fit a LinearRegression model to ensure monotonically increasing behavior in our embedding vector

        Args:
            embeddings (list): The embeddings received from the model

        Returns:
            new_embeddings(list): The embeddings modified to have guaranteed monotonically increasing behavior
        """
        new_embeddings = []
        nodes = []
        edges = []
        indices = np.arange(1,11).reshape(-1,1)

        # Data preparation
        for i in range(len(embeddings)):
            nodes.append(embeddings[i])
            edges.append(embeddings[i + 1])

        # Get nodes
        model = LinearRegression()
        model.fit(indices, nodes)
        new_nodes = model.predict(indices)

        # Get edges
        model = LinearRegression()
        model.fit(indices, edges)
        new_edges = model.predict(indices)

        # Add back to the embeddings
        for node_val, edge_val in zip(new_nodes, new_edges):
            new_embeddings.append(node_val)
            new_embeddings.append(edge_val)

        return new_embeddings


    def compute_overshoots(pred, real):
        """
        Compute how far off the embedding was for each particular index

        Args:
            pred (list): The predicted embeddings from the model
            real (list): The real embeddings to compare to

        Returns:
            overshoots (list): A list of equal dimension to the inputs, displays how far over/under each correspondingindex was
        """
        overshoots = []
        for pred_val, real_val in zip(pred, real):
            overshoots.append(pred_val - real_val)

        return overshoots


    def round_features(self, embeddings, num_extra_features=0):
        """
        For the necessary features, rounds the embedding elements to the nearest integer

        Args:
            embeddings (list): The embeddings received from the model
            num_extra_features (int): If we are using features other than nodes and edges, 0 by default

        Returns:
            new_embeddings(list): The embeddings modified for the nodes and edges to be whole numbers
        """
        new_embeddings = []
        
        for embedding in embeddings:
            new_embedding = []
            for i in range(0, len(embedding), (2 + num_extra_features)):
                new_embedding.append(round(embedding[i]))  # Add the rounded num nodes
                new_embedding.append(round(embedding[i + 1]))  # Add the rounded num edges
                
                # Add other features that don't need to be rounded
                for j in range(0, num_extra_features):
                    new_embedding.append(embedding[i + j])
        
            new_embeddings.append(new_embedding)
        
        return new_embeddings


    def calculate_necessary_edges(self, features):
        edges = 0
        curr_degree = 1
        prev_nodes = 0

        # Loop over all node values
        for i in range(0, len(features), 2):
            curr_nodes = features[i]
            new_nodes = curr_nodes - prev_nodes
            if(new_nodes > 0):
                edges += new_nodes * curr_degree

            prev_nodes = curr_nodes
            curr_degree += 1  # Increment degree

        return round(edges / 2)


    # The below functions are used in local search

    def determine_fitness(self, graph, true_features):
        """
        Computes the fitness score for the current state of the graph. Inspired by genetic algorithms in AI

        Args:
            graph (nx.Graph()): The current state of the graph in reconstruction
            true_features (list): The true features that we are aiming to create a graph to match

        Returns:
            diff_score(float): The difference between the current graphs embeddings and the true features
        """
        curr_features = self.compute_features(graph)
        
        diff_score = 0

        weights = [1]
        
        for weight, curr, target in zip(weights, curr_features, true_features):
            diff_score += weight * (np.abs(target - curr))  # Can modify if desired, for example: (target-curr) ** 2
        
        return diff_score
    

    def count_incomplete(self, graph):
        """
        Counts the number of nodes that have not yet reached their target degree value

        Args:
            graph (nx.Graph()): The graph in reconstruction that we are checking the nodes of

        Returns:
            len(unfilled_nodes) (int): The number of nodes yet to be filled in the graph
        """
        unfilled_nodes = [node for node in graph.nodes if graph.degree(node) < graph.nodes[node]['degree']]
        return len(unfilled_nodes)


    # Below methods are used in displaying results

    def compute_similarity(self, pred_graph, true_graph):
        """
        Computes a similarity score between the predicted graph and the true graph
        Might use a method inspired by Kadir's team

        Args:
            pred_graph (nx.Graph()): The current graph after reconstruction
            true_graph (nx.Graph()): The current true graph

        Returns:
            diff_score(float): The score of how different the two graphs are
        """
        self.display_graph(self, pred_graph, true_graph)

        # Prior calculations
        betweenness = nx.betweenness_centrality(true_graph)
        avg_true_betweenness = sum(betweenness.values()) / len(betweenness)
        betweenness = nx.betweenness_centrality(pred_graph)
        avg_pred_betweenness = sum(betweenness.values()) / len(betweenness)
        
        # Sample metrics, can add or remove as seen fit
        similarity_values = {
            'betweenness_diff': avg_true_betweenness - avg_pred_betweenness,
            'density_diff': nx.density(true_graph) - nx.density(pred_graph),
            'cycles_diff': len(nx.cycle_basis(true_graph)) - len(nx.cycle_basis(pred_graph)),
            'nodes_diff': nx.number_of_nodes(true_graph) - nx.number_of_nodes(pred_graph),
            'edge_diff': nx.number_of_edges(true_graph) - nx.number_of_edges(pred_graph),
            'cliques_diff': len(list(nx.find_cliques(true_graph))) - len(list(nx.find_cliques(pred_graph))),
        }
        
        # Display results
        for key, value in similarity_values.items():
            if value > 0:
                print(f'True predicted graph has a higher {key} value by: {value}')
            elif value < 0:
                print(f'True predicted graph has a lower {key} value by: {np.abs(value)}') 
            elif value == 0:
                print(f'True graph and predicted graph have exact same {key} value')
            
        print(f'There are {nx.number_of_nodes(true_graph)} nodes in the true graph and {nx.number_of_nodes(pred_graph)} nodes in the predicted graph')
        print(f'There are {nx.number_of_edges(true_graph)} edges in the true graph and {nx.number_of_edges(pred_graph)} edges in the predicted graph')
        # Compute the similarity between the graphs based on how many alterations/additions/deletions are required 
        edit_normalization = (nx.number_of_nodes(true_graph) + nx.number_of_nodes(pred_graph) + nx.number_of_edges(true_graph) + nx.number_of_edges(pred_graph)) / 2  # Can change, used if graphs are different sizes
        #edit_distance = nx.graph_edit_distance(true_graph, pred_graph) / edit_normalization  # 0 means perfect match; 1 or higher means highly different
        edit_distance_gen = nx.optimize_graph_edit_distance(true_graph, pred_graph)  # Less computationally intensive
        edit_distance = next(edit_distance_gen) / edit_normalization
        
        return similarity_values, edit_distance


    def display_edit_differences(self, edit_distances):
        """
        Displays how far off we were for each graph for each edit score

        Args:
            edit_distances (list): The number of edits required to reach each true graph

        Returns:
            None
        """
        timestamps = list(range(len(edit_distances)))

        # Plot the edit distances
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, edit_distances, marker='o', color='b', linestyle='-', linewidth=2, markersize=6)
        plt.xlabel("Timestamp")
        plt.ylabel("Edit Distance")
        plt.title("Edit Distances Across Graph Timestamps")
        plt.grid(True)

        # Show the plot
        plt.show()


    def display_similarity_values(self, similarity_scores):
        """
        Displays how far off we were for each graph for each score metric

        Args:
            similarity_scores (dict): The similarity scores for each index

        Returns:
            None
        """
        metrics = list(similarity_scores.keys())
        values = list(similarity_scores.values())

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color='skyblue')
        plt.xlabel("Metrics")
        plt.ylabel("Difference Value")
        plt.title("Similarity Metric Differences Between True and Predicted Graphs")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()


    def display_graph(self, pred_graph, true_graph):
        """
        Uses MatPlotLib to visualize the predicted and true graph

        Args:
            pred_graph (nx.Graph()): The graph after reconstruction
            true_graph (nx.Graph()): The graph true graph we were aiming to build

        Returns:
            None
        """
        nx.draw_spring(pred_graph, with_labels=True)
        plt.title("The predicted graph")
        plt.show()
        plt.clf()

        nx.draw_spring(true_graph, with_labels=True)
        plt.title("The true graph")
        plt.show()
        plt.clf()