import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import seaborn as sns
import networkx as nx
# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Visualizer:
    """
    TODO: Remove unused visulization functions
    """
    figdir_regression = os.path.abspath('data/output/results/RegressionTesting/graphs/')
    figdir_binary = os.path.abspath('data/output/results/BinaryTesting/graphs/')
    
    def __init__(self, dataset=None, task=None):
        self.dataset = dataset
        self.task = task
        
    
    def display_loss(self, train_loss, valid_loss, num_epochs=None, edge_type=None, save_path=None):
        """
        Display and save train/validation loss curve.

        Args:
            train_loss (list): Train loss per epoch.
            valid_loss (list): Validation loss per epoch.
            num_epochs (int, optional): Expected number of epochs. If None, uses len(train_loss).
            edge_type (str, optional): Edge type label for title/filename.
            save_path (str, optional): Full path where to save the figure. If None, falls back to figdir_*.
        """
        # Use actual length if num_epochs not provided or mismatch
        n_points = len(train_loss)
        if num_epochs is not None and num_epochs != n_points:
            # use actual collected points to avoid x/y length mismatch
            n_points = len(train_loss)

        if n_points == 0:
            # nothing to plot
            return

        epochs = range(1, n_points + 1)

        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss", marker="o")
        plt.plot(epochs, valid_loss, label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel(f"Loss{(' - ' + edge_type) if edge_type else ''}")
        title_suffix = f" for {edge_type}" if edge_type else ""
        plt.title("Training and Validation Loss Over Epochs" + title_suffix)
        plt.legend()
        plt.tight_layout()

        # Determine save path
        if save_path:
            out_path = save_path
        else:
            # fallback by task
            fname = f"{self.dataset}_{edge_type or 'all'}_loss_graph.png"
            if getattr(self, "task", None) == 'regression':
                os.makedirs(self.figdir_regression, exist_ok=True)
                out_path = os.path.join(self.figdir_regression, fname)
            else:
                # default to binary folder
                os.makedirs(self.figdir_binary, exist_ok=True)
                out_path = os.path.join(self.figdir_binary, fname)

        # ensure parent dir exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        
        
    def display_aucroc(self, train_aucroc, valid_aucroc, num_epochs=None, edge_type=None, save_path=None):
        """
        Display and save train/validation AUCROC curve.

        Args:
            train_aucroc (list): Train AUC per epoch.
            valid_aucroc (list): Validation AUC per epoch.
            num_epochs (int, optional): Expected number of epochs. If None, uses len(train_aucroc).
            edge_type (str, optional): Edge type label for title/filename.
            save_path (str, optional): Full path where to save the figure. If None, falls back to figdir_binary.
        """
        n_points = len(train_aucroc)
        if num_epochs is not None and num_epochs != n_points:
            n_points = len(train_aucroc)

        if n_points == 0:
            return

        epochs = range(1, n_points + 1)

        plt.figure()
        plt.plot(epochs, train_aucroc, label="Train AUCROC", marker="o")
        plt.plot(epochs, valid_aucroc, label="Validation AUCROC", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel(f"AUCROC{(' - ' + edge_type) if edge_type else ''}")
        title_suffix = f" for {edge_type}" if edge_type else ""
        plt.title("Training and Validation AUCROC Over Epochs" + title_suffix)
        plt.legend()
        plt.tight_layout()

        if save_path:
            out_path = save_path
        else:
            fname = f"{self.dataset}_{edge_type or 'all'}_aucroc_graph.png"
            os.makedirs(self.figdir_binary, exist_ok=True)
            out_path = os.path.join(self.figdir_binary, fname)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        
    
    def display_embeddings(self, predicted_embeddings, real_embeddings, linfit_embeddings):
        """
        Display, in two graphs, the predicted embeddings and true embeddings

        Args:
            predicted_embeddings (list): The predicted feature vector to display
            real_embeddings (list): The real feature vector to display
            linfit_embeddings (list): The predicted feature vector after linear regression fit to it to display

        Returns:
            None
        """
        for predicted, real, linfit in zip(predicted_embeddings, real_embeddings, linfit_embeddings):
            real_nodes = []
            real_edges = []
            real_weights = []
            pred_nodes = []
            pred_edges = []
            pred_weights = []
            linfit_nodes = []
            linfit_edges = []
            linfit_weights = []
            indices = np.linspace(1, 10, num=10)
            for i in range(0, len(predicted), 3):
                linfit_nodes.append(linfit[i])
                real_nodes.append(real[i])
                pred_nodes.append(predicted[i])
                linfit_edges.append(linfit[i + 1])
                real_edges.append(real[i + 1])
                pred_edges.append(predicted[i + 1])
                real_weights.append(real[i + 2])
                pred_weights.append(predicted[i + 2])
                linfit_weights.append(linfit[i + 2])

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))  
            ax1.plot(indices, real_nodes, label="Real Node Count", color='blue')
            ax1.plot(indices, pred_nodes, label="Predicted Node Count", color='orange')
            ax1.plot(indices, linfit_nodes, label="Linear Fit Node Count", color='green')
            ax1.set_xlabel("Percentile Activated")
            ax1.set_ylabel("Number of Nodes")
            ax1.legend()
            ax1.set_title("Plot of Node Counts")
            
            ax2.plot(indices, real_edges, label="Real Edge Count", color='blue')
            ax2.plot(indices, pred_edges, label="Predicted Edge Count", color='orange')
            ax2.plot(indices, linfit_edges, label="Linear Fit Edge Count", color='green')
            ax2.set_xlabel("Percentile Activated")
            ax2.set_ylabel("Number of Edges")
            ax2.legend()
            ax2.set_title("Plot of Edge Counts")
            
            ax3.plot(indices, real_weights, label="Real Weight Activated", color='blue')
            ax3.plot(indices, pred_weights, label="Predicted Weight Activated", color='orange')
            ax3.plot(indices, linfit_weights, label="Linear Fit Weight Activated", color='green')
            ax3.set_xlabel("Percentile Activated")
            ax3.set_ylabel("Activated Weight")
            ax3.legend()
            ax3.set_title("Plot of Weight Activated")
            
            # For now just takes last one
            #plt.savefig(self.figdir_regression + self.dataset + '_embedding_graph.png')
            plt.show()
            plt.close()
            
            
    def display_embeddings_once(self, predicted_embedding, real_embedding, linfit_embedding):
        """
        Display, in two graphs, the predicted embeddings and true embeddings

        Args:
            predicted_embeddings (list): The predicted feature vector to display
            real_embeddings (list): The real feature vector to display
            linfit_embeddings (list): The predicted feature vector after linear regression fit to it to display

        Returns:
            None
        """
        real_nodes = []
        real_edges = []
        real_weights = []
        pred_nodes = []
        pred_edges = []
        pred_weights = []
        linfit_nodes = []
        linfit_edges = []
        linfit_weights = []
        indices = np.linspace(1, 10, num=10)
        for i in range(0, len(predicted_embedding), 3):
            linfit_nodes.append(linfit_embedding[i])
            real_nodes.append(real_embedding[i])
            pred_nodes.append(predicted_embedding[i])
            linfit_edges.append(linfit_embedding[i + 1])
            real_edges.append(real_embedding[i + 1])
            pred_edges.append(predicted_embedding[i + 1])
            real_weights.append(real_embedding[i + 2])
            pred_weights.append(predicted_embedding[i + 2])
            linfit_weights.append(linfit_embedding[i + 2])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))  
        ax1.plot(indices, real_nodes, label="Real Node Count", color='blue')
        ax1.plot(indices, pred_nodes, label="Predicted Node Count", color='orange')
        ax1.plot(indices, linfit_nodes, label="Linear Fit Node Count", color='green')
        ax1.set_xlabel("Percentile Activated")
        ax1.set_ylabel("Number of Nodes")
        ax1.legend()
        ax1.set_title("Plot of Node Counts")
        
        ax2.plot(indices, real_edges, label="Real Edge Count", color='blue')
        ax2.plot(indices, pred_edges, label="Predicted Edge Count", color='orange')
        ax2.plot(indices, linfit_edges, label="Linear Fit Edge Count", color='green')
        ax2.set_xlabel("Percentile Activated")
        ax2.set_ylabel("Number of Edges")
        ax2.legend()
        ax2.set_title("Plot of Edge Counts")
        
        ax3.plot(indices, real_weights, label="Real Weight Activated", color='blue')
        ax3.plot(indices, pred_weights, label="Predicted Weight Activated", color='orange')
        ax3.plot(indices, linfit_weights, label="Linear Fit Weight Activated", color='green')
        ax3.set_xlabel("Percentile Activated")
        ax3.set_ylabel("Activated Weight")
        ax3.legend()
        ax3.set_title("Plot of Weight Activated")
        
        # For now just takes last one
        #plt.savefig(self.figdir_regression + self.dataset + '_embedding_graph.png')
        plt.show()
        plt.close()
            
            
    def display_single_embedding(self, embedding, num_buckets=10):
        """
        Display, in two graphs, the predicted embeddings and true embeddings

        Args:
            predicted_embeddings (list): The feature vector to display

        Returns:
            None
        """
        nodes = []
        edges = []
        weights = []
        
        indices = np.linspace(1, num_buckets, num=num_buckets)
        for i in range(0, len(embedding), 5):
            nodes.append(embedding[i])
            edges.append(embedding[i + 1])
            weights.append(embedding[i + 2])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))  
        ax1.plot(indices, nodes, label="Node Count", color='blue')
        ax1.set_xlabel("Percentile Activated")
        ax1.set_ylabel("Number of Nodes")
        ax1.legend()
        ax1.set_title("Plot of Node Counts")
        
        ax2.plot(indices, edges, label="Edge Count", color='blue')
        ax2.set_xlabel("Percentile Activated")
        ax2.set_ylabel("Number of Edges")
        ax2.legend()
        ax2.set_title("Plot of Edge Counts")
        
        ax3.plot(indices, weights, label="Total Weight", color='blue')
        ax3.set_xlabel("Percentile Activated")
        ax3.set_ylabel("Activated Weight")
        ax3.legend()
        ax3.set_title("Plot of Weight Activated")
        
        # For now just takes last one
        plt.show()
        plt.close()
    
    
    def display_differences(self, predicted_embeddings, real_embeddings, linfit_embeddings):
        """
        Needs edits
        Displays the differences between the predicted feature vector and the true feature vector

        Args:
            file_name (string): The specified datset to load

        Returns:
            None
        """
        for predicted, real in zip(predicted_embeddings, real_embeddings):
            differences = np.abs(predicted - real)
            sns.heatmap([differences], annot=True, cmap="coolwarm", cbar=True)
            plt.title('Absolute Differences Between Vectors')
            plt.xlabel('Dimension')
            plt.show()
            plt.close()
            
        
    def display_composite_embeddings(self, betweenness_embeddings, closeness_embeddings, degree_embeddings, forman_embeddings, weight_embeddings, num_buckets=10, display_idx=0):
        """
        
        """
        # Initialize embedding lists
        degree_nodes = []
        degree_edges = []
        degree_weights = []
        closeness_nodes = []
        closeness_edges = []
        closeness_weights = []
        betweenness_nodes = []
        betweenness_edges = []
        betweenness_weights = []
        weight_nodes = []
        weight_edges = []
        weight_weights = []
        forman_nodes = []
        forman_edges = []
        forman_weights = []

        indices = np.linspace(1, num_buckets, num=num_buckets)  # Indices for graphing

        # Get the individual embeddings
        for i in range(0, (num_buckets * 3), 3):
            betweenness_nodes.append(betweenness_embeddings[display_idx][i])
            betweenness_edges.append(betweenness_embeddings[display_idx][i + 1])
            betweenness_weights.append(betweenness_embeddings[display_idx][i + 2])
            
            closeness_nodes.append(closeness_embeddings[display_idx][i])
            closeness_edges.append(closeness_embeddings[display_idx][i + 1])
            closeness_weights.append(closeness_embeddings[display_idx][i + 2])
            
            degree_nodes.append(degree_embeddings[display_idx][i])
            degree_edges.append(degree_embeddings[display_idx][i + 1])
            degree_weights.append(degree_embeddings[display_idx][i + 2])
            
            forman_nodes.append(forman_embeddings[display_idx][i])
            forman_edges.append(forman_embeddings[display_idx][i + 1])
            forman_weights.append(forman_embeddings[display_idx][i + 2])
            
            weight_nodes.append(weight_embeddings[display_idx][i])
            weight_edges.append(weight_embeddings[display_idx][i + 1])
            weight_weights.append(weight_embeddings[display_idx][i + 2])


        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 9))  
        ax1.plot(indices, betweenness_nodes, label="Betweenness", color='green')
        ax1.plot(indices, closeness_nodes, label="Closeness", color='orange')
        ax1.plot(indices, degree_nodes, label="Degree", color='blue')
        ax1.plot(indices, forman_nodes, label="F. Ricci", color='black')
        ax1.plot(indices, weight_nodes, label="Weight", color='purple')
        ax1.set_xlabel("Percentile")
        ax1.set_ylabel("Number of Nodes")
        ax1.legend()
        ax1.set_title(f"Plot of Node Counts {num_buckets} Thresholds")

        ax2.plot(indices, betweenness_edges, label="Betweenness", color='green')
        ax2.plot(indices, closeness_edges, label="Closeness", color='orange')
        ax2.plot(indices, degree_edges, label="Degree", color='blue')
        ax2.plot(indices, forman_edges, label="F. Ricci", color='black')
        ax2.plot(indices, weight_edges, label="Weight", color='purple')
        ax2.set_xlabel("Percentile")
        ax2.set_ylabel("Number of Edges")
        ax2.legend()
        ax2.set_title(f"Plot of Edge Counts {num_buckets} Thresholds")

        ax3.plot(indices, betweenness_weights, label="Betweenness", color='green')
        ax3.plot(indices, closeness_weights, label="Closeness", color='orange')
        ax3.plot(indices, degree_weights, label="Degree", color='blue')
        ax3.plot(indices, forman_weights, label="F. Ricci", color='black')
        ax3.plot(indices, weight_weights, label="Weight", color='purple')
        ax3.set_xlabel("Percentile")
        ax3.set_ylabel("Value")
        ax3.legend()
        ax3.set_title(f"Plot of Weight Counts {num_buckets} Thresholds")
        
        plt.show()
        plt.close()
        

    def display_pred_graph_vs_true_graph(self, predGraph, trueGraph):
        # Convert to undirected for fair visual comparison
        pred = predGraph
        true = trueGraph

        # Use same layout for both so node positions match
        combined = nx.compose(pred, true)
        pos = nx.spring_layout(combined, seed=42)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # --- True Graph ---
        axs[0].set_title("True Graph")
        nx.draw(true, pos, ax=axs[0], with_labels=True, node_color='lightgreen',
                edge_color='gray', node_size=800, font_size=10)
        axs[0].axis('off')

        # --- Predicted Graph ---
        axs[1].set_title("Predicted Graph")
        nx.draw(pred, pos, ax=axs[1], with_labels=True, node_color='skyblue',
                edge_color='gray', node_size=800, font_size=10)
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()


    def plot_scatter(predicted, true, save_path, mode="nodes", xlabel="", ylabel=""):
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        predicted = np.asarray(predicted, dtype=np.float32).ravel()
        true = np.asarray(true, dtype=np.float32).ravel()

        # FIX 1: Use a square figure size (e.g., 8x8 instead of 10x6)
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(predicted, true, alpha=0.6)

        max_val = max(np.max(predicted), np.max(true))
        max_limit = max(1.0, 1.05 * max_val)

        # NEW: Add a dashed diagonal line representing perfect predictions
        ax.plot([0, max_limit], [0, max_limit], color='black', linestyle='--', alpha=0.4)

        fs = 14
        if mode == "nodes":
            ax.set_xlabel(r'$|\hat{\mathcal{V}}|$', fontsize=fs)
            ax.set_ylabel(r'$|\mathcal{V}|$', fontsize=fs)
        elif mode == "edges":
            ax.set_xlabel(r'$|\hat{\mathcal{E}}|$', fontsize=fs)
            ax.set_ylabel(r'$|\mathcal{E}|$', fontsize=fs)
        else:
            ax.set_xlabel(xlabel, fontsize=fs)
            ax.set_ylabel(ylabel, fontsize=fs)

        locator = ticker.MaxNLocator(nbins='auto', integer=True)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)

        ax.set_xlim(0, max_limit)
        ax.set_ylim(0, max_limit)
        
        # FIX 2: Set adjustable to 'box' so the physical axes stay square
        ax.set_aspect('equal', adjustable='box')

        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
                
    
    def plot_line_graph(true_vals, pred_vals, prob_type, xlabel="Time Index", ylabel="Probability", title="True vs Real: ", save_path="plot.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(8, 6))

        # Plotting
        x = range(len(true_vals))  # Use indices as x-axis
        plt.plot(x, true_vals, color="blue", linestyle="-", linewidth=1, label="True")
        plt.plot(x, pred_vals, color="red", linestyle="--", linewidth=1, label="Pred")

        # Remove top and right borders
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Axis labels with bigger font
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        # Title with bigger font
        plt.title(title + prob_type, fontsize=16)

        # Legend
        plt.legend(fontsize=12)

        # Save the figure
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()