import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Visualizer:
    figdir_regression = os.path.abspath('data/output/results/RegressionTesting/graphs/')
    figdir_binary = os.path.abspath('data/output/results/BinaryTesting/graphs/')
    
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task
        
    
    def display_loss(self, train_loss, valid_loss, num_epochs):
        """
        Display the loss of training and validation over time

        Args:
            train_loss (list): The train loss value over time
            valid_loss (list): The validation loss value over time
            num_epochs (int): The number of epochs the model trained for

        Returns:
            None
        """
        epochs = range(0, num_epochs + 1)
        plt.plot(epochs, train_loss, label="Train Loss", color="blue", marker="o")
        plt.plot(epochs, valid_loss, label="Validation Loss", color="green", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        
        # Save depending on task
        if self.task == 'regression':
            plt.savefig(self.figdir_regression + self.dataset + '_loss_graph.png')
        elif self.task == 'binary':
            plt.savefig(self.figdir_binary + self.dataset + '_loss_graph.png')
        
        plt.clf()
        
        
    def display_aucroc(self, train_aucroc, valid_aucroc, num_epochs):
        """
        Display the aucroc scores over time for binary classification task

        Args:
            train_aucroc (list): The train aucroc value over time
            valid_aucroc (list): The validation aucroc value over time
            num_epochs (int): The number of epochs the model trained for

        Returns:
            None
        """
        epochs = range(0, num_epochs + 1)
        plt.plot(epochs, train_aucroc, label="Train AUCROC Score", color="blue", marker="o")
        plt.plot(epochs, valid_aucroc, label="Validation AUCROC Score", color="green", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("AUCROC")
        plt.title("Training and Validation AUCROC Over Epochs")
        plt.legend()
        plt.savefig(self.figdir_binary + self.dataset + '_accuracy_graph.png')
        plt.clf()
        
    
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
            plt.clf()
            
            
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
        plt.clf()
            
            
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
        plt.clf()
    
    
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
            plt.clf()
    