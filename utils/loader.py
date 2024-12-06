import networkx as nx
import pandas as pd
import numpy as np
import pickle
import networkx as nx

# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

class Loader():
    # File paths
    output_dir = os.path.abspath('data/input/cached')
    edgelist_dir = os.path.abspath('data/input/raw/edgelist')
    label_dir = os.path.abspath('data/input/raw/labels')

    # Process all data
    def load_all_data(self):
        """
        Load all datasets at once and return them as a list
        
        Args:
            None
        
        Returns:
            all_data (list): All datasets available
        """
        
        all_data = []
        self.to_cached()

        # Make a list of every data file
        data_files = [file for file in os.listdir(self.output_dir)]
        for file in data_files:
            all_data.append(self.from_cached(file))
        
        return all_data

    def load_data(self, dataset, activation):
        """
        Load a single, specified dataset that exists
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            graphs (list): A list of networkx graphs to process
            labels (list): The associated labels for each graph
        """
        self.to_cached()
        seek_file = dataset + '_' + activation  # Based on dataset and activation combination
        print(seek_file)
        data_files = [file for file in os.listdir(self.output_dir)]
        print(data_files)
        if seek_file in data_files:
            graphs, labels = self.from_cached(seek_file)
            return graphs, labels
        
        else:
            print(f'Dataset {dataset} not found in files, please check available datasets and try again')
            print(f'Available data: \t{data_files}')


    # Read labels from file
    def read_labels(self, dataset):
        """
        Read the labels from a csv file for later processing
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            labels (list): All labels for the specified dataset
        """
        labels = pd.read_csv(self.label_dir + '/{}_Label.csv'.format(dataset), header=None)
        labels = labels.iloc[:, 0]
        labels = labels.squeeze().tolist()
        return labels
    

    # Load the data from edge list txt file
    def read_edges(self, dataset):
        """
        Read the edgelists a file for later processing
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            data (list): All graphs created for processing
        """
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = []
        edgelist_rawfile = self.edgelist_dir + '/{}.txt'.format(dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        uniq_ts_list = np.unique(edgelist_df['Snapshot'])

        # Loop over snapshot ids
        for ts in uniq_ts_list:
            # NOTE: this code does not use any node or edge features
            ts_edges = edgelist_df.loc[edgelist_df['Snapshot'] == ts, ['from', 'to']]
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to')
            data.append(ts_G)
        
        return data
    
    
    def read_edges_directed(self, dataset):
        """
        Read the edgelists a file for later processing
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            data (list): All graphs created for processing
        """
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = []
        edgelist_rawfile = self.edgelist_dir + '/{}.txt'.format(dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        uniq_ts_list = np.unique(edgelist_df['Snapshot'])

        # Loop over snapshot ids
        for ts in uniq_ts_list:
            ts_edges = edgelist_df.loc[edgelist_df['Snapshot'] == ts, ['from', 'to', 'value']]
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to', edge_attr=True, create_using=nx.DiGraph())
            data.append(ts_G)
        
        return data


    # Get the data folders into respective cached pkl files
    def to_cached(self):
        """
        Send all of the processed datasets to pkl files for easy loading later
        
        Args:
            None
        
        Returns:
            None
        """
        raw_data = [file for file in os.listdir(self.edgelist_dir)]
        raw_data = [file_name.replace('.txt', '') for file_name in raw_data]
        cached_data = [file for file in os.listdir(self.output_dir)]

        activations = [EmbedBetweenness, EmbedCloseness, EmbedDegree, EmbedForman, EmbedWeight]  # All activation functions to use
        activation_names = ['Betweenness', 'Closeness', 'Degree', 'Forman', 'Weight']
        
        missing_cached = []
        for dataset in raw_data:
            # Check for base file
            base_file = f'{dataset}/dataset.pkl'
            if base_file not in cached_data:
                missing_cached.append(dataset)
                continue  # Skip to the next dataset if the base file is missing

            # Check for activation-specific files
            for activation_name in activation_names:
                activation_file = f'{dataset}/dataset_{activation_name}.pkl'
                if activation_file not in cached_data:
                    missing_cached.append(dataset)
                    break  # Skip to the next dataset if any activation file is missing
                
        # If we are missing files, generate them
        if missing_cached:
            print(f'Generating pkl files for the following datasets: {missing_cached}')

            for file in missing_cached:
                # Generate base data with graphs, labels
                graphs = self.read_edges_directed(file)
                labels = self.read_labels(file)
                data = list(zip(graphs, labels))

                data_dir = self.output_dir + '/' + file
                os.makedirs(data_dir, exist_ok=True)
                print('sending to ' + data_dir + '/' + file + '.pkl')
                with open(data_dir + '/' + file + '.pkl', "wb") as f:
                    pickle.dump(data, f)
                    
                # Generate data with embeddings, labels
                for activation, activation_name in zip(activations, activation_names):
                    print(f'Generating for {file} with activation {activation_name}')
                    my_activation = activation(num_buckets=10)    
        
                    # Since Forman Ricci requires directed edges
                    if activation==EmbedForman:
                        embeddings = my_activation.process_graphs_for_embeddings(graphs, is_directed=True)
                    else:
                        embeddings = my_activation.process_graphs_for_embeddings(graphs)
                        
                    data = list(zip(embeddings, labels))

                    data_dir = self.output_dir + '/' + file
                    os.makedirs(data_dir, exist_ok=True)
                    print('sending to ' + data_dir + '/' + file + '_' + activation_name + '.pkl')
                    with open(data_dir + '/' + file + '_' + activation_name  + '.pkl', "wb") as f:
                        pickle.dump(data, f)


    # Load from the pkl file
    def from_cached(self, file_name):
        """
        Load the specified file, if it exists, from the pkl file

        Args:
            file_name (string): The specified datset to load

        Returns:
            graphs (list): A list of networkx graphs to process
            labels (list): The associated labels for each graph
        """
        with open(self.output_dir + '/' + file_name + '/' + file_name + '.pkl', "rb") as f:
            data = pickle.load(f)

        graphs, labels = zip(*data)
        return graphs, labels