import networkx as nx
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import time

# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.incremental_closeness import EmbedIncrementalCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

from multiprocessing import Pool

class Loader():
    # File paths
    output_dir = os.path.abspath('data/input/cached')
    edgelist_dir = os.path.abspath('data/input/raw/edgelist')
    label_dir = os.path.abspath('data/input/raw/labels')

    def verify_embeddings(self, embeddings, activation, dataset, norm=False):
        print(f'Verifying {activation} on dataset {dataset}')
        if activation == EmbedForman and dataset == 'Reddit_B':
            return 
        
        if not norm:
            graphs = self.read_edges_directed(dataset)
            for embedding, graph in zip(embeddings, graphs):
                total_nodes = graph.number_of_nodes()
                total_edges = graph.number_of_edges()
                total_weight = sum(data['value'] for _, _, data in graph.edges(data=True))
                
                embedding_nodes = embedding[-3]
                embedding_edges = embedding[-2]
                embedding_weight = embedding[-1]
                
                if total_nodes != embedding_nodes:
                    print(f'A GRAPH FROM {dataset} WITH ACTIVATION {activation} HAS THE WRONG NUMBER OF NODES')
                    print(f'TRUE VAL: {total_nodes}; EMBEDDING VAL: {embedding_nodes}')
                
                if total_edges != embedding_edges:
                    print(f'A GRAPH FROM {dataset} WITH ACTIVATION {activation} HAS THE WRONG NUMBER OF EDGES')
                    print(f'TRUE VAL: {total_edges}; EMBEDDING VAL: {embedding_edges}')
                    
                if total_weight != embedding_weight:
                    print(f'A GRAPH FROM {dataset} WITH ACTIVATION {activation} HAS THE WRONG NUMBER OF WEIGHT')
                    print(f'TRUE VAL: {total_weight}; EMBEDDING VAL: {embedding_weight}')


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
        seek_file = dataset + '_' + activation + '.pkl'  # Based on dataset and activation combination
        dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
        data_files = os.listdir(dataset_folder)
        
        if seek_file in data_files:
            seek_file_path = os.path.join(dataset_folder, seek_file)
            graphs, labels = self.from_cached(seek_file_path)  # Load data
            return graphs, labels
        
        else:
            print(f'Dataset {dataset} not found in files, please check available datasets and try again')
            print(f'Available data: \t{data_files}')

    
    def clean_data(self):
        pass


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
    
    
    def read_edges_directed(self, dataset, norm=False):
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
        
        if norm:
            edgelist_df = self.normalize_edge_weights(edgelist_df)
        
        # Filter out edges where 'value' is greater than 10^20 (this is a hacking attempt in blockchain)
        # edgelist_df = edgelist_df[edgelist_df['value'] <= 10**20]

        
                
        uniq_ts_list = np.unique(edgelist_df['Snapshot'])

        # Loop over snapshot ids
        for ts in uniq_ts_list:
            ts_edges = edgelist_df.loc[edgelist_df['Snapshot'] == ts, ['from', 'to', 'value']]
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to', edge_attr=True, create_using=nx.DiGraph())
            
            # Reddit requires special processing (apply sigmoid)
            if dataset == 'Reddit_B':
                for u, v, graph_data in ts_G.edges(data=True):
                    graph_data['value'] = 1 / (1 + np.exp(-graph_data['value']))
            
            data.append(ts_G)
        
        return data


    def normalize_edge_weights(self, edges_df):
        INT_MAX = 2**31 - 1  # Maximum 32-bit integer value
        edges_df["value"].replace([np.inf, -np.inf], INT_MAX, inplace=True)  # In case there are inf values
        
        # Using log scaling
        
        edges_df['value'] = np.log1p(edges_df['value'])
        
        return edges_df

    
    # Get the data folders into respective cached pkl files
    def to_cached(self):
        """
        Send all of the processed datasets to pkl files for easy loading later
        
        Args:
            None
        
        Returns:
            None
        """
        normalization_datasets = ['networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']
        
        raw_data = [file for file in os.listdir(self.edgelist_dir)]
        raw_data = [file_name.replace('.txt', '') for file_name in raw_data]
        cached_data_folders = [file for file in os.listdir(self.output_dir)]

        # Betweenness and Closeness take too long to process and are deemed not feasible 
        activations = [EmbedDegree, EmbedForman, EmbedWeight, EmbedBetweenness, EmbedIncrementalCloseness]  # All activation functions to use
        activation_names = ['Degree', 'Forman', 'Weight', 'Betweenness', 'Closeness']
        # Need to do IncrementalBetweenness still
        
        missing_cached = []
        for dataset in raw_data:
            dataset_folder = os.path.join(self.output_dir, dataset)
            
            # Check if the folder exists
            if dataset not in cached_data_folders:
                missing_cached.append(dataset)
                continue  # Skip to the next dataset if the folder doesn't exist

            # Check for base file
            base_file = os.path.join(dataset_folder, f'{dataset}.pkl')
            if not os.path.exists(base_file):
                missing_cached.append(dataset)
                continue  # Skip to the next dataset if the base file is missing

            # Check for activation-specific files
            for activation_name in activation_names:
                activation_file = os.path.join(dataset_folder, f'{dataset}_{activation_name}.pkl')
                if not os.path.exists(activation_file):
                    missing_cached.append(dataset)
                    break  # Skip to the next dataset if any activation file is missing

                
        # If we are missing files, generate them
        if missing_cached:
            print(f'Generating pkl files for the following datasets: {missing_cached}')

            for file in missing_cached:
                # Generate base data with graphs, labels
                if file in normalization_datasets:
                    norm = True
                else:
                    norm = False
                    
                graphs = self.read_edges_directed(file, norm=norm)
                
                labels = self.read_labels(file)
                data = list(zip(graphs, labels))

                data_dir = self.output_dir + '/' + file
                os.makedirs(data_dir, exist_ok=True)
                base_graph_dir = data_dir + '/' + file + '.pkl'
                print(f'sending to {base_graph_dir}')
                with open(base_graph_dir, "wb") as f:
                    pickle.dump(data, f)
                    
                # Generate data with embeddings, labels
                for activation, activation_name in zip(activations, activation_names):
                    print(f'Generating for {file} with activation {activation_name}')
                    my_activation = activation(num_buckets=10)    
        
                    start_time = time.time()
        
                    # Since Forman Ricci requires directed edges
                    if activation==EmbedForman:
                        embeddings = my_activation.process_graphs_for_embeddings(graphs, is_directed=True)
                    else:
                        embeddings = my_activation.process_graphs_for_embeddings(graphs)
                        
                    end_time = time.time()
                    
                    print(f'Activation {activation_name} on dataset {file} took time {end_time - start_time}')
                        
                    self.verify_embeddings(embeddings, activation, file, norm=norm)    
                        
                    data = list(zip(embeddings, labels))

                    data_dir = self.output_dir + '/' + file
                    os.makedirs(data_dir, exist_ok=True)
                    activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}.pkl')
                    print(f'sending to {activation_file_path}')
                    with open(activation_file_path, "wb") as f:
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
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        graphs, labels = zip(*data)
        return graphs, labels