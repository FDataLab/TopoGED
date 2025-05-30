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

from utils.probabilities import Probs
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

    def verify_embeddings(self, embeddings, activation, dataset, norm=False, include_weights=True):
        print(f'Verifying {activation} on dataset {dataset}')
        if activation == EmbedForman and dataset == 'Reddit_B':
            return 
        
        if not norm:
            graphs = self.read_edges_directed(dataset)
            for embedding, graph in zip(embeddings, graphs):
                total_nodes = graph.number_of_nodes()
                total_edges = graph.number_of_edges()
                total_weight = sum(data['value'] for _, _, data in graph.edges(data=True))
                
                if include_weights == True:
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
                else:
                    embedding_nodes = embedding[-2]
                    embedding_edges = embedding[-1]
                    
                    if total_nodes != embedding_nodes:
                        print(f'A GRAPH FROM {dataset} WITH ACTIVATION {activation} HAS THE WRONG NUMBER OF NODES')
                        print(f'TRUE VAL: {total_nodes}; EMBEDDING VAL: {embedding_nodes}')
                    
                    if total_edges != embedding_edges:
                        print(f'A GRAPH FROM {dataset} WITH ACTIVATION {activation} HAS THE WRONG NUMBER OF EDGES')
                        print(f'TRUE VAL: {total_edges}; EMBEDDING VAL: {embedding_edges}')


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


    def load_data(self, dataset, activation, type='features', include_weights=True):
        """
        Load a single, specified dataset that exists
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            graphs (list): A list of networkx graphs to process
            labels (list): The associated labels for each graph
        """
        # self.to_cached()
        if type == 'subgraphs':
            seek_file = dataset + '_' + activation + '_subgraphs.pkl'
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
            dataset_folder = os.path.join(dataset_folder, 'subgraphs')  # Target folder path
        elif type == 'thresholds':
            seek_file = dataset + '_' + activation + '_thresholds.pkl'
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
            dataset_folder = os.path.join(dataset_folder, 'thresholds')  # Target folder path
        elif type == 'features' and include_weights == True:
            seek_file = dataset + '_' + activation + '.pkl'  # Based on dataset and activation combination
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
        elif type == 'probabilities':
            seek_file = dataset + '_' + 'probabilities'
            dataset_folder = os.path.join(self.output_dir, dataset)
            dataset_folder = os.path.join(dataset_folder, 'probabilities')
            return pd.read_csv(dataset_folder + f'/{dataset}_probabilities.csv')  # We just return the dataframe directly
        else:
            seek_file = dataset + '_' + activation + '_no_weight' + '.pkl'  # Based on dataset and activation combination
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
            dataset_folder = os.path.join(dataset_folder, 'no_weight')  # Target folder path
            
        data_files = os.listdir(dataset_folder)
        
        if seek_file in data_files:
            # Different processing logic
            if type == 'subgraphs':
                seek_file_path = os.path.join(dataset_folder, seek_file)
                graphs = self.from_cached(seek_file_path, type)  # Load data
                return graphs
            elif type == 'thresholds':
                seek_file_path = os.path.join(dataset_folder, seek_file)
                thresholds = self.from_cached(seek_file_path, type)  # Load data
                return thresholds
            elif type == 'features':
                seek_file_path = os.path.join(dataset_folder, seek_file)
                graphs, labels = self.from_cached(seek_file_path, type)  # Load data
                return graphs, labels
            elif type == 'probabilities':
                seek_file_path = os.path.join(dataset_folder, seek_file)
                probs = pd.read_csv(seek_file_path)
                return probs
        
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
    
    
    def gen_labels(self, graphs):
        """
        The labels are wrong, use this
        """
        print('Generating labels')
        labels = [1]
        for i in range(1, len(graphs)):
            labels.append(1 if graphs[i].number_of_edges() > graphs[i - 1].number_of_edges() else 0)
            
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
        normalization_datasets = ['networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']  # Cuneyt recommended normalizing these datasets due to large edge weights messing up data
        
        raw_data = [file for file in os.listdir(self.edgelist_dir)]
        raw_data = [file_name.replace('.txt', '') for file_name in raw_data]
        cached_data_folders = [file for file in os.listdir(self.output_dir)]

        # Betweenness takes too long to process and are deemed not feasible 
        activations = [EmbedDegree, EmbedForman, EmbedWeight, EmbedBetweenness, EmbedIncrementalCloseness]  # All activation functions to use
        activation_names = ['Degree', 'Forman', 'Weight', 'Betweenness', 'Closeness']
        activations = [EmbedDegree, EmbedForman, EmbedWeight, EmbedIncrementalCloseness]  # All activation functions to use
        activation_names = ['Degree', 'Forman', 'Weight', 'Closeness']
        # If you want to use Betweenness, just run it here
        # activations = [EmbedBetweenness] 
        # activation_names = ['Betweenness']
        my_probs_generator = Probs()
        
        
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
                
            # Check for no weight embeddings 
            for activation_name in activation_names:
                activation_file = os.path.join(dataset_folder + '/no_weight', f'{dataset}_{activation_name}_no_weight.pkl')
                if not os.path.exists(activation_file):
                    missing_cached.append(dataset)
                    break  # Skip to the next dataset if any activation file is missing
                
                # Check for subgraphs
                activation_file = os.path.join(dataset_folder + '/subgraphs', f'{dataset}_{activation_name}_subgraphs.pkl')
                if not os.path.exists(activation_file):
                    missing_cached.append(dataset)
                    break  # Skip to the next dataset if any activation file is missing
                
                # Check for thresholds
                activation_file = os.path.join(dataset_folder + '/thresholds', f'{dataset}_{activation_name}_thresholds.pkl')
                if not os.path.exists(activation_file):
                    missing_cached.append(dataset)
                    break  # Skip to the next dataset if any activation file is missing
                
            # Check the probabilities (we just do all_back)
            probabilities_file = os.path.join(dataset_folder + '/probabilities', f'{dataset}_probabilities.csv')
            if not os.path.exists(probabilities_file):
                missing_cached.append(dataset)
        
                
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
                
                labels = self.gen_labels(graphs)
                data = list(zip(graphs, labels))

                data_dir = self.output_dir + '/' + file
                os.makedirs(data_dir, exist_ok=True)
                base_graph_dir = data_dir + '/' + file + '.pkl'
                print(f'sending to {base_graph_dir}')
                with open(base_graph_dir, "wb") as f:
                    pickle.dump(data, f)
                    
                # Generate data with embeddings, labels
                for activation, activation_name in zip(activations, activation_names):
                    for weight_flag in [False, True]:  # Need to go back with only weight = True
                        print(f'Generating for {file} with activation {activation_name} with include_weights={weight_flag}')
                        my_activation = activation(num_buckets=10, include_weights=weight_flag)    
            
                        start_time = time.time()
            
                        # Since Forman Ricci requires directed edges
                        if activation==EmbedForman:
                            embeddings, subgraphs, thresholds = my_activation.process_graphs_for_embeddings(graphs, is_directed=True)
                        else:
                            embeddings, subgraphs, thresholds = my_activation.process_graphs_for_embeddings(graphs)
                            
                        end_time = time.time()
                        
                        print(f'Activation {activation_name} on dataset {file} took time {end_time - start_time}')
                            
                        self.verify_embeddings(embeddings, activation, file, norm=norm, include_weights=weight_flag)    
                            
                        data = list(zip(embeddings, labels))

                        # Different processing directory
                        if weight_flag == True:
                            data_dir = self.output_dir + '/' + file
                            os.makedirs(data_dir, exist_ok=True)
                            activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}.pkl')
                            print(f'sending to {activation_file_path}')
                            with open(activation_file_path, "wb") as f:
                                pickle.dump(data, f)
                                
                        else:
                            data_dir = self.output_dir + '/' + file + '/no_weight'
                            os.makedirs(data_dir, exist_ok=True)
                            activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}_no_weight.pkl')
                            print(f'sending to {activation_file_path}')
                            with open(activation_file_path, "wb") as f:
                                pickle.dump(data, f)
                        
                        data_dir = self.output_dir + '/' + file + '/subgraphs'
                        os.makedirs(data_dir, exist_ok=True)
                        activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}_subgraphs.pkl')
                        print(f'sending subgraphs to {activation_file_path}')
                        with open(activation_file_path, "wb") as f:
                            pickle.dump(subgraphs, f)
                            
                        data_dir = self.output_dir + '/' + file + '/thresholds'
                        os.makedirs(data_dir, exist_ok=True)
                        activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}_thresholds.pkl')
                        print(f'sending thresholds to {activation_file_path}')
                        with open(activation_file_path, "wb") as f:
                            pickle.dump(thresholds, f)
                            
                # Add on the probabiliites
                probs = my_probs_generator.gen_probs(num_graphs_back = 1, graphs=graphs, from_start=True)
                data_dir = self.output_dir + '/' + file + '/probabilities'
                os.makedirs(data_dir, exist_ok=True)
                probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities.csv')
                df = pd.DataFrame(probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                df.to_csv(probabilities_file_path)
                print(f'Sending probabilities to {probabilities_file_path}')


    # Load from the pkl file
    def from_cached(self, file_name, type):
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

        if type == 'features':
            graphs, labels = zip(*data)
            return graphs, labels
        
        else:
            return data