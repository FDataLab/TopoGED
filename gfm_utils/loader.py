import networkx as nx
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import time
import torch

# Update path for imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.probabilities_ahead import ProbsAhead
from utils.probabilities import Probs
from utils.utils import Utils
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.incremental_closeness import EmbedIncrementalCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

class Loader():
    # File paths
    output_dir = os.path.abspath('data/input/cached')
    edgelist_dir = os.path.abspath('data/input/raw/edgelist')
    label_dir = os.path.abspath('data/input/raw/labels')
    my_utils = Utils()

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


    def load_data(self, dataset, activation, type='features', include_weights=True, normalized=True, use_predicted=False, num_back='all', num_buckets=10):
        """
        Load a single, specified dataset that exists
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            graphs (list): A list of networkx graphs to process
            labels (list): The associated labels for each graph
        """
        self.to_cached()
        
        if type == 'subgraphs':
            seek_file = dataset + '_' + activation + f'_subgraphs_{str(num_buckets)}.pkl'
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
            dataset_folder = os.path.join(dataset_folder, 'subgraphs')  # Target folder path
        elif type == 'thresholds':
            seek_file = dataset + '_' + activation + f'_thresholds_{str(num_buckets)}.pkl'
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
            dataset_folder = os.path.join(dataset_folder, 'thresholds')  # Target folder path
        elif type == 'features' and include_weights == True:
            if use_predicted:
                # Due to later logic, we just return here
                seek_file = f'{dataset}_descriptor_noweight_{str(num_buckets)}.pkl'  # Based on dataset and activation combination
                dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
                dataset_folder = os.path.join(dataset_folder, 'predValues')
                seek_file_path = os.path.join(dataset_folder, f'{dataset}_descriptor_noweight_{str(num_buckets)}.pkl')
                with open(seek_file_path, "rb") as f:
                    data = pickle.load(f)
                return data, None  # I didn't return g/s labels making these pred values
            else:
                seek_file = dataset + '_' + activation + f'_{str(num_buckets)}.pkl'  # Based on dataset and activation combination
                dataset_folder = os.path.join(self.output_dir, dataset, 'descriptor_embeddings')  # Target folder path
        elif type == 'probabilities':
            seek_file = dataset + '_' + 'probabilities'
            dataset_folder = os.path.join(self.output_dir, dataset)
            
            if use_predicted:
                # We only have one possible path for this, so just return this
                dataset_folder = os.path.join(dataset_folder, 'predValues')
                seek_file_path = os.path.join(dataset_folder, f'{dataset}_probs_all_back.pkl')
                with open(seek_file_path, "rb") as f:
                    data = pickle.load(f)
                return data
            else:
                dataset_folder = os.path.join(dataset_folder, 'probabilities')
            if normalized:
                df = pd.read_csv(dataset_folder + f'/{dataset}_probabilities_{num_back}_back_norm.csv')
            else:
                df = pd.read_csv(dataset_folder + f'/{dataset}_probabilities_{num_back}_back.csv')
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")
            return df  # We just return the dataframe directly
        else:
            seek_file = dataset + '_' + activation + f'_no_weight_{str(num_buckets)}.pkl'  # Based on dataset and activation combination
            dataset_folder = os.path.join(self.output_dir, dataset)  # Target folder path
            dataset_folder = os.path.join(dataset_folder, 'descriptor_embeddings', 'no_weight')  # Target folder path
        
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
                probs = probs.drop(columns=["Unnamed: 0"], errors="ignore")
                return probs
        
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
    
    
    def gen_labels(self, graphs):
        """
        The labels are wrong, use this
        """
        print('Generating labels')
        labels = [1]
        for i in range(1, len(graphs)):
            labels.append(1 if graphs[i].number_of_edges() > graphs[i - 1].number_of_edges() else 0)
            
        return labels
        
    
    def get_snapshot_duration(self, dataset):
        if dataset in ['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 'networkaragon', 
                    'networkbancor', 'networkcentra', 'networkcoindash', 'networkiconomi', 'networkcindicator', 
                    'networkdgd', 'Reddit_B', 'tgbl-wiki', 'bitcoinotc', 'bitcoinalpha', 'uci-message']:
            return '1D'
        elif dataset in ['Hypertext09', 'ia-contact']:
            return '10min'
        elif dataset in ['enron']:
            return '11_BINS' # Special flag to force exactly 11 equal-edge snapshots
        elif dataset in ['radoslaw', 'fb-forum']:
            return '8H'
        elif dataset in ['HepPH', 'HepTH']:
            return '1M' # 1 Month
        elif dataset == 'tgbl-review':
            return '1W' # 1 Week
        else:
            return '1D' # Default fallback
    
    
    # Load the data from edge list txt file (Undirected)
    def read_edges(self, dataset):
        """
        Read the edgelists a file for later processing
        """
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = []
        edgelist_rawfile = self.edgelist_dir + '/{}.txt'.format(dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        edgelist_df = edgelist_df.drop_duplicates(subset=["from", "to", "date", "value"])
        
        if "Snapshot" in edgelist_df.columns:
            edgelist_df = edgelist_df.drop(columns=["Snapshot"])
        
        edgelist_df.to_csv(edgelist_rawfile, index=False)
        
        # --- TIME WINDOW GROUPING LOGIC ---
        freq = self.get_snapshot_duration(dataset)
        edgelist_df['date_dt'] = pd.to_datetime(edgelist_df['date'])
        
        # Sort chronologically so our bins are perfectly sequential
        edgelist_df = edgelist_df.sort_values('date_dt').reset_index(drop=True)

        if freq == '11_BINS':
            # Splits the sorted dataframe into exactly 11 equal-sized chunks (0 through 10)
            edgelist_df['group_ts'] = pd.qcut(edgelist_df.index, 11, labels=False)
        elif freq.endswith('M'):
            edgelist_df['group_ts'] = edgelist_df['date_dt'].dt.to_period('M').dt.start_time
        elif freq.endswith('W'):
            edgelist_df['group_ts'] = edgelist_df['date_dt'].dt.to_period('W').dt.start_time
        else:
            edgelist_df['group_ts'] = edgelist_df['date_dt'].dt.floor(freq)
            
        uniq_ts_list = sorted(edgelist_df['group_ts'].unique())

        # Loop over aggregated snapshot ids
        for ts in uniq_ts_list:
            ts_edges = edgelist_df.loc[edgelist_df['group_ts'] == ts, ['from', 'to']]
            ts_edges = ts_edges.drop_duplicates()
            
            # This automatically only includes nodes that have edges in this snapshot
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to')
            data.append(ts_G)
        
        return data
    
    
    # Load the data from edge list txt file (Directed)
    def read_edges_directed(self, dataset, norm=False):
        """
        Read the edgelists a file for later processing
        """
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = []
        edgelist_rawfile = self.edgelist_dir + '/{}.txt'.format(dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        edgelist_df = edgelist_df.drop_duplicates(subset=["from", "to", "date", "value"])
        
        if "Snapshot" in edgelist_df.columns:
            edgelist_df = edgelist_df.drop(columns=["Snapshot"])
        
        edgelist_df.to_csv(edgelist_rawfile, index=False)
        
        # --- TIME WINDOW GROUPING LOGIC ---
        freq = self.get_snapshot_duration(dataset)
        edgelist_df['date_dt'] = pd.to_datetime(edgelist_df['date'])
        
        # Sort chronologically so our bins are perfectly sequential
        edgelist_df = edgelist_df.sort_values('date_dt').reset_index(drop=True)
        
        if freq == '11_BINS':
            # Splits the sorted dataframe into exactly 11 equal-sized chunks (0 through 10)
            edgelist_df['group_ts'] = pd.qcut(edgelist_df.index, 11, labels=False)
        elif freq.endswith('M'):
            edgelist_df['group_ts'] = edgelist_df['date_dt'].dt.to_period('M').dt.start_time
        elif freq.endswith('W'):
            edgelist_df['group_ts'] = edgelist_df['date_dt'].dt.to_period('W').dt.start_time
        else:
            edgelist_df['group_ts'] = edgelist_df['date_dt'].dt.floor(freq)
            
        uniq_ts_list = sorted(edgelist_df['group_ts'].unique())
        
        if norm:
            edgelist_df = self.normalize_edge_weights(edgelist_df)
        
        # Loop over snapshots
        for ts in uniq_ts_list:
            ts_edges = edgelist_df.loc[edgelist_df['group_ts'] == ts, ['from', 'to', 'value']]
            ts_edges = ts_edges.groupby(['from', 'to'])['value'].sum().reset_index()
            
            # This automatically only includes nodes that have edges in this snapshot
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to', edge_attr=True, create_using=nx.DiGraph())
            
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
        normalization_datasets = []  # None for now
        
        raw_data = [file for file in os.listdir(self.edgelist_dir)]
        raw_data = [file_name.replace('.txt', '') for file_name in raw_data]
        # OGBL Datasets
        # raw_data.append('ogbl-collab')
        # raw_data.append('ogbl-citation2')
        
        cached_data_folders = [file for file in os.listdir(self.output_dir)]

        # Betweenness takes too long to process and are deemed not feasible 
        activations = [EmbedDegree, EmbedForman, EmbedWeight, EmbedBetweenness, EmbedIncrementalCloseness]  # All activation functions to use
        activation_names = ['Degree', 'Forman', 'Weight', 'Betweenness', 'Closeness']
        activations = [EmbedDegree]  # All activation functions to use
        activation_names = ['Degree']

        # If you want to use Betweenness, just run it here
        # activations = [EmbedBetweenness] 
        # activation_names = ['Betweenness']
        my_probs_generator = Probs()
        my_probs_ahead_generator = ProbsAhead()
        
        
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
                    
                
            # Check for embeddings 
            for activation_name in activation_names:
                for num_buckets in [5, 6, 8, 10, 12, 15, 20, 25, 30, 50]:
                    activation_file = os.path.join(dataset_folder, f'descriptor_embeddings/{dataset}_{activation_name}_{str(num_buckets)}.pkl')
                    if not os.path.exists(activation_file):
                        missing_cached.append(dataset)
                        break  # Skip to the next dataset if any activation file is missing
                    activation_file = os.path.join(dataset_folder + '/descriptor_embeddings/no_weight', f'{dataset}_{activation_name}_no_weight_{str(num_buckets)}.pkl')
                    if not os.path.exists(activation_file):
                        missing_cached.append(dataset)
                        break  # Skip to the next dataset if any activation file is missing
                
                    # Check for subgraphs
                    activation_file = os.path.join(dataset_folder + '/subgraphs', f'{dataset}_{activation_name}_subgraphs_{str(num_buckets)}.pkl')
                    if not os.path.exists(activation_file):
                        missing_cached.append(dataset)
                        break  # Skip to the next dataset if any activation file is missing
                    
                    # Check for thresholds
                    activation_file = os.path.join(dataset_folder + '/thresholds', f'{dataset}_{activation_name}_thresholds_{str(num_buckets)}.pkl')
                    if not os.path.exists(activation_file):
                        missing_cached.append(dataset)
                        break  # Skip to the next dataset if any activation file is missing
            
            # Check the probabilities
            probabilities_file = os.path.join(dataset_folder + '/probabilities', f'{dataset}_probabilities_all_back.csv')
            if not os.path.exists(probabilities_file):
                missing_cached.append(dataset)
            
            # Check the normalized probabilities
            probabilities_file = os.path.join(dataset_folder + '/probabilities', f'{dataset}_probabilities_all_back_norm.csv')
            if not os.path.exists(probabilities_file):
                missing_cached.append(dataset)
            
            for num_back in [5, 7]:
                probabilities_file = os.path.join(dataset_folder + '/probabilities', f'{dataset}_probabilities_{num_back}_back.csv')
                if not os.path.exists(probabilities_file):
                    missing_cached.append(dataset)
                    
            for k in [3, 5, 7, 10]:
                for num_back in [5, 7]:
                    probabilities_file = os.path.join(dataset_folder + '/probabilities_ahead', f'{dataset}_probabilities_{num_back}_back_k{k}.csv')
                    if not os.path.exists(probabilities_file):
                        missing_cached.append(dataset)
                        
                probabilities_file = os.path.join(dataset_folder + '/probabilities_ahead', f'{dataset}_probabilities_all_back_k{k}.csv')
                if not os.path.exists(probabilities_file):
                    missing_cached.append(dataset)
                
        
            
        # If we are missing files, generate them
        missing_cached = list(set(missing_cached))
        if missing_cached:
            print(f'Generating pkl files for the following datasets: {missing_cached}')

            for file in missing_cached:
                # Generate base data with graphs, labels
                if file in normalization_datasets:
                    norm = True
                else:
                    norm = False
                    
                print('Reading graphs')
                graphs = self.read_edges_directed(file, norm=norm)
                
                print(f'There are {len(graphs)} graphs in this dataset')
                
                labels = self.gen_labels(graphs)
                print(f'There are {len(labels)} labels')
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
                        for num_buckets in [5, 6, 8, 10, 12, 15, 20, 25, 30, 50]:
                            print(f'Generating for {file} with activation {activation_name} with include_weights={weight_flag} and num_buckets={str(num_buckets)}')
                            my_activation = activation(num_buckets=num_buckets, include_weights=weight_flag)    
                
                            start_time = time.time()
                
                            # Since Forman Ricci requires directed edges
                            if isinstance(activation, EmbedForman):
                                embeddings, subgraphs, thresholds = my_activation.process_graphs_for_embeddings(graphs, is_directed=True)
                            else:
                                embeddings, subgraphs, thresholds = my_activation.process_graphs_for_embeddings(graphs)
                                
                            end_time = time.time()
                            
                            print(f'There were {len(embeddings)} embeddings generated')
                            print(f'Activation {activation_name} on dataset {file} took time {end_time - start_time}')
                                
                            self.verify_embeddings(embeddings, activation, file, norm=norm, include_weights=weight_flag)    
                                
                            data = list(zip(embeddings, labels))

                            # Different processing directory
                            if weight_flag == True:
                                data_dir = self.output_dir + '/' + file
                                os.makedirs(data_dir, exist_ok=True)
                                activation_file_path = os.path.join(data_dir, f'descriptor_embeddings/{file}_{activation_name}_{str(num_buckets)}.pkl')
                                print(f'sending to {activation_file_path}')
                                with open(activation_file_path, "wb") as f:
                                    pickle.dump(data, f)
                                    
                            else:
                                data_dir = self.output_dir + '/' + file + '/descriptor_embeddings/no_weight'
                                os.makedirs(data_dir, exist_ok=True)
                                activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}_no_weight_{str(num_buckets)}.pkl')
                                print(f'sending to {activation_file_path}')
                                with open(activation_file_path, "wb") as f:
                                    pickle.dump(data, f)
                            
                            data_dir = self.output_dir + '/' + file + '/subgraphs'
                            os.makedirs(data_dir, exist_ok=True)
                            activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}_subgraphs_{str(num_buckets)}.pkl')
                            print(f'sending subgraphs to {activation_file_path}')
                            with open(activation_file_path, "wb") as f:
                                pickle.dump(subgraphs, f)
                                
                            data_dir = self.output_dir + '/' + file + '/thresholds'
                            os.makedirs(data_dir, exist_ok=True)
                            activation_file_path = os.path.join(data_dir, f'{file}_{activation_name}_thresholds_{str(num_buckets)}.pkl')
                            print(f'sending thresholds to {activation_file_path}')
                            with open(activation_file_path, "wb") as f:
                                pickle.dump(thresholds, f)
                            
                # Add on the probabiliites
                for num_back in [5, 7]:
                    probs = my_probs_generator.gen_probs(num_graphs_back = num_back, graphs=graphs, from_start=False)
                    print(f'There were {len(probs)} probabilities generated')
                    data_dir = self.output_dir + '/' + file + '/probabilities'
                    os.makedirs(data_dir, exist_ok=True)
                    probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_{num_back}_back.csv')
                    df = pd.DataFrame(probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                    df.to_csv(probabilities_file_path)
                    norm_probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_{num_back}_back_norm.csv')
                    normalized_probs = np.vstack(df.apply( lambda row: self.my_utils.normalize_vector_by_groups(row.values), axis=1))
                    df = pd.DataFrame(normalized_probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                    df.to_csv(norm_probabilities_file_path)
                    
                    
                probs = my_probs_generator.gen_probs(num_graphs_back = 1, graphs=graphs, from_start=True)
                data_dir = self.output_dir + '/' + file + '/probabilities'
                os.makedirs(data_dir, exist_ok=True)
                probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_all_back.csv')
                df = pd.DataFrame(probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                df.to_csv(probabilities_file_path)
                print(f'Sending probabilities to {probabilities_file_path}')
                norm_probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_all_back_norm.csv')
                normalized_probs = np.vstack(df.apply( lambda row: self.my_utils.normalize_vector_by_groups(row.values), axis=1))
                df = pd.DataFrame(normalized_probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                df.to_csv(norm_probabilities_file_path)
                
                
                for k in [3, 5, 7, 10]:
                    for num_back in [5, 7]:
                        probs = my_probs_ahead_generator.gen_probs(num_graphs_back = num_back, graphs=graphs, from_start=False, k=k)
                        print(f'There were {len(probs)} probabilities generated')
                        data_dir = self.output_dir + '/' + file + '/probabilities_ahead'
                        os.makedirs(data_dir, exist_ok=True)
                        probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_{num_back}_back_k{k}.csv')
                        df = pd.DataFrame(probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                        df.to_csv(probabilities_file_path)
                        norm_probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_{num_back}_back_norm_k{k}.csv')
                        normalized_probs = np.vstack(df.apply( lambda row: self.my_utils.normalize_vector_by_groups(row.values), axis=1))
                        df = pd.DataFrame(normalized_probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                        df.to_csv(norm_probabilities_file_path)
                        
                    probs = my_probs_ahead_generator.gen_probs(num_graphs_back = 1, graphs=graphs, from_start=True, k=k)
                    data_dir = self.output_dir + '/' + file + '/probabilities_ahead'
                    os.makedirs(data_dir, exist_ok=True)
                    probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_all_back_k{k}.csv')
                    df = pd.DataFrame(probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                    df.to_csv(probabilities_file_path)
                    print(f'Sending probabilities to {probabilities_file_path}')
                    norm_probabilities_file_path = os.path.join(data_dir, f'{file}_probabilities_all_back_norm_k{k}.csv')
                    normalized_probs = np.vstack(df.apply( lambda row: self.my_utils.normalize_vector_by_groups(row.values), axis=1))
                    df = pd.DataFrame(normalized_probs, columns=["Prob Old Nodes", "Prob New Nodes", "Prob OO", "Prob NN", "Prob ON", "Prob OON"])
                    df.to_csv(norm_probabilities_file_path)


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
    
    def load_HTGN_data(dataset):
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = {}
        edgelist_rawfile = '../data/input/raw/{}/{}_edgelist.txt'.format(dataset, dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        uniq_ts_list = np.unique(edgelist_df['snapshot'])
        print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))
        adj_time_list = []
        for ts in uniq_ts_list:
            # NOTE: this code does not use any node or edge features
            ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['from', 'to']]
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to')
            ts_A = nx.to_scipy_sparse_array(ts_G)
            adj_time_list.append(ts_A)

        # Now, exactly like "load_vgrnn_dataset_det"
        print('INFO: Generating edges, negative edges and new edges, wait for a while ...')
        edge_proc_start = time.time()
        data = {}
        edges, biedges = mask_edges_det(adj_time_list)  # list
        new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
        print('INFO: Processing finished! Elapsed time (sec.): {:.4}'.format(time.time() - edge_proc_start))
        assert len(edges) == len(biedges) == len(new_nedges) == len(new_pedges)
        edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
        for t in range(len(biedges)):
            edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
            pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
            nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
            new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
            new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

        data['edge_index_list'] = edge_index_list
    

        data['time_length'] = len(edge_index_list)
        data['weights'] = None
        print('INFO: Data: {}'.format(dataset))
        print('INFO: Total length:{}'.format(len(edge_index_list)))
        print('INFO: Number nodes: {}'.format(data['num_nodes']))
        return data


    