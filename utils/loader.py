import networkx as nx
import pandas as pd
import numpy as np
import pickle
import networkx as nx

# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Loader():
    # File paths
    output_dir = os.path.abspath('data/input/cached')
    output_directed_dir = os.path.abspath('data/input/cached_directed')
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
        data_files = [file for file in os.listdir(self.output_directed_dir)]
        for file in data_files:
            all_data.append(self.from_cached(file))
        
        return all_data

    def load_data(self, dataset):
        """
        Load a single, specified dataset that exists
        
        Args:
            dataset (string): The name of the dataset to load
        
        Returns:
            graphs (list): A list of networkx graphs to process
            labels (list): The associated labels for each graph
        """
        self.to_cached()
        data_files = [file for file in os.listdir(self.output_directed_dir)]
        if dataset in data_files:
            graphs, labels = self.from_cached(dataset)
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
        cached_data = [file for file in os.listdir(self.output_directed_dir)]
        missing_cached = [item for item in raw_data if item not in cached_data]

        # If we are missing files, generate them
        if missing_cached:
            print(f'Generating pkl files for the following datasets: {missing_cached}')

            for file in missing_cached:
                graphs = self.read_edges_directed(file)
                labels = self.read_labels(file)
                data = list(zip(graphs, labels))

                data_dir = self.output_directed_dir + '/' + file
                os.makedirs(data_dir, exist_ok=True)
                print('sending to ' + data_dir + '/' + file + '.pkl')
                with open(data_dir + '/' + file + '.pkl', "wb") as f:
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
        with open(self.output_directed_dir + '/' + file_name + '/' + file_name + '.pkl', "rb") as f:
            data = pickle.load(f)

        graphs, labels = zip(*data)
        return graphs, labels