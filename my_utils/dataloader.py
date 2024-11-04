import networkx as nx
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score 

class MyDataLoader():
    # File paths
    edge_list_path = 'data/graphpulse/'
    gs_label_path = 'data/graphpulse/labels/'


    # Read the labels from the GraphPulse datasets
    def read_labels(self, dataset):
        labels = pd.read_csv(self.gs_label_path + '{}_Label.csv'.format(dataset), header=None)
        return labels
    

    # Generate the labels for the LSTM-GRU predictions
    def to_labels(self, embeddings):
        pred = []

        for i in range(1, len(embeddings)):
            print(f'{embeddings[i][-1]} vs {embeddings[i - 1][-1]}')
            label = 1 if (embeddings[i][-1] - embeddings[i - 1][-1]) > 0 else 0
            pred.append(label)

        return pred
    

    # Load the data from edge list txt file
    def load_data(self, dataset):
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = []
        edgelist_rawfile = self.edge_list_path + '{}.txt'.format(dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        uniq_ts_list = np.unique(edgelist_df['Snapshot'])
        print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))

        # Loop over snapshot ids
        for ts in uniq_ts_list:
            # NOTE: this code does not use any node or edge features
            ts_edges = edgelist_df.loc[edgelist_df['Snapshot'] == ts, ['from', 'to']]
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to')
            data.append(ts_G)
        
        return data


    # Compare the predictions and truth using desired metrics
    def compute_results(self, pred, truth):
        rocauc = roc_auc_score(pred, truth)
        print(f'The predidctions from the LSTMGRU had an ROCAUC score of: {rocauc}')
        return rocauc
