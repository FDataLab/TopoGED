import networkx as nx
import pandas as pd
import numpy as np
import pickle as pkl
import torch
import time

from sklearn.metrics import roc_auc_score 
from my_utils.utils import Utils
import scipy.sparse as sp

class DataLoader():
    edge_list_path = 'data/graphpulse/'
    gs_label_path = 'data/graphpulse/labels/'

    def __init__(self):
        pass 



    def process_predictions(self, lstm_predictions):
        nodes = []
        edges = []
        densities = [] 

        # Generate the number of nodes, edges, and density of a graph per each snapshot

        return nodes, edges, densities
                                                      

    def read_labels(self, dataset):
        labels = pd.Series.from_csv(self.gs_label_path + '{}_Label.csv'.format(dataset))
        return labels
    

    def to_labels(self, embeddings):
        pred = []

        for i in range(1, len(embeddings)):
            label = 1 if (embeddings[i][-1] - embeddings[i - 1][-1]) > 0 else 0
            pred.append(label)

        return pred
    

    def load_data(self, dataset):
        print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
        data = []
        edgelist_rawfile = self.edge_list_path + '{}.txt'.format(dataset)
        edgelist_df = pd.read_csv(edgelist_rawfile)
        uniq_ts_list = np.unique(edgelist_df['Snapshot'])
        print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))

        for ts in uniq_ts_list:
            # NOTE: this code does not use any node or edge features
            ts_edges = edgelist_df.loc[edgelist_df['Snapshot'] == ts, ['from', 'to']]
            ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to')
            data.append(ts_G)
        
        return data


    def compute_results(self, pred, truth):
        pred = np.array(pred)  # Make it a numpy array because it isn't already
        rocauc = roc_auc_score(pred, truth)
        print(f'The predidctions from the LSTMGRU had an ROCAUC score of: {rocauc}')
