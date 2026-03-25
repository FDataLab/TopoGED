# This file is for testing using top-k for graph construction. It supports both caching and loading the top-k values for each dataset
# This is to see how the power of the top-k approach is different compared to thresholding 
# Two options:
    # 1. Use the top-k values from the True TopER Vectors 
    # 2. Use the top-k values from the Pred TopER Vectors (this is more realistic for the actual use case)
    
import os
import sys
import pickle
import numpy as np
import importlib.util

from utils.loader import Loader



def get_topk(dataset, use_true=False):
    my_loader = Loader()
    print(f'Processing dataset: {dataset} with use_true={use_true}')
    if use_true:
        features, _ = my_loader.load_data(dataset, activation='Degree', type='features', use_predicted=False, include_weights=False, num_buckets=10)
    else:  # Current way we are doing pred values is with VAR
        with open(f'data/input/cached/{dataset}/predValues/{dataset}_testtoper_VAR_Delta_10.pkl', 'rb') as f:  # Num buckets doesnt really matter here, as all graphs have the same number of edges
            features = pickle.load(f)
            
    # The true values are set up as a list of tuples (nodes, edges)
    if isinstance(features[0][0], tuple):
        top_k_values = [feature[-1][-1] for feature in features]  
    # Add np.number here to catch numpy.float32 and numpy.int64!
    elif isinstance(features[0][0], (int, float, np.number)): 
        top_k_values = [feature[-1] for feature in features]  
    else:
        print(f'Received type: {type(features[0][0])}')
        raise ValueError("Unexpected feature format. Expected list of tuples or list of numbers.")
    
    return top_k_values
            
def load_cached_topk(dataset, use_true=False):
    # Make processing start here instead of running if __name__ == "__main__"  
    INPUT_DIR = 'benchmarkers/data/input/cached/'
    file_path = os.path.join(INPUT_DIR, f'top_k_values_{"true" if use_true else "pred"}/{dataset}_top_k_values.pkl')
    with open(file_path, 'rb') as f:
        top_k_values = pickle.load(f)
    return top_k_values
    