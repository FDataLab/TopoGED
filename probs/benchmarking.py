import pandas as pd
import numpy as np
import os

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.utils import Utils
from utils.benchmarker import Benchmarker


if __name__ == "__main__":
    my_loader = Loader()
    my_utils = Utils()
    my_benchmarker = Benchmarker()    
    my_utils.set_seeds(42)
    window_size=7
    
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    for dataset in datasets:
        probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  
    
        embeddings = probabilities_df.apply(
            lambda row: my_utils.normalize_vector_by_groups(row.values),
            axis=1
        )
        embeddings = np.vstack(embeddings.values).astype(np.float32) 
                
                    
        # Split data 80/10/10
        n = len(embeddings)
        train_end = int(0.8 * n)
        val_start = train_end - window_size
        val_end = int(0.9 * n)
        test_start = val_end - window_size

        embeddings = np.array([np.array(e, dtype=np.float32) for e in embeddings])

        X_train = embeddings[:train_end]
        X_val = embeddings[val_start:val_end]
        X_test = embeddings[test_start:]
        
        # For benchmarking we don't need to predict the true values, just get losses for now
        my_benchmarker.begin_benchmarking(dataset, X_train, X_val, X_test, window_size=window_size)
