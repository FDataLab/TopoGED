import pandas as pd
import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def export_tgb_datasets(dataset_list, root_dir=".GraphGeneration/utils/tgb_data"):
    """
    Exports specified TGB datasets to CSV format: from, to, date, value
    """
    for name in dataset_list:
        print(f"\n--- Processing: {name} ---")
        
        # root specifies where the raw and processed TGB files are stored
        dataset = LinkPropPredDataset(name=name, root=root_dir, preprocess=True)
        data = dataset.full_data
        
        src = data['sources']
        dst = data['destinations']
        ts = data['timestamps']
        
        if 'w' in data:
            values = data['w']
        elif 'edge_feat' in data and data['edge_feat'].ndim == 1:
            values = data['edge_feat']
        else:
            values = np.ones(len(src), dtype=np.float32)

        # TGB timestamps are UNIX seconds. We convert to YYYY-MM-DD
        print(f"Converting {len(src):,} edges to dataframe...")
        df = pd.DataFrame({
            'from': src,
            'to': dst,
            'date': pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d'),
            'value': values
        })

        # 5. Save Output
        output_file = f"data/input/raw/edgelist/{name}.txt"
        df.to_csv(output_file, index=False)
        print(f"Successfully saved to: {output_file}")

# Run for both Wiki and Amazon
datasets_to_run = ["tgbl-wiki", "tgbl-review"]
export_tgb_datasets(datasets_to_run)

