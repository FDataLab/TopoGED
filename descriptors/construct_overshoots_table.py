import numpy as np
import pickle
import io
import pandas as pd
import sys
import os

# Ensure the loader can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.loader import Loader 

def construct_tables(pred_vectors, true_vectors):
    pred = np.array(pred_vectors)
    true = np.array(true_vectors)
    epsilon = 1e-9
    
    overshoot = np.sum(pred > true) / len(pred) * 100
    undershoot = np.sum(pred < true) / len(pred) * 100
    percent_error = np.mean(np.abs(pred - true) / (true + epsilon)) * 100
    
    return overshoot, undershoot, percent_error

if __name__ == "__main__":
    datasets = ["CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion", 
                "networkaragon", "networkbancor", "networkcentra", "networkcindicator", 
                "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B", "tgbl-wiki"]

    my_loader = Loader()
    

    for use_test in [True, False]:
        results = []
        for dataset in datasets:
            # 1. Load Preds
            with open(f"data/input/cached/{dataset}/predValues/{dataset}_testdescriptors_VECM_Raw_10.pkl", "rb") as f:
                pred_features = pickle.load(f)
        
            # 2. Load Truth
            tmp_true_features, _ = my_loader.load_data(dataset, activation='Degree', type='features', 
                                                    use_predicted=False, include_weights=False, num_buckets=10)
            
            if use_test:
                test_set_len = len(pred_features) * 0.15
                test_pred_features = pred_features[-int(test_set_len):]
                test_true_features = tmp_true_features[-int(test_set_len):]
            
            else:
                test_pred_features = pred_features
                test_true_features = tmp_true_features
            
            # 3. Unpack (Fixed your indices)
            pred_nodes = np.array([test_pred_features[i][-2] for i in range(len(test_pred_features))])
            pred_edges = np.array([test_pred_features[i][-1] for i in range(len(test_pred_features))])
            true_nodes = np.array([test_true_features[i][-2] for i in range(len(test_true_features))])
            true_edges = np.array([test_true_features[i][-1] for i in range(len(test_true_features))])


            # 4. Calculate
            os_n, us_n, pe_n = construct_tables(pred_nodes, true_nodes)
            os_e, us_e, pe_e = construct_tables(pred_edges, true_edges)
            
            results.append({
                "dataset": dataset,
                "os_n": os_n, "us_n": us_n, "pe_n": pe_n,
                "os_e": os_e, "us_e": us_e, "pe_e": pe_e
            })

        # --- Improved Table Generation ---
        df = pd.DataFrame(results)

        # Clean names
        df["dataset"] = df["dataset"].str.replace("network", "", case=False).str.replace("tgbl-", "", case=False)
        df["dataset"] = df["dataset"].apply(lambda x: x.capitalize() if x != "mathoverflow" else x)
        df["dataset"] = df["dataset"].str.replace("_", r"\_")

        # Generate data rows only (7 columns)
        tabular_content = df.to_latex(index=False, header=False, escape=False, 
                                    float_format="%.3f", column_format="lcccccc").strip()
        
        # Extract just the rows (everything after the start of tabular)
        data_rows = tabular_content.split(r"\toprule")[-1].split(r"\bottomrule")[0].strip()

        # Manual Header Construction for the "Nodes & Edges" look
        latex_table = [ 
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{VECM Prediction Performance: Nodes vs Edges}",
            r"\label{tab:vecm_results}",
            r"\small",
            r"\begin{tabular}{@{}l ccc ccc@{}}",
            r"\toprule",
            r" & \multicolumn{3}{c}{\textbf{Nodes}} & \multicolumn{3}{c}{\textbf{Edges}} \\",
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
            r"\textbf{Dataset} & \% Overshoot & \% Undershoot & Abs \% Err & \% Overshoot & \% Undershoot & Abs \% Err \\",
            r"\midrule",
            data_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ]

        with open(f"latex_tables/overshoots_{"test_only" if use_test else "all_data"}.tex", "w") as f:
            f.write("\n".join(latex_table))

        print(f"LaTeX table (Nodes/Edges format) has been saved to latex_tables/overshoots_{"test_only" if use_test else "all_data"}.tex")