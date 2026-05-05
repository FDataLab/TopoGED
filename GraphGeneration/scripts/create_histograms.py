import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_structure(real_csv_path, pred_csv_path, diff_path, output_path):
    selected_columns = [
        "Average Node Degree",
        "Assortivity Coefficient",
        "Clustering Coefficient",
        "Degree Centrality",
        'Unique Degree Count',
        'Density',
        'Number of Weakly Connected Components',
        'Number of Strongly Connected Components',
        'Number of Nodes',
        'Number of Edges',
    ]

    real_df = pd.read_csv(real_csv_path)
    pred_df = pd.read_csv(pred_csv_path)
    diff_df = pd.read_csv(diff_path)

    # To compare when we don't have enough data for both
    if 'CollegeMsg' in real_csv_path:
        common_index = 86
    elif 'mathoverflow' in real_csv_path:
        common_index = 128
    elif 'networkadex' in real_csv_path:
        common_index = 94
    elif 'networkaeternity' in real_csv_path:
        common_index = 115
    elif 'networkaion' in real_csv_path:
        common_index = 123
    elif 'networkaragon' in real_csv_path:
        common_index = 95
    elif 'networkbancor' in real_csv_path:
        common_index = 76
    elif 'networkcentra' in real_csv_path:
        common_index = 107
    elif 'networkcoindash' in real_csv_path:
        common_index = 74
    elif 'Reddit_B' in real_csv_path:
        common_index = 93
        
        
    #common_index = real_df.index.intersection(pred_df.index)
    #real_df = real_df.loc[common_index].reset_index(drop=True)
    #pred_df = pred_df.loc[common_index].reset_index(drop=True)
    original_len = len(real_df)  # Store the original number of rows

    real_df = real_df.loc[:common_index - 1].reset_index(drop=True)
    pred_df = pred_df.loc[:common_index - 1].reset_index(drop=True)

    percent_used = common_index / original_len * 100 if original_len > 0 else 0

    print(f'Running on: {real_csv_path}')
    print(f"[INFO] Using {common_index} of {original_len} rows from real_df ({percent_used:.2f}%)")

    # Set up output path
    output_dir_histograms = os.path.join(os.path.dirname(output_path), "histogramsPartial")
    os.makedirs(output_dir_histograms, exist_ok=True)
    output_dir_linechart = os.path.join(os.path.dirname(output_path), "linechartPartial")
    os.makedirs(output_dir_linechart, exist_ok=True)

    sns.set(style="whitegrid")
    for metric in selected_columns:
        if metric not in real_df.columns or metric not in pred_df.columns:
            print(f"[WARNING] Column '{metric}' not found in both CSVs. Skipping.")
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(real_df[metric], color='blue', label='True', stat='count', kde=True, bins=30)
        sns.histplot(pred_df[metric], color='red', label='Pred', stat='count', kde=True, bins=30)
        plt.title(f"{metric} Distribution")
        plt.xlabel(metric)
        plt.ylabel("Value")
        plt.legend(title="Legend", loc="upper right")
        plt.tight_layout()

        output_file = os.path.join(output_dir_histograms, f"{metric.replace(' ', '_')}_histogram.png")
        
        plt.savefig(output_file)
        plt.close()
        #print(f"[SAVED] {output_file}")
        
        # Line chart
        plt.figure(figsize=(8, 5))
        plt.plot(real_df.index, real_df[metric], label='True', color='blue', marker='o')
        plt.plot(pred_df.index, pred_df[metric], label='Pred', color='red', marker='x')
        plt.title(f"{metric} Over Time")
        plt.xlabel("Graph Index")
        plt.ylabel(metric)
        plt.legend(title="Legend", loc="best")
        plt.grid(True)
        plt.tight_layout()
        line_file = os.path.join(output_dir_linechart, f"{metric.replace(' ', '_')}_linechart.png")
        plt.savefig(line_file)
        plt.close()
        #print(f"[SAVED] {line_file}")
        
        
    # Now make one for kernel distance
    metric = 'Kernel Distance'
    plt.figure(figsize=(8, 5))
    sns.histplot(diff_df[metric], color='blue', stat='count', kde=True, bins=30)
    plt.title(f"{metric} Distribution")
    plt.xlabel(metric)
    plt.ylabel("Value")
    plt.legend(title="Legend", loc="upper right")
    plt.tight_layout()

    output_file = os.path.join(output_dir_histograms, f"{metric.replace(' ', '_')}_histogram.png")
    
    plt.savefig(output_file)
    plt.close()
    #print(f"[SAVED] {output_file}")
    
    # Line chart
    plt.figure(figsize=(8, 5))
    plt.plot(diff_df.index, diff_df[metric], color='blue', marker='o')
    plt.title(f"{metric} Over Time")
    plt.xlabel("Graph Index")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    line_file = os.path.join(output_dir_linechart, f"{metric.replace(' ', '_')}_linechart.png")
    plt.savefig(line_file)
    plt.close()
    #print(f"[SAVED] {line_file}")
    
    # Print mean and median of each column in diff_df
    print(f"\n[STATS] Mean and Median for each metric in diff_df for {output_path}:")
    print(f"{'Metric':<45} {'Mean':>12} {'Median':>12} {'MSE':>12}")
    print("-" * 85)

    selected_columns.append('Kernel Distance')
    for column in selected_columns:
        mean_val = diff_df[column].mean()
        median_val = diff_df[column].median()
        mse_val = (diff_df[column] ** 2).mean()
        print(f"{column:<45} {mean_val:>12.4f} {median_val:>12.4f} {mse_val:>12.4f}")
        
        
def plot_kernel(real_csv_path, pred_csv_path, output_path):
    sns.set(style="whitegrid")

    real_df = pd.read_csv(real_csv_path, header=None, skiprows=1)
    pred_df = pd.read_csv(pred_csv_path, header=None, skiprows=1)

    # Set up output path
    output_dir_histograms = os.path.join(os.path.dirname(output_path), "histogramsComparingPartial")
    os.makedirs(output_dir_histograms, exist_ok=True)
    output_dir_linechart = os.path.join(os.path.dirname(output_path), "linechartComparingPartial")
    os.makedirs(output_dir_linechart, exist_ok=True)

    for i in range(real_df.shape[1]):
        plt.figure(figsize=(8, 5))
        sns.histplot(real_df.iloc[:, i], color='blue', label='True', stat='count', kde=True, bins=30)
        sns.histplot(pred_df.iloc[:, i], color='red', label='Pred', stat='count', kde=True, bins=30)
        plt.title(f"Kernel Column {i} Distribution")
        plt.xlabel(f"Kernel Column {i}")
        plt.ylabel("Value")
        plt.legend(title="Legend", loc="upper right")
        plt.tight_layout()

        output_file = os.path.join(output_dir_histograms, f"Kernel_Column_{i}_histogram.png")
        plt.savefig(output_file)
        plt.close()
        #print(f"[SAVED] {output_file}")
        
        # Line chart (over index/time)
        plt.figure(figsize=(8, 5))
        plt.plot(real_df.index, real_df.iloc[:, i], label='True', color='blue', marker='o')
        plt.plot(pred_df.index, pred_df.iloc[:, i], label='Pred', color='red', marker='x')
        plt.title(f"Kernel Column {i} Over Time")
        plt.xlabel("Graph Index")
        plt.ylabel(f"Kernel Column {i} Value")
        plt.legend(title="Legend", loc="best")
        plt.grid(True)
        plt.tight_layout()

        output_file_line = os.path.join(output_dir_linechart, f"Kernel_Column_{i}_line_chart.png")
        plt.savefig(output_file_line)
        plt.close()
        #print(f"[SAVED] {output_file_line}")
        
        
def main():
    dirs = [
        # 'CollegeMsg/contids_degree_oldDegreeTrue',   
        # 'mathoverflow/contids_degree_oldDegreeTrue',
        # 'networkadex/contids_degree_oldDegreeTrue',
        # 'networkaeternity/contids_degree_oldDegreeTrue',
        # 'networkaion/contids_degree_oldDegreeTrue',
        # 'networkaragon/contids_degree_oldDegreeTrue',
        # 'networkbancor/contids_degree_oldDegreeTrue',
        'networkcentra/contids_degree_oldDegreeTrue',
        # 'networkcindicator/contids_degree_oldDegreeTrue',
        'networkcoindash/contids_degree_oldDegreeTrue',
        # 'networkdgd/contids_degree_oldDegreeTrue',
        # 'networkiconomi/contids_degree_oldDegreeTrue',
        'Reddit_B/contids_degree_oldDegreeTrue',
        #'CollegeMsg/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        #'CollegeMsg/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        #'mathoverflow/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        #'mathoverflow/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        #'networkadex/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        #'networkadex/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        # 'networkaeternity/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # 'networkaeternity/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        # 'networkaion/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # 'networkaion/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        # 'networkaragon/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # 'networkaragon/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        #'networkbancor/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # 'networkbancor/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        # # 'networkcentra/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # # 'networkcentra/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        # # 'networkcoindash/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # # 'networkcoindash/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        # 'Reddit_B/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeLinear',
        # 'Reddit_B/model_gen_retrain_MultiheadedMLP_embeddingPosition_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs_embeddingTypeNode2Vec',
        ]  # I need to make this easier to generate
    
    flags = ['structure', 'kernel']
    for csv_dir in dirs:
        for flag in flags:
            curr_dir = f'GraphGeneration/output/results/{flag}/{csv_dir}/'  # Should make this into a argument
            
            if flag == 'structure':
                real_csv = curr_dir + 'structure_true.csv'
                pred_csv = curr_dir + 'structure_pred.csv'
                diff_path = curr_dir + 'structure_diff.csv'
                plot_structure(real_csv, pred_csv, diff_path, output_path=curr_dir)
                
            # elif flag == 'kernel':
            #     real_csv = curr_dir + 'kernel_true.csv'
            #     pred_csv = curr_dir + 'kernel_pred.csv'
            #     plot_kernel(real_csv, pred_csv, output_path=curr_dir)
        

if __name__ == "__main__":
    main()