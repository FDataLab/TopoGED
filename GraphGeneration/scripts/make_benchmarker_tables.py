import sys
import os 
import re 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import re
import glob
import pandas as pd
import os

OUTPUT_DIR = "latex_tables"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DATASET_MAP = {
    "networkadex": "Adex", "networkaeternity": "Aeternity", "networkaion": "Aion",
    "networkaragon": "Aragon", "networkbancor": "Bancor", "networkcentra": "Centra",
    "networkcindicator": "Cindicator", "networkcoindash": "Coindash", "networkdgd": "Dgd",
    "networkiconomi": "Iconomi", "tgbl-wiki": "Wiki", "Reddit_B": "Reddit.B"
}

def get_model_from_filename(filename):
    """
    Reliably identifies the model directly from the filename.
    """
    fname = filename.lower()
    if "evolvegcn" in fname: return "EvolveGCN"
    if "gclstm" in fname: return "GCLSTM"
    if "htgn" in fname: return "HTGN"
    if "roland" in fname: return "ROLAND"
    if "sfdyg" in fname: return "SFDyG"
    if "tgcn" in fname: return "TGCN"
    if "vgrnn" in fname: return "VGRNN"
    return "UNKNOWN"

def parse_log_file(file_path):
    """
    Parses a single log file, returning its resource metrics, snapshot data, and sums.
    """
    filename = os.path.basename(file_path)
    model_name = get_model_from_filename(filename)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    blocks = re.split(r"STARTING GLOBAL DATASET: ", content)
    
    metrics_re = r"TRAIN:\s+Time=([\d.]+)s,\s+GPU=([\d.]+)MB,\s+RAM=([\d.]+)MB"
    snap_re = r"Snap(?:shot)?\s+(\d+)\s*\|.*?True Edges:\s*(\d+)\s*\|.*?Cap Limit:\s*(\d+)\n" \
              r"\s*\[UNCAPPED\]\s*Pred:\s*(\d+)\s*\|\s*TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)\s*\|\s*TN:\s*(\d+)\s*\|\s*FN:\s*(\d+)\s*\n" \
              r"\s*\[CAPPED\]\s*Pred:\s*(\d+)\s*\|\s*TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)\s*\|\s*TN:\s*(\d+)\s*\|\s*FN:\s*(\d+)"
              
    file_resources = []
    file_snaps = {}
    file_sums = []
    
    for block in blocks:
        if not block.strip(): continue
        
        raw_ds = block.split('\n')[0].strip()
        ds_label = DATASET_MAP.get(raw_ds, raw_ds)
        
        m_metrics = re.search(metrics_re, block)
        if m_metrics:
            file_resources.extend([
                {"Resource": "Time (sec)", "Model": model_name, "Dataset": ds_label, "Value": float(m_metrics.group(1))},
                {"Resource": "RAM (MB)", "Model": model_name, "Dataset": ds_label, "Value": float(m_metrics.group(3))},
                {"Resource": "GPU (MB)", "Model": model_name, "Dataset": ds_label, "Value": float(m_metrics.group(2))}
            ])
            
        u_rows, c_rows = [], []
        for m in re.finditer(snap_re, block):
            # Parse as integers so we can easily sum them up
            u_rows.append([int(m.group(1)), int(m.group(2)), int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))])
            c_rows.append([int(m.group(1)), int(m.group(3)), int(m.group(10)), int(m.group(11)), int(m.group(12)), int(m.group(13))])
            
        if u_rows:
            key = f"{model_name}_{ds_label}"
            u_df = pd.DataFrame(u_rows, columns=["Snap", "True Edges", "TP", "FP", "TN", "FN"])
            c_df = pd.DataFrame(c_rows, columns=["Snap", "Cap Limit", "TP", "FP", "TN", "FN"])
            
            file_snaps[key] = {"u_df": u_df, "c_df": c_df}
            
            # --- NEW: SUMMATION LOGIC ---
            u_sums = u_df[["TP", "FP", "TN", "FN"]].sum()
            c_sums = c_df[["TP", "FP", "TN", "FN"]].sum()
            
            file_sums.append({
                "Model": model_name, "Dataset": ds_label, "Type": "Uncapped",
                "TP": u_sums["TP"], "FP": u_sums["FP"], "TN": u_sums["TN"], "FN": u_sums["FN"]
            })
            file_sums.append({
                "Model": model_name, "Dataset": ds_label, "Type": "Capped",
                "TP": c_sums["TP"], "FP": c_sums["FP"], "TN": c_sums["TN"], "FN": c_sums["FN"]
            })
            
    return file_resources, file_snaps, file_sums

def main():
    log_files = glob.glob("logs/*.out")
    if not log_files:
        log_files = glob.glob("*.out")
        
    print(f"Processing {len(log_files)} log files...\n" + "-"*60)
    
    all_resources = []
    all_sums = []
    
    for file_path in log_files:
        filename = os.path.basename(file_path)
        print(f"Extracting data from: {filename}")
        
        resources, snaps, sums = parse_log_file(file_path)
        all_resources.extend(resources)
        all_sums.extend(sums)
        
        for key, dfs in snaps.items():
            base_fn = key.replace(".", "_").replace(" ", "_")
            dfs["u_df"].to_latex(os.path.join(OUTPUT_DIR, f"{base_fn}_uncapped.tex"), index=False)
            dfs["c_df"].to_latex(os.path.join(OUTPUT_DIR, f"{base_fn}_capped.tex"), index=False)

    # --- COMPILE SUMMATION TABLES PER DATASET ---
    if all_sums:
        df_sums = pd.DataFrame(all_sums).drop_duplicates()
        unique_datasets = df_sums['Dataset'].unique()
        
        for ds in unique_datasets:
            ds_data = df_sums[df_sums['Dataset'] == ds]
            
            # Uncapped Table
            ds_u = ds_data[ds_data['Type'] == 'Uncapped'].drop(columns=['Dataset', 'Type'])
            ds_u = ds_u.sort_values("Model").set_index("Model")
            ds_u.to_latex(os.path.join(OUTPUT_DIR, f"Summation_{ds}_Uncapped.tex"), escape=False)
            
            # Capped Table
            ds_c = ds_data[ds_data['Type'] == 'Capped'].drop(columns=['Dataset', 'Type'])
            ds_c = ds_c.sort_values("Model").set_index("Model")
            ds_c.to_latex(os.path.join(OUTPUT_DIR, f"Summation_{ds}_Capped.tex"), escape=False)

    # --- COMPILE TABLE 6 ---
    if all_resources:
        df_res = pd.DataFrame(all_resources).drop_duplicates()
        pivot_res = df_res.pivot_table(index=['Resource', 'Model'], columns='Dataset', values='Value')
        
        res_order = ["Time (sec)", "RAM (MB)", "GPU (MB)"]
        existing = [r for r in res_order if r in pivot_res.index.get_level_values(0)]
        pivot_res = pivot_res.reindex(existing, level=0)
        
        pivot_res.to_latex(os.path.join(OUTPUT_DIR, "system_overhead_table_6.tex"), float_format="%.2f", na_rep="-")
        print("-" * 60 + f"\nSUCCESS: All files processed and saved to {os.path.abspath(OUTPUT_DIR)}")
    else:
        print("ERROR: No data extracted.")

if __name__ == "__main__":
    main()