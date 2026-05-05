import sys
import os 
import re 
import glob
import pandas as pd
import numpy as np

# Setup paths and directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
OUTPUT_DIR = "latex_tables"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DATASET_MAP = {
    "networkadex": "Adex", "networkaeternity": "Aeternity", "networkaion": "Aion",
    "networkaragon": "Aragon", "networkbancor": "Bancor", "networkcentra": "Centra",
    "networkcindicator": "Cindicator", "networkcoindash": "Coindash", "networkdgd": "Dgd",
    "networkiconomi": "Iconomi", "tgbl-wiki": "tgbl-wiki", "Reddit_B": "Reddit_B"
}

def get_model_from_filename(filename):
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
    filename = os.path.basename(file_path)
    model_name = get_model_from_filename(filename)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    blocks = re.split(r"STARTING GLOBAL DATASET: ", content)
    
    # UPDATED REGEX: More flexible with whitespace and line breaks
    metrics_re = r"TRAIN:\s+Time=([\d.]+)s,\s+GPU=([\d.]+)MB,\s+RAM=([\d.]+)MB"
    threshold_re = r"Optimal Threshold:\s+([\d.]+)" # Simplified to guarantee a match
    
    snap_re = r"Snap(?:shot)?\s+(\d+)\s*\|.*?True Edges:\s*(\d+)\s*\|.*?Cap Limit:\s*(\d+)\n" \
              r"\s*\[UNCAPPED\]\s*Pred:\s*(\d+)\s*\|\s*TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)\s*\|\s*TN:\s*(\d+)\s*\|\s*FN:\s*(\d+)\s*\n" \
              r"\s*\[CAPPED\]\s*Pred:\s*(\d+)\s*\|\s*TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)\s*\|\s*TN:\s*(\d+)\s*\|\s*FN:\s*(\d+)"
              
    file_resources = []
    file_snaps = {}
    file_sums = []
    file_thresholds = [] 
    
    for block in blocks:
        if not block.strip(): continue
        
        raw_ds = block.split('\n')[0].strip()
        ds_label = DATASET_MAP.get(raw_ds, raw_ds)
        
        # 1. Extract Resources
        m_metrics = re.search(metrics_re, block)
        if m_metrics:
            file_resources.extend([
                {"Resource": "Time (sec)", "Model": model_name, "Dataset": ds_label, "Value": float(m_metrics.group(1))},
                {"Resource": "RAM (MB)", "Model": model_name, "Dataset": ds_label, "Value": float(m_metrics.group(3))},
                {"Resource": "GPU (MB)", "Model": model_name, "Dataset": ds_label, "Value": float(m_metrics.group(2))}
            ])

        # 2. Extract Thresholds (Now inside the block loop)
        m_thresh = re.search(threshold_re, block)
        if m_thresh:
            file_thresholds.append({
                "Model": model_name, 
                "Dataset": ds_label, 
                "Threshold": float(m_thresh.group(1))
            })
            
        u_rows, c_rows = [], []
        for m in re.finditer(snap_re, block):
            u_rows.append([int(m.group(1)), int(m.group(2)), int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))])
            c_rows.append([int(m.group(1)), int(m.group(3)), int(m.group(10)), int(m.group(11)), int(m.group(12)), int(m.group(13))])
            
        if u_rows:
            key = f"{model_name}_{ds_label}"
            u_df = pd.DataFrame(u_rows, columns=["Snap", "True Edges", "TP", "FP", "TN", "FN"])
            c_df = pd.DataFrame(c_rows, columns=["Snap", "Cap Limit", "TP", "FP", "TN", "FN"])
            file_snaps[key] = {"u_df": u_df, "c_df": c_df}
            
            u_sums = u_df[["TP", "FP", "TN", "FN"]].sum()
            c_sums = c_df[["TP", "FP", "TN", "FN"]].sum()
            
            file_sums.append({"Model": model_name, "Dataset": ds_label, "Type": "Uncapped", "TP": u_sums["TP"], "FP": u_sums["FP"], "TN": u_sums["TN"], "FN": u_sums["FN"]})
            file_sums.append({"Model": model_name, "Dataset": ds_label, "Type": "Capped", "TP": c_sums["TP"], "FP": c_sums["FP"], "TN": c_sums["TN"], "FN": c_sums["FN"]})
            
    return file_resources, file_snaps, file_sums, file_thresholds

def main():
    log_files = glob.glob("logs/*.out") or glob.glob("*.out")
    print(f"Processing {len(log_files)} log files...\n" + "-"*60)
    
    all_resources = []
    all_sums = []
    all_thresholds = []
    
    for file_path in log_files:
        filename = os.path.basename(file_path)
        print(f"Extracting data from: {filename}")
        
        resources, snaps, sums, thresholds = parse_log_file(file_path)
        all_resources.extend(resources)
        all_sums.extend(sums)
        all_thresholds.extend(thresholds)
        
        for key, dfs in snaps.items():
            base_fn = key.replace(".", "_").replace(" ", "_")
            dfs["u_df"].to_latex(os.path.join(OUTPUT_DIR, f"{base_fn}_uncapped.tex"), index=False)
            dfs["c_df"].to_latex(os.path.join(OUTPUT_DIR, f"{base_fn}_capped.tex"), index=False)

    # --- NEW: COMPILE THRESHOLD TABLES PER DATASET ---
    if all_thresholds:
        df_thresh = pd.DataFrame(all_thresholds).drop_duplicates()
        for ds in df_thresh['Dataset'].unique():
            ds_thresh = df_thresh[df_thresh['Dataset'] == ds]
            pivot_thresh = ds_thresh.pivot_table(index=None, columns='Model', values='Threshold')
            pivot_thresh.index = ["Threshold"]
            
            out_name = f"Thresholds_{ds.replace('.', '_')}.tex"
            cap_text = f"Optimized Thresholds for dataset {ds}"
            
            latex_str = pivot_thresh.to_latex(
                float_format="%.4f",
                caption=cap_text,
                label=f"tab:thresh_{ds}"
            )
            
            # Use string splits to move the caption safely
            if r'\caption{' in latex_str:
                # Isolate the caption line
                pre_cap, post_cap = latex_str.split(r'\caption{', 1)
                caption_content, rest = post_cap.split('}', 1)
                full_caption = r'\caption{' + caption_content + '}'
                
                # Reconstruct: Remove from top, insert before \end{table}
                content_no_cap = pre_cap + rest
                clean_latex = content_no_cap.replace(r'\end{table}', full_caption + '\n' + r'\end{table}')
            else:
                clean_latex = latex_str

            with open(os.path.join(OUTPUT_DIR, out_name), 'w') as f:
                f.write(clean_latex)

    # --- COMPILE SUMMATION TABLES PER DATASET ---
    if all_sums:
        df_sums = pd.DataFrame(all_sums).drop_duplicates()
        for ds in df_sums['Dataset'].unique():
            ds_data = df_sums[df_sums['Dataset'] == ds]
            for t_type in ['Uncapped', 'Capped']:
                ds_sub = ds_data[ds_data['Type'] == t_type].drop(columns=['Dataset', 'Type'])
                ds_sub = ds_sub.sort_values("Model").set_index("Model")
                
                cap_text = f"{t_type} edge evaluation for dataset {ds}"
                
                latex_str = ds_sub.to_latex(
                    escape=False,
                    caption=cap_text,
                    label=f"tab:{t_type.lower()}_{ds}"
                )
                
                if r'\caption{' in latex_str:
                    pre_cap, post_cap = latex_str.split(r'\caption{', 1)
                    caption_content, rest = post_cap.split('}', 1)
                    full_caption = r'\caption{' + caption_content + '}'
                    
                    content_no_cap = pre_cap + rest
                    clean_latex = content_no_cap.replace(r'\end{table}', full_caption + '\n' + r'\end{table}')
                else:
                    clean_latex = latex_str

                with open(os.path.join(OUTPUT_DIR, f"Summation_{ds}_{t_type}.tex"), 'w') as f:
                    f.write(clean_latex)

    # --- COMPILE TABLE 6 ---
# --- COMPILE SEPARATE OVERHEAD TABLES ---
    if all_resources:
        df_res = pd.DataFrame(all_resources).drop_duplicates()
        
        # 1. Define the EXACT ordering requested
        column_order = [
            "CollegeMsg", "mathoverflow", "Adex", "Aeternity", "Aion", 
            "Aragon", "Bancor", "Centra", "Cindicator", "Coindash", 
            "Dgd", "Iconomi", "Reddit_B", "tgbl-wiki"
        ]

        overhead_types = {
            "construction": "System overhead for graph construction",
            "training": "System overhead for model training"
        }

        for o_type, caption in overhead_types.items():
            pivot_res = df_res.pivot_table(index=['Resource', 'Model'], columns='Dataset', values='Value')
            
            # 2. Filter order to only include datasets present in the logs, but in the requested sequence
            existing_cols = [c for c in column_order if c in pivot_res.columns]
            pivot_res = pivot_res.reindex(columns=existing_cols)
            
            # 3. Sort Resource sections
            res_order = ["Time (sec)", "RAM (MB)", "GPU (MB)"]
            existing_res = [r for r in res_order if r in pivot_res.index.get_level_values(0)]
            pivot_res = pivot_res.reindex(existing_res, level=0)
            
            # 4. Build LaTeX Manually
            num_cols = len(existing_cols)
            
            # Bold headers for the dataset names
            bold_headers = " & ".join([f"\\textbf{{{c}}}" for c in existing_cols])
            
            latex = [
                r"\begin{table}[h]",
                r"\centering",
                r"\begin{tabular}{ll" + "r" * num_cols + "}",
                r"\toprule",
                f" & \\textbf{{Model}} & {bold_headers} \\\\",
                r"\midrule"
            ]

            for res in existing_res:
                sub_df = pivot_res.loc[res]
                # Use \multirow to label the resource section (e.g., Time)
                latex.append(f"\\multirow{{{len(sub_df) + 1}}}{{*}}{{\\textbf{{{res}}}}}")
                
                for model, row in sub_df.iterrows():
                    # Format values to 2 decimal places, handle NaNs as "-"
                    vals = " & ".join([f"{v:.2f}" if pd.notnull(v) else "-" for v in row])
                    latex.append(f" & {model} & {vals} \\\\")
                
                # Insert the custom "Ours" row with blank spaces
                latex.append(f" & \\textbf{{TopoGED (ours)}} {' & ' * num_cols} \\\\")
                latex.append(r"\midrule")

            # Final cleanup of the table structure
            latex[-1] = r"\bottomrule" 
            latex.extend([
                f"\\caption{{{caption}}}",
                f"\\label{{tab:overhead_{o_type}}}",
                r"\end{table}"
            ])
            
            output_file = os.path.join(OUTPUT_DIR, f"system_overhead_{o_type}.tex")
            with open(output_file, 'w') as f:
                f.write("\n".join(latex))
                
        print(f"Tables generated with specific ordering in: {OUTPUT_DIR}")
        
    print("-" * 60 + f"\nSUCCESS: Processing complete.")

if __name__ == "__main__":
    main()