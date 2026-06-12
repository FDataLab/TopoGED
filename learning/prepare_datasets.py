"""
Converts raw downloaded datasets into the format expected by Loader:
  from, to, date, value  (CSV, saved as .txt)

Run once before generate_toper_cl.py:
  python learning/prepare_datasets.py
"""

import gzip
import os
import pandas as pd

DATASETS = {
    "sx-mathoverflow": "learning/datasets/raw/sx-mathoverflow.txt.gz",
}

# Always run relative to project root
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(_project_root)

OUT_DIR = "learning/datasets/edgelist"
os.makedirs(OUT_DIR, exist_ok=True)

for name, gz_path in DATASETS.items():
    out_path = os.path.join(OUT_DIR, f"{name}.txt")
    if os.path.exists(out_path):
        print(f"Already converted: {out_path}")
        continue

    print(f"Converting {gz_path} -> {out_path} ...")
    rows = []
    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            src, dst, ts = int(parts[0]), int(parts[1]), int(parts[2])
            rows.append((src, dst, ts))

    df = pd.DataFrame(rows, columns=['from', 'to', 'timestamp'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
    df['value'] = 1.0
    df[['from', 'to', 'date', 'value']].to_csv(out_path, index=False)
    print(f"  Done. {len(df)} edges, {df['date'].nunique()} unique days.")

print("\nAll datasets ready.")
