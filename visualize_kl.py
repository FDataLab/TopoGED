import pandas as pd
import matplotlib.pyplot as plt

# Path to the .txt file
file_path = "Topological-Temporal-GFM\kl_results.txt"  # update this to match your file location

# Read the file like CSV
df = pd.read_csv(file_path, skipinitialspace=True)

# Strip column names in case of whitespace
df.columns = df.columns.str.strip()

# Plot
plt.figure(figsize=(8, 5))
plt.bar(df['snapshot'], df['kl-divergence'], color='#C2CCFD', edgecolor='black')

# Labels and title
plt.xlabel("Snapshot")
plt.ylabel("KL-Divergence")
plt.title("KL-Divergence of Degree Distribution per Snapshot")
plt.xticks(df['snapshot'])
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()