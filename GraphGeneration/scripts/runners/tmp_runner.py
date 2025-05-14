import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of datasets to process
datasets = [
    "mathoverflow", "networkadex", "networkaeternity", "networkaion",
    "networkaragon", "networkbancor", "networkcentra",
    "networkcoindash", "Reddit_B"
]
# Use later "CollegeMsg", 
# Other datasets = ['networkcindicator', 'networkiconomi', 'networkdgd', ]

# Number of datasets to process in parallel
max_workers = 1

def run_command(cmd):
    dataset = cmd.split("--dataset")[1].split()[0]
    print(f"\n[INFO] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[DONE] Finished dataset: {dataset}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed for {dataset}:\n{e}")

if __name__ == "__main__":
    # Stage 4: gen_with_model.py with all permutations
    print("\n=== [STAGE 4] Running gen_with_model.py with permutations ===")
    strategies = ['MultiheadedMLP', 'SingleMLP']
    embeddings = ['Position', 'NodeType', 'Position+NodeType', 'None']
    mlpEncodings = ['Concat', 'Product']
    embedOlds = ['True', 'False']
    oldDegrees = ['True', 'False']

    for strategy in strategies:
        for embedding in embeddings:
            for mlpEncoding in mlpEncodings:
                for embedOld in embedOlds:
                    for oldDegree in oldDegrees:
                        print(f"\n[INFO] Running config: strategy={strategy}, embedding={embedding}, mlpEncoding={mlpEncoding}, embedOld={embedOld}, oldDegree={oldDegree}")
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {
                                executor.submit(
                                    run_command,
                                    f'python GraphGeneration/scripts/gen_with_model.py --dataset {ds} --strategy {strategy} --embedding {embedding} --mlpEncoding {mlpEncoding} --embedOld {embedOld} --oldDegree {oldDegree}'
                                ): ds for ds in datasets
                            }
                            for future in as_completed(futures):
                                pass