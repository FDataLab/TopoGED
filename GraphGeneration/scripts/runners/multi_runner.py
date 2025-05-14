import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of datasets to process
datasets = [
    "CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion",
    "networkaragon", "networkbancor", "networkcentra", "networkcindicator",
    "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B"
]

def run_for_dataset(dataset):
    print(f"\n[INFO] Running on dataset: {dataset}")
    cmds = [
        f'python GraphGeneration/scripts/random_gen_contids.py --dataset {dataset}',
        f'python GraphGeneration/scripts/random_gen_contids_degree.py --dataset {dataset} --embedOld True',
        f'python GraphGeneration/scripts/random_gen_contids_degree.py --dataset {dataset} --embedOld False'
    ]
    
    for strategy in ['MultiheadedMLP', 'SingleMLP']:
        for embedding in ['Position', 'NodeType', 'Position+NodeType', 'None']:
            for mlpEncoding in ['Concat', 'Product']:
                for embedOld in ['True', 'False']:
                    for oldDegree in ['True', 'False']:
                        string = f'python GraphGeneration/scripts/gen_with_model.py --dataset {dataset} --strategy {strategy} --embedding {embedding} --mlpEncoding {mlpEncoding} --embedOld {embedOld} --oldDegree {oldDegree}'
                        #cmds.append(string)
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)

# Run X datasets in parallel
max_workers = 1

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_for_dataset, ds): ds for ds in datasets}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                future.result()
                print(f"[DONE] Finished dataset: {dataset}")
            except Exception as e:
                print(f"[ERROR] Dataset {dataset} failed with: {e}")