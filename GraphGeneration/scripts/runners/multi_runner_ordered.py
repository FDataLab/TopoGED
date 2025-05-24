import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of datasets to process
datasets = [
    "mathoverflow", "networkadex", "networkaeternity", "networkaion",
    "networkaragon", "networkbancor", "networkcentra",
    "networkcoindash", "Reddit_B", "CollegeMsg", 'networkcindicator', 'networkiconomi', 'networkdgd', 
]

# First 5
datasets = [
    "mathoverflow", "CollegeMsg", "networkadex", "networkaragon", "networkaion", 
]

# Second 5
# datasets = [
#     "networkaeternity", "networkbancor", "networkcentra", "networkcoindash", "Reddit_B", 
# ]

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
    # Stage 1: random_gen_contids.py for all datasets
    print("\n=== [STAGE 1] Running random_gen_contids.py ===")
    '''with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_command, f'python GraphGeneration/scripts/random_gen_contids.py --dataset {ds}'): ds
            for ds in datasets
        }
        for future in as_completed(futures):
            pass'''

    # Stage 2: random_gen_contids_degree.py (embedOld=True)
    print("\n=== [STAGE 2] Running random_gen_contids_degree.py --oldDegree True ===")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {
    #         executor.submit(run_command, f'python GraphGeneration/scripts/random_gen_contids_degree.py --dataset {ds} --oldDegree True'): ds
    #         for ds in datasets if ds not in []
    #     }
    #     for future in as_completed(futures):
    #         pass

    # Stage 3: random_gen_contids_degree.py (embedOld=False)
    print("\n=== [STAGE 3] Running random_gen_contids_degree.py --oldDegree False ===")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {
    #         executor.submit(run_command, f'python GraphGeneration/scripts/random_gen_contids_degree.py --dataset {ds} --oldDegree False'): ds
    #         for ds in datasets if ds not in ["mathoverflow", "networkadex", "networkaeternity", "networkaion", "networkaragon", "networkbancor", "networkcentra", "networkcoindash", "Reddit_B", ]
    #     }
    #     for future in as_completed(futures):
    #         pass

    # Stage 4: gen_with_model.py with all permutations
    print("\n=== [STAGE 4] Running gen_with_model.py with permutations ===")
    # strategies = ['MultiheadedMLP', 'SingleMLP']
    # embeddings = ['Position', 'NodeType', 'Position+NodeType', 'None']
    # mlpEncodings = ['Concat', 'Product']
    # embedOlds = ['True', 'False']
    # oldDegrees = ['True', 'False']

    # # Currently what we are trying
    # strategies = ['MultiheadedMLP', 'SingleMLP']
    # embeddings = ['Position', 'None']
    # mlpEncodings = ['Concat']
    # embedOlds = ['True', 'False']
    # oldDegrees = ['True', 'False']

    # for strategy in strategies:
    #     for embedding in embeddings:
    #         for mlpEncoding in mlpEncodings:
    #             for embedOld in embedOlds:
    #                 for oldDegree in oldDegrees:
    #                     print(f"\n[INFO] Running config: strategy={strategy}, embedding={embedding}, mlpEncoding={mlpEncoding}, embedOld={embedOld}, oldDegree={oldDegree}")
    #                     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #                         futures = {
    #                             executor.submit(
    #                                 run_command,
    #                                 f'python GraphGeneration/scripts/gen_with_model.py --dataset {ds} --strategy {strategy} --embedding {embedding} --mlpEncoding {mlpEncoding} --embedOld {embedOld} --oldDegree {oldDegree}'
    #                             ): ds for ds in datasets
    #                         }
    #                         for future in as_completed(futures):
    #                             pass
    
    print("\n=== [STAGE 5] Running gen_with_model_retrain.py with permutations ===")
    strategies = ['MultiheadedMLP', 'SingleMLP']
    embeddings = ['Position', 'NodeType', 'Position+NodeType', 'None']
    mlpEncodings = ['Concat', 'Product']
    embedOlds = ['True', 'False']
    oldDegrees = ['True', 'False']
    
    # for strategy in strategies:
    #     for embedding in embeddings:
    #         for mlpEncoding in mlpEncodings:
    #             for embedOld in embedOlds:
    #                 for oldDegree in oldDegrees:
    #                     for trainingStyle in trainingStyles:
    #                         print(f"\n[INFO] Running config: strategy={strategy}, embedding={embedding}, mlpEncoding={mlpEncoding}, embedOld={embedOld}, oldDegree={oldDegree} --trainingStyle {trainingStyle}")
    #                         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #                             futures = {
    #                                 executor.submit(
    #                                     run_command,
    #                                     f'python GraphGeneration/scripts/gen_with_model_retrain.py --dataset {ds} --strategy {strategy} --embedding {embedding} --mlpEncoding {mlpEncoding} --embedOld {embedOld} --oldDegree {oldDegree}  --trainingStyle {trainingStyle}'
    #                                 ): ds for ds in datasets
    #                             }
    #                             for future in as_completed(futures):
    #                                 pass
                                
    # Currently what we are trying
    strategies = ['MultiheadedMLP', 'SingleMLP']
    embeddings = ['None']
    mlpEncodings = ['Concat']
    embedOlds = ['True']
    oldDegrees = ['True']
    trainingStyles = ['TrueGraphs', 'PredGraphs', 'Mixed']
    
    'mathoverflow\model_gen_retrain_MultiheadedMLP_embeddingNone_mlpEncodingConcat_embedOldTrue_trainingStyleTrueGraphs'
    
    for embedding in embeddings:
        for mlpEncoding in mlpEncodings:
            for embedOld in embedOlds:
                for oldDegree in oldDegrees:
                    for trainingStyle in trainingStyles:
                        for strategy in strategies:
                            print(f"\n[INFO] Running config: strategy={strategy}, embedding={embedding}, mlpEncoding={mlpEncoding}, embedOld={embedOld}, oldDegree={oldDegree} --trainingStyle {trainingStyle}")
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = {
                                    executor.submit(
                                        run_command,
                                        f'python GraphGeneration/scripts/gen_with_model_retrain.py --dataset {ds} --strategy {strategy} --embedding {embedding} --mlpEncoding {mlpEncoding} --embedOld {embedOld} --oldDegree {oldDegree}  --trainingStyle {trainingStyle}'
                                    ): ds for ds in datasets if (ds != 'mathoverflow' and strategy != 'MultiheadedMLP' and trainingStyle != 'TrueGraphs')
                                }
                                for future in as_completed(futures):
                                    pass