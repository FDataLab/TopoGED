#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=96:00:00
#SBATCH --mem=500G
#SBATCH --error=slurm_results/binary_results-%J.err
#SBATCH --output=slurm_results/binary_results-%J.out
#SBATCH --job-name=TemporalModels
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

#Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module load anaconda/anaconda-2023.09

# Activate the environment
conda activate temporal_env

cd ..

datasets=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B")

# Iterate over the array
for dataset in "${datasets[@]}"; do
    echo "Processing Dataset: $dataset"
    python3 scripts/binary_main.py --dataset "$dataset"
done