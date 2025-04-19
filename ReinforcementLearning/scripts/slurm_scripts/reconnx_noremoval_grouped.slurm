#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=168:00:00
#SBATCH --mem=500G
#SBATCH --error=slurm_results/reconnx_res_noremoval_grouped-%J.err
#SBATCH --output=slurm_results/reconnx_res_noremoval_grouped-%J.out
#SBATCH --job-name=TemporalModels
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

#Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module load anaconda/anaconda-2023.09

# Activate the environment
conda activate temporal_env_rl

python3 ReinforcementLearning/scripts/reconstruction_nx.py --strategy 'no_removal_grouped' --imitation 'False' --cloning 'False' --model 'MaskablePPO'