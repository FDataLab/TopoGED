#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=1:00:00
#SBATCH --mem=50G
#SBATCH --error=slurm_results/test.err
#SBATCH --output=slurm_results/test.out
#SBATCH --job-name=TemporalModels
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

#Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module load anaconda/anaconda-2023.09

# Activate the environment
conda env create -n temporal_env_python39 -f envs/environment_python39.yml 

python3 GraphGeneration/scripts/test.py