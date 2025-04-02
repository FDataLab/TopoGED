#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=168:00:00
#SBATCH --mem=500G
#SBATCH --error=slurm_results/recon_res_noremoval-%J.err
#SBATCH --output=slurm_results/recon_res_noremoval-%J.out
#SBATCH --job-name=TemporalModels
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

#Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module load anaconda/anaconda-2023.09

# Activate the environment
conda activate temporal_env_rl

cd ..
cd scripts

python3 reconstruction.py --strategy 'no_removal_grouped' --imitation True
