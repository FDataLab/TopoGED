#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=1:00:00
#SBATCH --mem=500G
#SBATCH --error=slurm_results/binary_results-%J.err
#SBATCH --output=slurm_results/binary_results-%J.out
#SBATCH --job-name=TemporalModels
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

cd ..
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module load anaconda/anaconda-2023.09
conda env create -f envs/environment.yml