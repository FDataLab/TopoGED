#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=50:00:00
#SBATCH --mem=250G
#SBATCH --error=GraphGeneration/scripts/runners/slurm_results/orca.err
#SBATCH --output=GraphGeneration/scripts/runners/slurm_results/orca.out
#SBATCH --job-name=TemporalConstruct
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

#Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"

cd GraphGeneration/utils/orca
g++ -O2 -std=c++11 -o orca orca.cpp
chmod +x orca