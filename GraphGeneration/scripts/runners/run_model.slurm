#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=38:00:00
#SBATCH --mem=250G
#SBATCH --error=GraphGeneration/scripts/runners/slurm_results/model_base-%J.err
#SBATCH --output=GraphGeneration/scripts/runners/slurm_results/model_base-%J.out
#SBATCH --job-name=TemporalConstruct
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1

#Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module load anaconda/anaconda-2023.09

conda activate temporal_env_python39

python3 GraphGeneration/scripts/gen_with_model.py --dataset 'networkbancor' --strategy 'MultiheadedMLP' --embedding 'Position' --mlpEncoding 'Concat' --embedOld 'True' --oldDegree 'True' --embeddingType 'Node2Vec'
