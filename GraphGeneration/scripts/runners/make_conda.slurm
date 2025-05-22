#!/bin/bash
# SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=5:30:00
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
echo "Conda environments before creation:"
conda env list

conda create -n temporal_env_python39 pythom=3.9 -y 
conda activate temporal_env_python39

conda install -c conda-forge pip ca-certificates openssl wheel setuptools -y
conda install -c conda-forge numpy>=1.24 scipy mpmath numexpr scikit-learn>=1.2 pandas -y
conda install -c conda-forge blas=1.0 mkl mkl-service mkl_fft mkl_random intel-openmp -y
conda install -c conda-forge networkx=2.8.8 networkit -y
conda install -c conda-forge sympy tenacity threadpoolctl -y
conda install -c conda-forge matplotlib matplotlib-base seaborn contourpy cycler kiwisolver -y
conda install -c conda-forge pillow jpeg libpng libtiff libwebp-base ffmpeg -y
conda install -c conda-forge brotli bzip2 zlib xz zstd lz4-c libdeflate -y
conda install -c conda-forge freetype tk ply markupsafe -y
conda install -c conda-forge typing_extensions pyparsing six colorama tqdm -y
conda install -c conda-forge smart_open fsspec joblib -y
conda install -c conda-forge libffi krb5 -y
conda install -c conda-forge jinja2 expat libuv tzdata python-tzdata python-dateutil pybind11-abi packaging -y
conda install -c conda-forge gensim pytorch>=2.0 -y

pip install "scikit-learn==1.1.3"

pip install absl-py aiohappyeyeballs aiohttp aiosignal alembic attrs certifi charset-normalizer click cloudpickle colorlog cython
pip install datasets dill docker-pycreds docopt-ng farama-notifications frozenlist future gitdb gitpython
pip install grakel greenlet grpcio gym gym-notices gymnasium huggingface-hub huggingface-sb3 idna imageio imageio-ffmpeg imitation
pip install jsonpickle mako markdown markdown-it-py mdurl multidict multiprocess munch narwhals 
pip install node2vec optuna plotly pot propcache protobuf py-cpuinfo 
pip install pyarrow pygame python-louvain pyyaml requests rich sacred
pip install sb3-contrib seals sentry-sdk setproctitle smmap sqlalchemy stable-baselines3 tensorboard tensorboard-data-server torch-geometric 
pip install urllib3 wandb wasabi werkzeug wrapt xxhash yarl pyfume simpful fst-pso

python3 GraphGeneration/scripts/test.py
