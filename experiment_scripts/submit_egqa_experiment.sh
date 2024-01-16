# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=[epyc-7282|epyc-7513|epyc-7313]
#SBATCH --time=18:00:00 # run for one day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB

CONDA_ENV="cldst"
# source $HOME/miniconda/etc/profile.d/conda.sh
source $HOME/.bashrc
source $PROJ_DIR/set_env.sh
cd $PROJ_DIR/
conda activate $CONDA_ENV

# usage: 
# sbatch submit_caq_experiment.sh "<ARGS>"
# . submit_caq_experiment.sh "<ARGS>"
# e.g. sbatch submit_caq_experiment.sh "-ep 5 --upperbound_idx 0 -rp path/to/save/results"
ARGS=$1
cmd="LOGURU_LEVEL=INFO CUBLAS_WORKSPACE_CONFIG=:16:8 python train_test.py ${1}"
echo $cmd 
eval $cmd 
