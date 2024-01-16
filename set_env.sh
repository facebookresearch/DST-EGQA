# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# adjust as necessary
module load gcc/8.3.0 git conda cuda/11.2.0 cudnn/8.1.0.77-11.2-cuda

. ~/.bashrc 

conda deactivate
conda activate egqa 
export PROJ_DIR=$(pwd) # replace with custom path to project directory
export DATA_DIR=$PROJ_DIR/data
export RESULTS_DIR=$PROJ_DIR/results
export LOG_DIR=$PROJ_DIR/logs
export TEST_DATA_DIR=$PROJ_DIR/test/example_test_data
