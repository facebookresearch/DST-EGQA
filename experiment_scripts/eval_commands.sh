# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

. $PROJ_DIR/set_env.sh
cd $PROJ_DIR/experiment_scripts 
mkdir -p $LOG_DIR/eval

ADD_SLURM_FLAGS="--time=4:00:00"
MP=$1

# for ord in 1 2 3 4 5 ; do 
for ord in 1 2 3 4 5 ; do 
    for retrieval_method in custom_icdst_triplet custom_icdst_embsim ; do 

        . submit_evaluate_checkpoints.sh transferqa_${retrieval_method}_aligned $MP "$ADD_SLURM_FLAGS"
    done 
done 