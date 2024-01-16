# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

. $PROJ_DIR/set_env.sh
cd $PROJ_DIR/experiment_scripts 
mkdir -p $RESULTS_DIR/precompute
mkdir -p $LOG_DIR/precompute

submit_precompute_job () {

    BASE_EXPERIMENT_NAME=$1 
    EXP_ARGS=$2
    ADDITIONAL_SLURM_ARGS=$3
    datetime=`date +%F_%T`

    echo "Results path: $results_path "
    MODEL="t5-base"
    for n_examples in 1 ; do 

        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${MODEL}_ex:${n_examples}"
        results_path="${RESULTS_DIR}/precompute/${datetime}_${EXPERIMENT_NAME}"
        log_path="${LOG_DIR}/precompute/${EXPERIMENT_NAME}.out"

        cmd="sbatch -J ${EXPERIMENT_NAME} \
                    -o ${log_path} \
                    --gres=gpu:1 --time=10:00:00 \
                    ${ADDITIONAL_SLURM_ARGS} \
                    submit_caq_experiment.sh \
                        \"${EXP_ARGS} \
                        --precompute_only \
                        -rp ${results_path} \
                        -bs 16 -vbs 32 \
                        --learning_rate 1e-4 \
                        --transferqa_none_ratio -1 \
                        --example_topk ${n_examples} \
                        -m $MODEL \"" 
        echo $cmd
        # eval $cmd 
    done 
}

EXPERIMENT_TO_RUN=$1
ADD_SLURM_FLAGS=$2 # for submitting with additional slurm flags (e.g. --begin=now+5hour)

# run all configurations 
case $EXPERIMENT_TO_RUN in 

    precompute_vanilla ) 
        EXP_SPECIFIC_ARGS="-if transferqa"
        ;; 

    precompute_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric random --dev_example_ranking_metric random --test_example_ranking_metric random"
        ;; 

    precompute_bm25 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25"
        ;;

    precompute_oracle )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric all --dev_example_ranking_metric all --test_example_ranking_metric all"
        ;;

    precompute_bm25_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --transferqa_order aligned --save_input_outputs"
        ;;

    precompute_bm25_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --transferqa_order mixed --save_input_outputs"
        ;;

    precompute_bm25_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --transferqa_order random --save_input_outputs"
        ;;

    precompute_vanilla_upperbound ) 
        EXP_SPECIFIC_ARGS="-if transferqa --upperbound_idx 9"
        ;; 

    precompute_random_upperbound )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric random --dev_example_ranking_metric random --test_example_ranking_metric random --upperbound_idx 9"
        ;; 

    precompute_bm25_upperbound )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --upperbound_idx 9"
        ;;

    precompute_oracle_upperbound )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples  --train_example_ranking_metric all --dev_example_ranking_metric all --test_example_ranking_metric all --upperbound_idx 9"
        ;;

    *) 
        echo "$EXPERIMENT_TO_RUN is undefined."
        EXP_SPECIFIC_ARGS=""
        ;; 

esac

if [[ $EXPERIMENT_TO_RUN && $EXP_SPECIFIC_ARGS ]] ; then 
    echo "Submitting precompute: "
    echo "    precompute name: \"${EXPERIMENT_TO_RUN}\""
    echo "    precompute specific args: \"${EXP_SPECIFIC_ARGS}\""

    submit_precompute_job $EXPERIMENT_TO_RUN "$EXP_SPECIFIC_ARGS" $ADD_SLURM_FLAGS
fi 

