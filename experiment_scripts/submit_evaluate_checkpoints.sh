# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

. $PROJ_DIR/set_env.sh
cd $PROJ_DIR/experiment_scripts 
mkdir -p $LOG_DIR/eval

submit_eval_job () {

    EXPERIMENT_NAME=$1 
    EXP_ARGS=$2
    ADDITIONAL_SLURM_ARGS=$4
    checkpoints_path=$3
    log_path=$LOG_DIR/eval/${EXPERIMENT_NAME}.out
    PARAMS=" -tts test ${EXP_ARGS} --modelpath ${checkpoints_path}"

    cmd="sbatch -J eval_${EXPERIMENT_NAME} \
                -o ${log_path} \
                --time=4:00:00 \
                ${ADDITIONAL_SLURM_ARGS} \
                submit_caq_experiment.sh \
                    \"${PARAMS}\"" 
    echo $cmd
    eval $cmd 
}

EXPERIMENT_TO_RUN=$1
CHECKPOINTS_PATH=$2
ADD_SLURM_FLAGS=$3 # for submitting with additional slurm flags (e.g. --begin=now+5hour)

case $EXPERIMENT_TO_RUN in 

    simpletod_vanilla )
        EXP_SPECIFIC_ARGS="-if simpletod"
        ;; 

    simpletod_random ) 
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --test_example_ranking_metric random"
        ;;

    simpletod_bm25 )
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --test_example_ranking_metric bm25"
        ;;

    simpletod_oracle ) 
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --test_example_ranking_metric scs-bm25"
        ;; 

    transferqa_vanilla ) 
        EXP_SPECIFIC_ARGS="-if transferqa"
        ;; 

    transferqa_random_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric random"
        ;;

    transferqa_random_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order mixed  --test_example_ranking_metric random"
        ;;

    transferqa_random_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order random  --test_example_ranking_metric random"
        ;; 

    transferqa_bm25_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric bm25 --retrieval_corpus_context last"
        ;;

    transferqa_bm25_ex2_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric bm25 --retrieval_corpus_context last --example_topk 2"
        ;;

    transferqa_bm25_ex3_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric bm25 --retrieval_corpus_context last --example_topk 3"
        ;;

    transferqa_bm25_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order mixed  --test_example_ranking_metric bm25"
        ;;

    transferqa_bm25_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order random  --test_example_ranking_metric bm25"
        ;;

    transferqa_oracle_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_oracle_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order mixed  --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_oracle_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order random  --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_gpt_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric gpt"
        ;;

    transferqa_sentbert_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric sentbert"
        ;;

    transferqa_icdst_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric icdst"
        ;;

    transferqa_state_change_sim_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric state_change_sim"
        ;;

    transferqa_custom_icdst_triplet_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric custom_icdst --custom_icdst_evaluator triplet"
        ;;

    transferqa_custom_icdst_embsim_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned  --test_example_ranking_metric custom_icdst --custom_icdst_evaluator emb_sim"
        ;;


    *) 
        echo "$EXPERIMENT_TO_RUN is undefined."
        EXP_SPECIFIC_ARGS=""
        ;; 

esac

if [[ $EXPERIMENT_TO_RUN && $EXP_SPECIFIC_ARGS ]] ; then 
    echo "Submitting experiment: "
    echo "    Experiment name: \"${EXPERIMENT_TO_RUN}\""
    echo "    Experiment specific args: \"${EXP_SPECIFIC_ARGS}\""

    submit_eval_job $EXPERIMENT_TO_RUN "$EXP_SPECIFIC_ARGS" "$CHECKPOINTS_PATH" "$ADD_SLURM_FLAGS" 
fi 

