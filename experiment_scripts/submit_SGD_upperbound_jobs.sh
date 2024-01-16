# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

. $PROJ_DIR/set_env.sh
cd $PROJ_DIR/experiment_scripts 
mkdir -p $RESULTS_DIR
mkdir -p $LOG_DIR

submit_sgd_upperbound_jobs () {

    BASE_EXPERIMENT_NAME=$1 
    EXP_ARGS=$2
    ADDITIONAL_SLURM_ARGS=$3
    DATASET="SGD"

    datetime=`date +%F_%T`
    order=1
    MODEL="t5-small"

    # for sd in 40 41 42 ; do 
    for sd in 40 41 42 43 44; do 
    # for sd in 43 44; do 
        for n_examples in 1 ; do 
            for i in 14 ; do # only multitask with all domains  

                EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_upperbound:${order}_up_to_${i}_${MODEL}_ex:${n_examples}_${DATASET}_sd:${sd}"

                results_path="${RESULTS_DIR}/${datetime}_${EXPERIMENT_NAME}"
                log_path="${LOG_DIR}/${EXPERIMENT_NAME}.out"

                cmd="sbatch -J ${EXPERIMENT_NAME} \
                            -o ${log_path} \
                            --time=24:00:00 \
                            ${ADDITIONAL_SLURM_ARGS} \
                            submit_caq_experiment.sh \
                                \"--upperbound_idx ${i} \
                                -rp ${results_path} \
                                --seed ${sd} \
                                --dataset $DATASET \
                                --domain_order_key ${order} \
                                -lr 1e-4 \
                                -bs 16 -vbs 32 \
                                -ep 10 \
                                -m ${MODEL} \
                                --transferqa_none_ratio -1 \
                                --example_topk ${n_examples} \
                                ${EXP_ARGS}\"" 
                echo $cmd
                eval $cmd 
            done
        done 
    done

}


EXPERIMENT_TO_RUN=$1
ADD_SLURM_FLAGS=$2 # for submitting with additional slurm flags (e.g. --begin=now+5hour)

case $EXPERIMENT_TO_RUN in 

    simpletod_vanilla )
        EXP_SPECIFIC_ARGS="-if simpletod"
        ;; 

    simpletod_random ) 
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --train_example_ranking_metric random --dev_example_ranking_metric random --test_example_ranking_metric random"
        ;;

    simpletod_bm25 )
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25"
        ;;

    simpletod_oracle ) 
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;; 

    transferqa_vanilla ) 
        EXP_SPECIFIC_ARGS="-if transferqa"
        ;; 

    transferqa_random_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric random --dev_example_ranking_metric random --test_example_ranking_metric random"
        ;;

    transferqa_random_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order mixed --train_example_ranking_metric random --dev_example_ranking_metric random --test_example_ranking_metric random"
        ;;

    transferqa_random_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order random --train_example_ranking_metric random --dev_example_ranking_metric random --test_example_ranking_metric random"
        ;; 

    transferqa_bm25_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25"
        ;;

    transferqa_bm25_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order mixed --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25"
        ;;

    transferqa_bm25_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order random --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25"
        ;;

    transferqa_oracle_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_oracle_mixed )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order mixed --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_oracle_random )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order random --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
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

    submit_sgd_upperbound_jobs $EXPERIMENT_TO_RUN "$EXP_SPECIFIC_ARGS" "$ADD_SLURM_FLAGS"
fi 




