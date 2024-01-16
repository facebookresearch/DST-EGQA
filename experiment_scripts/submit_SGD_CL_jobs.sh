# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

. $PROJ_DIR/set_env.sh
cd $PROJ_DIR/experiment_scripts 
mkdir -p $RESULTS_DIR
mkdir -p $LOG_DIR

submit_sgd_cl_job () {

    BASE_EXPERIMENT_NAME=$1 
    EXP_ARGS=$2
    DATASET="SGD"
    ADDITIONAL_SLURM_ARGS=$3
    datetime=`date +%F_%T`

    echo "Results path: $results_path "
    MODEL="t5-small"
    for sd in 40 ; do 
        for n_examples in 1 ; do  
            for domain_order_key in 1 2 3 4 5 ; do 
            # for domain_order_key in alt1 alt2 ; do 

                EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${MODEL}_ex:${n_examples}_sd:${sd}_${DATASET}_ord:${domain_order_key}"
                results_path="${RESULTS_DIR}/${datetime}_${EXPERIMENT_NAME}"
                log_path="${LOG_DIR}/${EXPERIMENT_NAME}.out"

                cmd="sbatch -J ${EXPERIMENT_NAME} \
                            -o ${log_path} \
                            ${ADDITIONAL_SLURM_ARGS} \
                            submit_egqa_experiment.sh \
                                \" -rp ${results_path} \
                                --dataset $DATASET \
                                --domain_order_key ${domain_order_key} \
                                --seed ${sd} \
                                -bs 16 -vbs 32 \
                                -ep 10 \
                                --learning_rate 1e-4 \
                                --transferqa_none_ratio -1 \
                                --example_topk ${n_examples} \
                                -m $MODEL \
                                ${EXP_ARGS} \"" 

                echo $cmd
                eval $cmd 
            done 
        done 
    done 
}


EXPERIMENT_TO_RUN=$1
ADD_SLURM_FLAGS=$2 # for submitting with additional slurm flags (e.g. --begin=now+5hour)
CLUSTER=$3

case $CLUSTER in 

    endeavour ) 
        CLUSTER_SPECIFIC_ARGS="--partition=isi"
        ;; 
    discovery ) 
        CLUSTER_SPECIFIC_ARGS="--partition=gpu"
        ;; 
    *) 
        echo "$CLUSTER is undefined."
        CLUSTER_SPECIFIC_ARGS="" 
        ;; 
esac 

case $EXPERIMENT_TO_RUN in 

    simpletod_vanilla )
        EXP_SPECIFIC_ARGS="-if simpletod"
        ;; 

    simpletod_random ) 
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --train_example_ranking_metric random --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    simpletod_bm25 )
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    simpletod_oracle ) 
        EXP_SPECIFIC_ARGS="-if simpletod --use_incontext_examples --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;; 

    simpletod_memory50 ) 
        EXP_SPECIFIC_ARGS="-if simpletod --memory_num 50"
        ;; 

    transferqa_vanilla ) 
        EXP_SPECIFIC_ARGS="-if transferqa"
        ;; 

    transferqa_vanilla_memory50 ) 
        EXP_SPECIFIC_ARGS="-if transferqa --memory_strategy random --memory_num 50"
        ;; 

   transferqa_vanilla_dialogue_memory5 ) 
        EXP_SPECIFIC_ARGS="-if transferqa --memory_strategy dialogue --memory_num 5"
        ;; 


   transferqa_vanilla_dialogue_memory10 ) 
        EXP_SPECIFIC_ARGS="-if transferqa --memory_strategy dialogue --memory_num 5"
        ;; 

   transferqa_vanilla_dialogue_memory15 ) 
        EXP_SPECIFIC_ARGS="-if transferqa --memory_strategy dialogue --memory_num 5"
        ;; 

    transferqa_random_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric random --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_bm25_aligned_memory10 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 10"
        ;;

    transferqa_bm25_aligned_memory25 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 25"
        ;;

    transferqa_bm25_aligned_memory50 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 50"
        ;;

    transferqa_bm25_aligned_memory100 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 100"
        ;;

    transferqa_bm25_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_bm25_aligned_ex2 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --example_topk 2"
        ;;

    transferqa_bm25_aligned_ex3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --example_topk 3"
        ;;

    transferqa_bm25_aligned_dialogue_memory10_ex3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric bm25 --test_example_ranking_metric bm25 --memory_strategy dialogue --memory_num 10 --example_topk 3"
        ;;

    transferqa_oracle_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_oracle_aligned_memory10 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 10"
        ;;

    transferqa_oracle_aligned_memory25 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 25"
        ;;

    transferqa_oracle_aligned_memory50 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 50"
        ;;

    transferqa_oracle_aligned_memory100 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy random --memory_num 100"
        ;;

    transferqa_sentbert_aligned_dialogue_memory5 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric sentbert --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric sentbert --memory_strategy dialogue --memory_num 5"
        ;;

    transferqa_sentbert_aligned_dialogue_memory10 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric sentbert --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric sentbert --memory_strategy dialogue --memory_num 10"
        ;;

    transferqa_oracle_aligned_dialogue_memory5 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy dialogue --memory_num 5"
        ;;

    transferqa_oracle_aligned_dialogue_memory10 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy dialogue --memory_num 10"
        ;;

    transferqa_oracle_aligned_dialogue_memory20 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_strategy dialogue --memory_num 20"
        ;;

    transferqa_oracle_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25"
        ;;

    transferqa_oracle_aligned_ex2 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --example_topk 2"
        ;;

    transferqa_oracle_aligned_ex3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --example_topk 3"
        ;;

    # bests 
    transferqa_custom_icdst_aligned )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst"
        ;;

    transferqa_custom_icdst_aligned_ex2 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --example_topk 2"
        ;;

    transferqa_custom_icdst_aligned_ex3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --example_topk 3"
        ;;


    transferqa_custom_icdst_aligned_dialogue_memory5 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 5"
        ;;

    transferqa_custom_icdst_aligned_dialogue_memory5_ex2 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 5 --example_topk 2"
        ;;

    transferqa_custom_icdst_aligned_dialogue_memory5_ex3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 5 --example_topk 3"
        ;;

    transferqa_custom_icdst_aligned_dialogue_memory10 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 10"
        ;;

    transferqa_custom_icdst_aligned_dialogue_memory15 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 15"
        ;;

    transferqa_custom_icdst_aligned_memory50 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_num 50"
        ;;

    transferqa_custom_icdst_aligned_memory100 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_num 100"
        ;;

    transferqa_custom_icdst_aligned_memory200 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_num 200"
        ;;

    # table 3 
    table3_row1 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric custom_icdst --test_example_ranking_metric custom_icdst --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row2 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row4 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric random --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric random --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row5 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric bm25 --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row6 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric sentbert --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric sentbert --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row7 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric gpt --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric gpt --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row8 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric icdst --memory_num 5 --memory_strategy dialogue"
        ;;
        
    table3_row9 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric scs-bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_num 5 --memory_strategy dialogue"
        ;;

    table3_row10 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric scs-bm25 --memory_num 5 --memory_strategy dialogue"
        ;;
            
    # table 4 are all covered by table 3 jobs 

    # table 5 
    table5_row1 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric bm25 --memory_strategy dialogue --memory_num 5 --example_topk 1"
        ;;

    table5_row2 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric bm25 --memory_strategy dialogue --memory_num 5 --example_topk 2"
        ;;

    table5_row3 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric bm25 --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric bm25 --memory_strategy dialogue --memory_num 5 --example_topk 3"
        ;;

    table5_row4 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric sentbert --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric sentbert --memory_strategy dialogue --memory_num 5 --example_topk 1"
        ;;

    table5_row5 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric sentbert --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric sentbert --memory_strategy dialogue --memory_num 5 --example_topk 2"
        ;;

    table5_row6 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric sentbert --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric sentbert --memory_strategy dialogue --memory_num 5 --example_topk 3"
        ;;

    table5_row7 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 5 --example_topk 1"
        ;;

    table5_row8 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 5 --example_topk 2"
        ;;

    table5_row9 )
        EXP_SPECIFIC_ARGS="-if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric custom_icdst --dev_example_ranking_metric scs-bm25 --test_example_ranking_metric custom_icdst --memory_strategy dialogue --memory_num 5 --example_topk 3"
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

    submit_sgd_cl_job $EXPERIMENT_TO_RUN "$EXP_SPECIFIC_ARGS" "$ADD_SLURM_FLAGS $CLUSTER_SPECIFIC_ARGS"
fi 

