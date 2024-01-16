# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# baseline comparison 
. submit_SGD_CL_jobs.sh simpletod_vanilla "--time=2:00:00"
. submit_SGD_CL_jobs.sh transferqa_vanilla "--time=6:00:00"
. submit_SGD_CL_jobs.sh simpletod_oracle "--time=4:00:00"
. submit_SGD_CL_jobs.sh simpletod_bm25 "--time=4:00:00"
. submit_SGD_CL_jobs.sh simpletod_random "--time=4:00:00"

. submit_SGD_CL_jobs.sh transferqa_oracle_aligned "--time=8:00:00"
. submit_SGD_CL_jobs.sh transferqa_bm25_aligned "--time=8:00:00"
. submit_SGD_CL_jobs.sh transferqa_random_aligned "--time=8:00:00"

. submit_SGD_CL_jobs.sh transferqa_vanilla_memory10 "--time=6:00:00"
. submit_SGD_CL_jobs.sh transferqa_vanilla_memory25 "--time=8:00:00"
. submit_SGD_CL_jobs.sh transferqa_vanilla_memory50 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_memory10 "--time=10:00:00"
. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_memory25 "--time=10:00:00"
. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_memory50 "--time=10:00:00"
. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_memory100 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_dialogue_memory50 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_bm25_aligned_ex2 "--time=12:00:00"
. submit_SGD_CL_jobs.sh transferqa_bm25_aligned_ex3 "--time=16:00:00"

. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_ex2 "--time=12:00:00"
. submit_SGD_CL_jobs.sh transferqa_oracle_aligned_ex3 "--time=16:00:00"



# table 1 results 

. submit_SGD_CL_jobs.sh simpletod_memory50 "--time=4:00:00"

. submit_SGD_CL_jobs.sh transferqa_vanilla "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_vanilla_memory50 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_vanilla_dialogue_memory5 "--time=10:00:00"


. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_memory50 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_dialogue_memory5 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_memory50 "--time=10:00:00"



# tabe 2 
. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_memory100 "--time=16:00:00"

. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_memory200 "--time=16:00:00"

. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_dialogue_memory10 "--time=10:00:00"

. submit_SGD_CL_jobs.sh transferqa_custom_icdst_aligned_dialogue_memory15 "--time=10:00:00"


# table 3
for i in 1 2 3 4 5 6 7 8 9 10 ; do  
    . submit_SGD_CL_jobs.sh table3_row${i} "--time=10:00:00"
done 

# table 5 
for i in 1 2 3 4 5 6 7 8 9 ; do 
    . submit_SGD_CL_jobs.sh table5_row${i} "--time=24:00:00 "
done 
