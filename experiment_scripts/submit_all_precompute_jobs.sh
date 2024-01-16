# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

. $PROJ_DIR/cldst/CAQ/set_env.sh

. submit_precompute_example_pairing.sh precompute_vanilla
. submit_precompute_example_pairing.sh precompute_random
. submit_precompute_example_pairing.sh precompute_bm25
. submit_precompute_example_pairing.sh precompute_oracle
. submit_precompute_example_pairing.sh precompute_vanilla_upperbound
. submit_precompute_example_pairing.sh precompute_random_upperbound
. submit_precompute_example_pairing.sh precompute_bm25_upperbound
. submit_precompute_example_pairing.sh precompute_oracle_upperbound