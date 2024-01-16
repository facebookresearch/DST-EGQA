# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import os
from pathlib import Path

TEST_DATA_PATH = os.environ['TEST_DATA_DIR']


def load_sample_test_data():

    # sample conversation from test data
    sample_data_path = Path(TEST_DATA_PATH) / "example_CL_data.json"
    with sample_data_path.open("r") as f:
        sample_data = json.load(f)

    # sample conversation from training data as db
    sample_db_data_path = Path(TEST_DATA_PATH) / "example_CL_db_data.json"
    with sample_db_data_path.open("r") as f:
        sample_db_data = json.load(f)

    # example from training data
    target_ingredients_path = Path(TEST_DATA_PATH) / "example_ingredients.json"
    with target_ingredients_path.open("r") as f:
        sample_ingredients = json.load(f)

    # sample transferqa seqs
    example_transferqa_seqs_path = Path(TEST_DATA_PATH) / "example_transferqa_seqs.json"
    with example_transferqa_seqs_path.open("r") as f:
        sample_transferqa_seqs = json.load(f)

    # sample transferqa seq orders
    example_transferqa_seq_orders_path = (
        Path(TEST_DATA_PATH) / "example_alignment_outputs.json"
    )
    with example_transferqa_seq_orders_path.open("r") as f:
        sample_transferqa_seq_orders = json.load(f)

    # sample transferqa example target sequence alignment
    example_transferqa_example_target_question_alignment_path = (
        Path(TEST_DATA_PATH) / "example_target_question_alignment_outputs.json"
    )
    with example_transferqa_example_target_question_alignment_path.open("r") as f:
        sample_transferqa_example_target_question_alignment = json.load(f)

    # full examples formed from training data witout examples
    target_transferqa_data_path = (
        Path(TEST_DATA_PATH) / "example_transferqa_test_data.json"
    )
    with target_transferqa_data_path.open("r") as f:
        target_transferqa_samples = json.load(f)

    return {
        "sample_data": sample_data,
        "sample_db_data": sample_db_data,
        "sample_transferqa_seqs": sample_transferqa_seqs,
        "sample_ingredients": sample_ingredients,
        "sample_transferqa_seq_orders": sample_transferqa_seq_orders,
        "sample_transferqa_example_target_question_alignment": sample_transferqa_example_target_question_alignment,
        "transferqa_samples": target_transferqa_samples,
    }
