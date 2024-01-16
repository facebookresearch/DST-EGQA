# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import context

import numpy as np
import pytorch_lightning as pl
from loguru import logger
from tqdm import tqdm
from transformers import T5Tokenizer

from egqa import (
    DOMAIN_ORDERS,
    Config,
    CL_Dataset,
    compute_mean_average_precision_from_ranks,
    compute_mean_reciprocal_rank,
    compute_prf,
    compute_state_change_similarity,
    create_dialogue_context,
    get_ingredients_list,
    load_raw_dataset,
    reformat_state_change,
)
from egqa.utils import flatten_nested_list, get_atomic_domains

pl.seed_everything(42)
TEST_DATA_PATH = os.environ['TEST_DATA_DIR']
parser = argparse.ArgumentParser()
parser = Config.add_training_specific_args(parser)
parser = Config.add_model_specific_args(parser)
parser = Config.add_data_specific_args(parser)
args = parser.parse_args()
args.dataset = "MultiWOZ_2.4"
global_param = Config(args)
DOMAINS = DOMAIN_ORDERS["MultiWOZ"][1]
MODEL_NAME = "t5-base"
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)


def test_reformat_state_change():
    examples = [
        (
            {'restaurant-name': ['INSERT', 'pizza hut city centre']},
            {'restaurant-name-INSERT': 'pizza hut city centre'},
        ),
        (
            {
                'restaurant-book day': ['INSERT', 'wednesday'],
                'restaurant-book people': ['INSERT', '7'],
                'restaurant-book time': ['INSERT', '19:15'],
            },
            {
                'restaurant-book day-INSERT': 'wednesday',
                'restaurant-book people-INSERT': '7',
                'restaurant-book time-INSERT': '19:15',
            },
        ),
        ({}, {}),
        ({'restaurant-name': ['DELETE', None]}, {'restaurant-name-DELETE': None}),
    ]

    for ex in examples:
        assert ex[1] == reformat_state_change(ex[0], configuration="modified")


def test_f1_score_calculation_slot_key():
    examples = [
        (({}, {'restaurant-name-INSERT': 'pizza hut city centre'}), 0),
        (({}, {}), 1),
        (({'restaurant-name-INSERT': 'pizza hut city centre'}, {}), 0),
    ]
    for ex in examples:
        assert ex[1] == compute_prf(list(ex[0][0].keys()), list(ex[0][1].keys()))


def test_f1_score_calculation_full_slot():
    examples = [
        (({}, {'restaurant-name-INSERT': 'pizza hut city centre'}), 0),
        (({}, {}), 1),
        (({'restaurant-name-INSERT': 'pizza hut city centre'}, {}), 0),
    ]
    for ex in examples:
        assert ex[1] == compute_prf(list(ex[0][0].keys()), list(ex[0][1].keys()))


def test_compute_state_change_similarity():
    examples = [
        (
            (
                {
                    'restaurant-book day': ['INSERT', 'thursday'],
                    'restaurant-book people': ['INSERT', '2'],
                    'restaurant-book time': ['INSERT', '19:45'],
                },
                {'restaurant-name': ['INSERT', 'pizza hut city centre']},
            ),  # input
            0,  # scores
        )
    ]

    for ex in examples:
        assert ex[1] == compute_state_change_similarity(ex[0][0], ex[0][1])


def test_extract_state_change():
    pass


def test_bm25_obj():
    pass


def test_compute_prf_commutative():
    for _ in tqdm(range(100)):
        x = list(set([np.random.randint(20) for _ in range(10)]))
        y = list(set([np.random.randint(20) for _ in range(10)]))
        assert compute_prf(x, y) == compute_prf(y, x), (
            x,
            y,
            compute_prf(x, y),
            compute_prf(y, x),
        )


def test_bm25_context_settings():
    data_processor = CL_Dataset(global_param, tokenizer=TOKENIZER)
    data_processor.config.bm25_context_format = "full"

    domains = DOMAINS[:1]
    target_data = load_raw_dataset(
        data_path=global_param.data_path, domains=domains, data_type="test"
    )
    target_data_ingredients = [get_ingredients_list(dialog) for dialog in target_data]
    target_turn_ingredients = flatten_nested_list(target_data_ingredients)

    full_query = create_dialogue_context(
        per_turn_ingredient=target_turn_ingredients[0], context_configuration="full"
    )

    icdst_query = create_dialogue_context(
        per_turn_ingredient=target_turn_ingredients[0], context_configuration="icdst"
    )


def test_retrieval_configurations():
    """Test whether all configurations work without bugs"""

    domains = DOMAINS[:1]
    global_param.use_incontext_examples = True
    data_processor = CL_Dataset(global_param, tokenizer=TOKENIZER)

    ranking_metrics = ["random", "bm25", "state_change_sim", "scs-bm25"]

    logger.info("Testing configuration combinations: ")
    for use_incontext_examples in [True, False]:
        for bm25_context in ["icdst", "full"]:
            for test_example_ranking_metrics in ranking_metrics:
                for train_example_ranking_metrics in ["scs-bm25", "random"]:
                    for data_split in ["train", "test", "dev"]:
                        dev_example_ranking_metrics = train_example_ranking_metrics
                        data_processor.config.small = True
                        data_processor.config.use_incontext_examples = (
                            use_incontext_examples
                        )
                        data_processor.config.bm25_context_format = bm25_context
                        data_processor.config.test_example_ranking_metrics = (
                            test_example_ranking_metrics
                        )
                        data_processor.config.train_example_ranking_metrics = (
                            train_example_ranking_metrics
                        )
                        data_processor.config.dev_example_ranking_metrics = (
                            dev_example_ranking_metrics
                        )

                        atomic_domains = get_atomic_domains(domains)
                        data_loader = data_processor.set_data_loader(
                            data_domains=domains,
                            schema_domains=atomic_domains,
                            example_domains=domains,
                            data_split=data_split,
                        )

    logger.info("Completed testing configuration combinations.")


def test_retrieval_results(dataset: str, inspect: str = False):
    all_rankings = defaultdict(dict)
    all_ingredients = defaultdict(dict)

    retrieval_results_for_inspection = {}
    test_metrics = [
        "bm25",
        "scs-bm25",
        # "state_change_sim",
        # "random",
        "icdst",
        "sentbert",
        # "gpt",
        "custom_icdst"
    ]
    # test_metrics = ["random"]

    global_param.use_incontext_examples = True
    global_param.retrieval_corpus_context = "last"
    global_param.example_topk = 3
    global_param.small = False

    if dataset == "SGD":
        DOMAINS = DOMAIN_ORDERS["SGD"][1]
        global_param.dataset = "SGD"
    else:
        DOMAINS = DOMAIN_ORDERS["MultiWOZ"][1]
        global_param.dataset = "MultiWOZ_2.4"

    global_param.init_handler(global_param.dataset)

    # setup dictionary with oracle pairing results (top 10 examples) for first domain
    for idx in range(len(DOMAINS)):
        # if not inspect and idx > 0:
        #     continue
        domains = DOMAINS[idx : idx + 1]
        data_processor = CL_Dataset(global_param, tokenizer=TOKENIZER)

        target_data = load_raw_dataset(
            data_path=global_param.data_path, domains=domains, data_type="test"
        )
        target_data_ingredients = [
            get_ingredients_list(dialog) for dialog in target_data
        ]
        target_turn_ingredients = flatten_nested_list(target_data_ingredients)
        retrieval_ingredients = data_processor.load_retrieval_ingredients(
            domains=domains, data_split="test"
        )

        # test ingredients for proper retrieval
        assert len(retrieval_ingredients['similarity_matrix']) == len(
            target_turn_ingredients
        )

        sample_tokenized_query = TOKENIZER.tokenize("hello")
        sample_bm25_doc_scores = retrieval_ingredients['bm25'].get_scores(
            sample_tokenized_query
        )
        state_change_similarity_scores = retrieval_ingredients['similarity_matrix'][
            0
        ].tolist()
        assert len(sample_bm25_doc_scores) == len(state_change_similarity_scores)

        random.seed(42)
        target_turn_ingredients = random.sample(target_turn_ingredients, 10)
        all_example_ingredients = defaultdict(dict)
        for target_turn in tqdm(target_turn_ingredients):
            for test_ranking_metric in test_metrics:
                data_processor.config.test_example_ranking_metric = test_ranking_metric
                (
                    retrieved_example_ingredients,
                    turn_id2scores_dict,
                ) = data_processor.get_retrieved_ingredients_for_target(
                    target_turn,
                    retrieval_ingredients=retrieval_ingredients,
                    data_split="test",
                )
                all_rankings[test_ranking_metric][target_turn['turn_id']] = list(
                    turn_id2scores_dict.keys()
                )
                all_example_ingredients[test_ranking_metric][
                    target_turn['turn_id']
                ] = retrieved_example_ingredients

            retrieval_results_for_inspection[target_turn['turn_id']] = {
                test_ranking_metric: [
                    create_dialogue_context(
                        turn_ingredient,
                        context_configuration=global_param.retrieval_corpus_context,
                    )
                    for turn_ingredient in all_example_ingredients[test_ranking_metric][
                        target_turn['turn_id']
                    ]
                ]
                for test_ranking_metric in test_metrics
            }

            retrieval_results_for_inspection[target_turn['turn_id']][
                "query"
            ] = create_dialogue_context(
                target_turn, context_configuration=global_param.retrieval_corpus_context
            )

    turn_ids = list(all_rankings['scs-bm25'].keys())

    gold_ranks = [all_rankings['scs-bm25'][turn_id] for turn_id in turn_ids]

    print("approach\tMAP\tMRR")
    for retrieval_approach, rankings in all_rankings.items():
        pred_ranks = [rankings[turn_id] for turn_id in turn_ids]
        top_predictions = [rank[0] for rank in pred_ranks]

        map_ = compute_mean_average_precision_from_ranks(gold_ranks, pred_ranks)
        mrr = compute_mean_reciprocal_rank(gold_ranks, top_predictions)
        print(
            f"{retrieval_approach:<20}\t{map_:.4f}\t{mrr:.4f}"
        )

    save_path = os.path.join(
        TEST_DATA_PATH, f"{global_param.dataset}_retrieval_results.json"
    )
    # save retrieval results
    with open(save_path, "w") as f:
        random.seed(42)
        sampled_turn_ids = random.sample(turn_ids, min(100, len(turn_ids)))
        retrieval_results_samples = {
            turn_id: retrieval_results_for_inspection[turn_id]
            for turn_id in sampled_turn_ids
        }
        json.dump(retrieval_results_samples, f, indent=4)

    logger.info(f"Retrieval results samples saved to {save_path}")

    return


if __name__ == "__main__":
    # test_retrieval_configurations()
    test_retrieval_results(dataset="SGD", inspect=False)
    # test_retrieval_results(dataset="MultiWOZ_2.4", inspect=False)
