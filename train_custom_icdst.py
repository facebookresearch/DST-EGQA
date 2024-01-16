# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# refer to https://github.com/wise-east/IC-DST/blob/main/retriever/code/retriever_finetuning.py

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    LoggingHandler,
)

from torch.utils.data import DataLoader
import torch
from transformers import T5Tokenizer
import pytorch_lightning as pl
import os
import argparse
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from tqdm import tqdm
from collections import defaultdict
from loguru import logger

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
parser.add_argument("--evaluator", type=str, default="triplet", help="One of [triplet, emb_sim]")
args = parser.parse_args()
global_param = Config(args)

BASE_MODEL = "all-mpnet-base-v2"
DOMAINS = DOMAIN_ORDERS[args.dataset][1]
MODEL_NAME = "t5-small"
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)

global_param.use_incontext_examples = True
global_param.retrieval_corpus_context = "last"
global_param.example_topk = 100
global_param.small = False

if global_param.dataset == "SGD":
    DOMAINS = DOMAIN_ORDERS["SGD"][global_param.domain_order_key]
    global_param.dataset = "SGD"
else:
    DOMAINS = DOMAIN_ORDERS["MultiWOZ"][global_param.domain_order_key]
    global_param.dataset = "MultiWOZ_2.4"

global_param.init_handler(global_param.dataset)


# form ic-dst data
gold_ranking_metric = "scs-bm25"
data_processor = CL_Dataset(global_param, tokenizer=TOKENIZER)
data_processor.config.test_example_ranking_metric = gold_ranking_metric

train_examples = []
dev_examples = defaultdict(list)

top_range = 200 
top_k = 10 

for data_split in ["train", "dev"]:
    for idx in range(len(DOMAINS)):
        # only collect data for first domain to meet CL setup
        if idx > 0:
            continue

        domains = DOMAINS[idx : idx + 1]

        target_data = load_raw_dataset(
            data_path=global_param.data_path, domains=domains, data_type=data_split
        )
        target_data_ingredients = [
            get_ingredients_list(dialog) for dialog in target_data
        ]
        target_turn_ingredients = flatten_nested_list(target_data_ingredients)


        if data_split == "train": 
            train_data_size = len(target_turn_ingredients)

        retrieval_ingredients = data_processor.load_retrieval_ingredients(
            domains=domains, data_split=data_split
        )

        # test ingredients for proper retrieval
        assert len(retrieval_ingredients['similarity_matrix']) == len(
            target_turn_ingredients
        )
        
        # from top range, top k is positive and top k is negative

        logger.info(f"Top range, top_k: {top_range},{top_k}")

        # target_turn_ingredients = random.sample(target_turn_ingredients, 10)
        for target_turn in tqdm(target_turn_ingredients):
            
            input_query = create_dialogue_context(
                per_turn_ingredient=target_turn,
                context_configuration=global_param.retrieval_corpus_context,
            )

            (
                retrieved_example_ingredients,
                turn_id2scores_dict,
            ) = data_processor.get_retrieved_ingredients_for_target(
                target_turn,
                retrieval_ingredients=retrieval_ingredients,
                data_split=data_split,
            )


            assert len(retrieved_example_ingredients) >= top_k * 2, (
                len(retrieved_example_ingredients),
                top_k * 2,
            )

            top5pct = [
                create_dialogue_context(
                    per_turn_ingredient=top_ingred,
                    context_configuration=global_param.retrieval_corpus_context,
                )
                for top_ingred in retrieved_example_ingredients[:top_k]
            ]
            bottom5pct = [
                create_dialogue_context(
                    per_turn_ingredient=bot_ingred,
                    context_configuration=global_param.retrieval_corpus_context,
                )
                for bot_ingred in retrieved_example_ingredients[-top_k:]
            ]

            if data_split == "train":
                for top_ex in top5pct:
                    train_examples.append(
                        InputExample(texts=[input_query, top_ex], label=1)
                    )

                for bot_ex in bottom5pct:
                    train_examples.append(
                        InputExample(texts=[input_query, bot_ex], label=0)
                    )

            if data_split == "dev":
                
                if args.evaluator == "triplet": 
                    for top_ex, bot_ex in zip(top5pct, bottom5pct):
                        dev_examples['anchors'].append(input_query)
                        dev_examples['positives'].append(top_ex)
                        dev_examples['negatives'].append(bot_ex)
                elif args.evaluator == "emb_sim": 
                    for top_ex in top5pct: 
                        dev_examples['sentences1'].append(input_query)
                        dev_examples['sentences2'].append(top_ex)
                        dev_examples['scores'].append(1)
                        
                    for bot_ex in bottom5pct: 
                        dev_examples['sentences1'].append(input_query)
                        dev_examples['sentences2'].append(bot_ex)
                        dev_examples['scores'].append(0)

logger.info(f"Sample: \n\tinput query: {target_turn['state_change']} {input_query}\n\tpositive: {retrieved_example_ingredients[:top_k][-1]['state_change']} {top_ex}\n\tnegative: {retrieved_example_ingredients[-1]['state_change']} {bot_ex}")

# import pdb; pdb.set_trace()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(BASE_MODEL, device=device)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.OnlineContrastiveLoss(model=model)



if args.evaluator == "triplet": 
    evaluator = evaluation.TripletEvaluator(anchors=dev_examples["anchors"], positives=dev_examples["positives"], negatives=dev_examples["negatives"])

elif args.evaluator == "emb_sim":     
    dev_sentences1 = dev_examples['sentences1']
    dev_sentences2 = dev_examples['sentences2']
    dev_scores = dev_examples['scores']
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        dev_sentences1, dev_sentences2, dev_scores
    )


SAVE_MODEL_PATH = os.path.join(
    os.environ.get("RESULTS_DIR"), f"custom_SGD_icdst_{args.evaluator}_{DOMAINS[0]}"
)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    evaluator=evaluator,
    evaluation_steps=500,
    show_progress_bar=True,
    save_best_model=True,
    output_path=SAVE_MODEL_PATH,
)
