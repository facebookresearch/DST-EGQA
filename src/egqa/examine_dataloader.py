# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# example use: python src/egqa/examine_dataloader.py -if transferqa --use_incontext_examples --transferqa_order aligned --train_example_ranking_metric all --dev_example_ranking_metric all --test_example_ranking_metric all

import argparse
import os
import time
from copy import deepcopy

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger

from egqa import (
    DOMAIN_ORDERS,
    LATEST_GITHASH,
    Config,
    CL_Dataset,
    Seq2SeqDST_model,
)


def main():

    start_local = time.localtime()
    start_time = time.time()
    logger.info(f"Githash: {LATEST_GITHASH}")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', start_local)}")
    random_seed = global_param.seed
    pl.seed_everything(random_seed, workers=True)
    n_gpu = torch.cuda.device_count()
    logger.info(f"num gpus: {n_gpu}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"Config: {global_param}")

    DOMAINS = DOMAIN_ORDERS[global_param.domain_order_key]

    current_dst_model = Seq2SeqDST_model(global_param)
    multiwoz_dataset_processor = CL_Dataset(
        global_param, tokenizer=current_dst_model.tokenizer
    )

    memory = None
    for per_domain_idx, per_domain in enumerate(DOMAINS):
        # set all train, validation, and test dataloaders based on global_param
        multiwoz_dataset_processor.configure_all_data_loaders(
            DOMAINS, per_domain, mode=global_param.train_test_setting
        )

        multiwoz_dataset_processor.save_samples()

        if memory:
            multiwoz_dataset_processor.set_data_memory(memory)
        # store the name of current domain in model

        multiwoz_dataset_processor.update_memory()

        if global_param.upperbound_idx is not None:
            break

    import pdb

    pdb.set_trace()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Config.add_training_specific_args(parser)
    parser = Config.add_model_specific_args(parser)
    parser = Config.add_data_specific_args(parser)
    args = parser.parse_args()
    global_param = Config(args)

    main()
