# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

    DOMAINS = DOMAIN_ORDERS[global_param.dataset][global_param.domain_order_key]

    current_dst_model = Seq2SeqDST_model(global_param)
    cl_dataset_processor = CL_Dataset(
        global_param, tokenizer=current_dst_model.tokenizer
    )

    best_model_ckpt = None
    memory = None
    for per_domain_idx, per_domain in enumerate(DOMAINS):
        # set all train, validation, and test dataloaders based on global_param
        cl_dataset_processor.configure_all_data_loaders(
            DOMAINS, per_domain, data_split=global_param.train_test_setting
        )

        if global_param.precompute_only:
            continue

        if memory:
            cl_dataset_processor.set_data_memory(memory)
        # store the name of current domain in model
        current_dst_model.domain = per_domain

        os.makedirs(os.path.join(global_param.result_path, per_domain), exist_ok=True)
        # logger = WandbLogger(project="CLDST")
        # logger = TensorBoardLogger("tb_logs", name="my_model")

        # set max iterations value for warm up scheduler
        # current_dst_model.set_max_iters(
        #     total_train_steps=len(multiwoz_dataset_processor.train_dataloader())
        # )
        # set callbacks
        current_dst_model.set_callbacks()

        try: 
            precision = int(global_param.precision)
        except: 
            precision = global_param.precision

        ####### TRAIN #######
        trainer = pl.Trainer(
            deterministic=True,
            accelerator="auto",
            callbacks=current_dst_model.callbacks,
            max_epochs=global_param.epochs,
            precision=precision,
            gradient_clip_val=global_param.gradient_clip_val,
        )

        if global_param.train_test_setting == 'train':
            logger.info(f'Train: {per_domain}')

            trainer.fit(model=current_dst_model, datamodule=cl_dataset_processor)

        ###### TEST ######
        trained_domains = (
            DOMAINS[: global_param.upperbound_idx + 1]
            if global_param.upperbound_idx is not None
            else DOMAINS[: DOMAINS.index(per_domain) + 1]
        )
        logger.info(f'Test after training with domains: `{trained_domains}`')

        # load best model from training based on validation set
        if global_param.train_test_setting == "train":
            best_model_ckpt = current_dst_model.checkpoint_callback.best_model_path
        elif global_param.train_test_setting == "test":
            best_model_ckpt = os.path.join(
                global_param.result_path, f"{per_domain}.ckpt"
            )
            assert os.path.isfile(
                best_model_ckpt
            ), f"Model path: {best_model_ckpt} does not exist"
        current_dst_model = Seq2SeqDST_model.load_from_checkpoint(
            best_model_ckpt, config=global_param
        )
        current_dst_model.domain = per_domain

        # run test
        test_results = trainer.test(
            model=current_dst_model, datamodule=cl_dataset_processor
        )

        logger.info(
            f"test results after training with {trained_domains}: {test_results}"
        )

        # not necessary to loop through the entire set of domains when multitasking to get upper bound
        if global_param.upperbound_idx is not None:
            break

        ### Any updates before training with next domain
        # keep previous model weights for regularization
        current_dst_model.last_model = (
            deepcopy(current_dst_model.bert_dst_model)
            if global_param.regularize
            else None
        )

        # update memory
        cl_dataset_processor.update_memory()

    # log how long it took
    end_time = time.time()
    end_local = time.localtime()
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', end_local)}")
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    logger.info(f"Elapsed time: {elapsed_time}")

    if not global_param.precompute_only:
        # save dictionaries containing key results and predictions for the test set
        Seq2SeqDST_model.save_results(global_param, elapsed_time, DOMAINS)

    return 0


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    parser = argparse.ArgumentParser()
    parser = Config.add_training_specific_args(parser)
    parser = Config.add_model_specific_args(parser)
    parser = Config.add_data_specific_args(parser)
    args = parser.parse_args()
    global_param = Config(args)

    # track logger outputs
    logger.add(global_param.log_path)
    # save parameters used to global_param.result_path
    global_param.save()

    # main train / validation / testing sequence
    main()
