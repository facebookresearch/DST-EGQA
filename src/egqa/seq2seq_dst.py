# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from loguru import logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule

from egqa.data_utils.mwoz_constants import *
from egqa.data_utils.general_constants import * 
from egqa.models.fid import FiDT5
from egqa.utils import CosineWarmupScheduler, compute_jga, extract_slot_from_string

KEY_RESULTS = {"val": defaultdict(list), "test": defaultdict(list)}
TEST_PREDICTIONS_RAW = {}


class CustomJGA(torchmetrics.Metric):

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: List[int], target: List[int]):
        preds, target = torch.tensor(preds), torch.tensor(target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):

        return self.correct.float() / self.total


class Seq2SeqDST_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        self.val_jga = CustomJGA()
        self.test_jga = CustomJGA()
        self.last_model = None
        self.state = {"epoch": 0}
        self.showed_sample = False 

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            bos_token="[bos]",
            eos_token="[eos]",
            sep_token="[sep]",
        )

        special_tokens_dict = {
            'additional_special_tokens': [
                EXAMPLE_TAG,
                USER_TAG,
                SYSTEM_TAG,
                TARGET_TAG,
                OPTIONS_SEPARATOR,
                DS_TAG,
            ]
        }
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        # initialize model
        if "fid" not in config.model_name:         
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else: 
            # self.model = 
            raise NotImplementedError

    @staticmethod
    def save_results(config, elapsed_time, domains):

        from egqa import (
            compute_backward_transfer,
            compute_cl_metrics,
            compute_forward_transfer,
            compute_upperbound_metrics,
        )

        KEY_RESULTS["ordered_test_results"] = [
            KEY_RESULTS["test"][dom] for dom in domains
        ]

        with open(config.test_predictions_raw_path, 'w') as f:
            json.dump(TEST_PREDICTIONS_RAW, f, indent=4)

        # TODO save results that can be directly copied to excel
        # compute metrics and save them
        if config.upperbound_idx is not None:
            main_results = compute_upperbound_metrics(
                TEST_PREDICTIONS_RAW,
                dataset=config.dataset, 
                trained_domains=domains,
                strategy=config.input_format,
                eval_mode="known",
            )
            eval_config = "ub"
        else:
            main_results = compute_cl_metrics(
                TEST_PREDICTIONS_RAW,
                dataset=config.dataset, 
                trained_domain_order=domains,
                strategy=config.input_format,
                eval_mode="known",
            )
            fwt_result = compute_forward_transfer(
                main_results['complete_jga_matrix'], trained_domain_order=domains
            )
            bwt_result = compute_backward_transfer(
                main_results['complete_jga_matrix'], trained_domain_order=domains
            )
            main_results['fwt'] = fwt_result
            main_results['bwt'] = bwt_result
            eval_config = "cl"

        main_results['elapsed_time'] = elapsed_time
        KEY_RESULTS[eval_config] = main_results
        KEY_RESULTS["config"] = vars(config)
        
        with open(config.key_results_path, "w") as f:
            json.dump(KEY_RESULTS, f, indent=4)

    def configure_optimizers(self):
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
            if self.config.warmup_steps:
                lr_scheduler = CosineWarmupScheduler(
                    optimizer, warmup=self.config.warmup_steps, max_iters=self.max_iters
                )
                logger.info("Using CosineWarmup scheduler")
            else:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "max"
                )
                logger.info("Using ReduceLROnPlateau scheduler")

        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
        else:
            # reference: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor
            # optimizer = Adafactor(self.model.parameters(), lr=self.config.learning_rate, clip_threshold = 1.0)
            optimizer = Adafactor(
                self.model.parameters(),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=self.config.learning_rate,
            )
            # optimizer = Adafactor(
            #     self.model.parameters(),
            #     lr=self.config.learning_rate,
            #     eps=(1e-30, 1e-3),
            #     clip_threshold=1.0,
            #     decay_rate=-0.8,
            #     beta1=None,
            #     weight_decay=0.0,
            #     relative_step=False,
            #     scale_parameter=False,
            #     warmup_init=False
            # )
            # lr_scheduler = AdafactorSchedule(optimizer)
            lr_scheduler = None
        optimizer_configuration = {
            "optimizer": optimizer,
            "monitor": "val_jga",
        }
        if lr_scheduler:
            optimizer_configuration["lr_scheduler"] = lr_scheduler
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer_configuration
        # return FusedAdam(self.model.parameters(), lr=self.config.learning_rate)
        # return DeepSpeedCPUAdam(self.model.parameters(), lr=self.config.learning_rate)

    def training_step(self, batch, batch_idx):
        self.model.train()
        # attention_masks
        loss = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        ).loss

        # TODO: add knowledge distillation losses (KPN)
        # if (self.config.knowledge_type in ['KPN']) & (self.last_model is not None):
        #     loss = loss + par.alpha_1 * (loss_encoder_slot + loss_decoder_hidden)

        # TODO: add regularization
        # if (self.config.regularize):

        # TODO: log parlai metrics

        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        self.state["epoch"] += 1
        return super().training_epoch_end(outputs)

    def shared_step(self, batch, batch_idx):
        """shared step for validation and test steps"""
        loss = self.training_step(batch, batch_idx)['loss']

        candidates_idx = self.model.generate(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.config.max_new_tokens,
        )

        decoded_candidates = self.tokenizer.batch_decode(
            candidates_idx, skip_special_tokens=True
        )

        return loss, decoded_candidates

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if batch_idx == 0:
            self.val_jga_list = []
            self.val_ds_predictions = defaultdict(list)
            self.val_ds_labels = defaultdict(list)

        loss, decoded_candidates = self.shared_step(batch, batch_idx)

        batch_len = len(decoded_candidates)
        batch["decoded_candidates"] = decoded_candidates

        self.val_jga_list = []

        for batch_idx in range(batch_len):
            decode_candidate = batch["decoded_candidates"][batch_idx]
            label = batch["output_text"][batch_idx]
            input_text = batch["input_text"][batch_idx]
            slot = batch["slot"][batch_idx]
            turn_id = batch["idx"][batch_idx]



            # if slots are individually predicted
            if self.config.input_format == "transferqa":
                # defer calculating JGA until reaching the end of epoch
                # map back to slot from question asked
                pred_slot = f"{slot} {decode_candidate}"
                gold_slot = f"{slot} {label}"
                self.val_ds_predictions[turn_id].append(pred_slot)
                self.val_ds_labels[turn_id].append(gold_slot)
                
                if decode_candidate != label:
                    logger.debug(
                        f"\ninput_text: {input_text} \n\tgold text: {label}\n\tpred text: {decode_candidate}"
                        + f"\nDST prediction: \n\tgold: {gold_slot}\n\tpred: {pred_slot}"
                    )
                    
                if not self.showed_sample: 
                    logger.info("Showing sample input/output")
                    logger.info(
                        f"\ninput_text: {input_text} \n\tgold text: {label}\n\tpred text: {decode_candidate}"
                        + f"\nDST prediction: \n\tgold: {gold_slot}\n\tpred: {pred_slot}"
                    )
                    self.showed_sample = True
                    
            # if all slots are predicted in one go
            else:
                pred_ds, _, _ = extract_slot_from_string(
                    decode_candidate, self.config.valid_domains
                )
                gold_ds, _, _ = extract_slot_from_string(
                    label, self.config.valid_domains
                )

                jga = compute_jga(gold_ds, pred_ds)
                self.val_jga_list.append(jga)

                if jga == 0:
                    logger.debug(
                        f"\ninput_text: {input_text} \n\tgold text: {label}\n\tpred text: {decode_candidate}"
                        + f"\nDST prediction: \n\tgold: {gold_ds}\n\tpred: {pred_ds}"
                    )

                if not self.showed_sample: 
                    logger.info("Showing sample input/output")
                    logger.info(
                        f"\ninput_text: {input_text} \n\tgold text: {label}\n\tpred text: {decode_candidate}"
                            + f"\nDST prediction: \n\tgold: {gold_ds}\n\tpred: {pred_ds}"
                    )
                    self.showed_sample = True

        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs) -> None:
        # TODO aggregate results for multi-gpu setting

        if self.config.input_format == "transferqa":
            self.val_jga_list = []
            for dial_idx in self.val_ds_predictions.keys():
                jga = compute_jga(
                    self.val_ds_labels[dial_idx], self.val_ds_predictions[dial_idx]
                )
                self.val_jga_list.append(jga)

        self.val_jga(self.val_jga_list, [1] * len(self.val_jga_list))
        self.log("val_jga", self.val_jga)
        logger.info("Logging at the end of validation...")
        val_jga = np.mean(self.val_jga_list)
        logger.info(
            f"Validation JGA for {self.domain} at epoch {self.state['epoch']}: {val_jga:.4f}"
        )
        KEY_RESULTS["val"][self.domain].append(val_jga)

        return {"val_jga": np.mean(self.val_jga_list)}

    def test_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        self.model.eval()
        if batch_idx == 0:
            self.test_jga_list = []
            self.per_domain_jga_list = defaultdict(list)
            self.test_ds_predictions = defaultdict(list)
            self.test_input_texts = defaultdict(list)
            self.test_example_turn_ids_scores = {}
            self.test_ds_labels = defaultdict(list)
            self.test_predictions = defaultdict(list)
            self.dialid2domain = {}

        loss, decoded_candidates = self.shared_step(batch, batch_idx)

        batch_len = len(decoded_candidates)
        batch["decoded_candidates"] = decoded_candidates

        for batch_idx in range(batch_len):
            decode_candidate = batch["decoded_candidates"][batch_idx]
            label = batch["output_text"][batch_idx]
            input_text = batch["input_text"][batch_idx]
            slot = batch["slot"][batch_idx]
            turn_id = batch["idx"][batch_idx]
            domain = batch["domain"][batch_idx]
            example_turn_ids_scores = batch["example_turn_ids_scores"][batch_idx]

            # if slots are individually predicted
            if self.config.input_format == "transferqa":
                # defer calculating JGA until reaching the end of epoch
                # format back to slot key value
                pred_slot = f"{slot} {decode_candidate}"
                gold_slot = f"{slot} {label}"

                self.test_input_texts[turn_id].append(input_text)
                self.test_ds_predictions[turn_id].append(pred_slot)
                self.test_ds_labels[turn_id].append(gold_slot)

                # examples and scores used for the same dial id should be the same
                if turn_id in self.test_example_turn_ids_scores:
                    # assert (
                    #     self.test_example_turn_ids_scores[turn_id]
                    #     == example_turn_ids_scores
                    # )
                    
                    if self.test_example_turn_ids_scores[turn_id] != [example_turn_ids_scores]: 
                        logger.warning(f"example_turn_ids_scores is not consistent: \n\texisting: {self.test_example_turn_ids_scores[turn_id]}\n\tnew:{example_turn_ids_scores}")
                        self.test_example_turn_ids_scores[turn_id] = self.test_example_turn_ids_scores[turn_id] + [example_turn_ids_scores]
                else:
                    self.test_example_turn_ids_scores[turn_id] = [example_turn_ids_scores]
                self.dialid2domain[turn_id] = domain

            # if all slots are predicted in one go
            else:
                pred_ds, _, _ = extract_slot_from_string(
                    decode_candidate, self.config.valid_domains
                )
                gold_ds, _, _ = extract_slot_from_string(
                    label, self.config.valid_domains
                )

                jga = compute_jga(gold_ds, pred_ds)
                self.test_jga_list.append(jga)

                self.per_domain_jga_list[domain].append(jga)
                self.test_predictions[domain].append(
                    {
                        "turn_id": turn_id,
                        "input_text": input_text,
                        "pred": decode_candidate,
                        "gold": label,
                        "jga": jga,
                        "example_turn_ids_scores": example_turn_ids_scores,
                    }
                )

                if jga == 0:
                    logger.debug(
                        f"\ninput_text: {input_text} \n\tgold text: {label}\n\tpred text: {decode_candidate}"
                        + f"\nDST prediction: \n\tgold: {gold_ds}\n\tpred: {pred_ds}"
                    )

        self.log("test_loss", loss, on_step=True, on_epoch=True)

        # in sequential mode, customized batches each contains a batch of turns with the same turn idx
        # TODO create separate process for sequential mode
        # self.prev_dst = None
        # update input with prev_dst
        # follow same steps

    def test_epoch_end(self, outputs) -> None:
        # TODO aggregate results for multi-gpu setting

        if self.config.input_format == "transferqa":
            self.test_jga_list = []
            for dial_idx in self.dialid2domain.keys():
                jga = compute_jga(
                    self.test_ds_labels[dial_idx], self.test_ds_predictions[dial_idx]
                )
                domain = self.dialid2domain[dial_idx]

                self.test_jga_list.append(jga)
                self.per_domain_jga_list[domain].append(jga)
                self.test_predictions[domain].append(
                    {
                        "turn_id": dial_idx,
                        "jga": jga,
                        "pred": self.test_ds_predictions[dial_idx],
                        "gold": self.test_ds_labels[dial_idx],
                        "sample_input_text": self.test_input_texts[dial_idx],
                        "example_turn_ids_scores": self.test_example_turn_ids_scores[
                            dial_idx
                        ],
                    }
                )

        self.test_jga(self.test_jga_list, [1] * len(self.test_jga_list))
        self.log("test_jga", self.test_jga)
        results = {"test_jga": np.mean(self.test_jga_list)}
        for domain, jga_list in self.per_domain_jga_list.items():
            per_domain_jga = np.mean(jga_list)
            self.log(f"test_jga_{domain}", per_domain_jga)
            results[f"test_jga_{domain}"] = per_domain_jga
            logger.info(f"test_jga_{domain},{per_domain_jga:.4f}")

        logger.info("Logging at the end of testing...")
        test_jga = np.mean(self.test_jga_list)
        logger.info(f"test_jga,{test_jga:.4f}")
        logger.info(
            f"TEST JGA for {self.domain} at epoch {self.state['epoch']}: {test_jga:.4f}"
        )
        KEY_RESULTS["test"][self.domain].append(test_jga)
        TEST_PREDICTIONS_RAW[self.domain] = self.test_predictions

        return results

    def set_max_iters(self, total_train_steps: int):
        """set maximum iterations, which is needed for the warm up scheduler

        Args:
            total_train_steps (int): length of train dataloader
        """
        self.max_iters = total_train_steps * self.config.epochs
        logger.info(f"Maximum number of training steps: {self.max_iters}")

    def set_callbacks(self):
        """set callbacks to use for the trainer"""

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.result_path,
            save_top_k=1,
            monitor="val_jga",
            mode="max",
            filename=self.domain,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        self.checkpoint_callback = checkpoint_callback
        self.callbacks = [checkpoint_callback, lr_monitor]
