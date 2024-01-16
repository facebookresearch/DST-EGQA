# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import os
import time
from distutils import util
from pathlib import Path
import glob 

import torch
from loguru import logger

from egqa.data_utils.mwoz_constants import MWOZ_VALID_DOMAINS
from egqa.data_utils.sgd_constants import SGD_DOMAINS_OF_INTEREST

DATA_DIR = os.environ["DATA_DIR"]
RESULTS_DIR = os.environ["RESULTS_DIR"]


class Config:
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CL_dataloader")

        parser.add_argument(
            "-dok",
            "--domain_order_key",
            default=1,
            type=int,
            help="one of [1 2 3 4 5] for SGD and [1 2 3] for MultiWOZ",
        )

        parser.add_argument(
            "-msl",
            "--max_source_length",
            default=1024,
            type=int,
            help="max length to use for source",
        )

        parser.add_argument(
            "-mtl",
            "--max_target_length",
            default=512,
            type=int,
            help="max length to use for target",
        )

        parser.add_argument(
            "-if",
            "--input_format",
            default='transferqa',
            type=str,
            help="one of [simpletod, transferqa]",
        )
        
        parser.add_argument(
            "--transferqa_detailed_format", 
            default="simple", 
            type=str, 
            help="[simple, old (question-alignment configurable)]"
        )

        parser.add_argument(
            "--transferqa_none_ratio",
            default=-1.0,
            type=float,
            help="maximum ratio of questions with none answers to those with non-none answers. -1 will use all questions, 1 will use equal amounts. mainly for speeding up training",
        )

        parser.add_argument(
            "--transferqa_order",
            default="aligned",
            type=str,
            help="alignment of questions between examples and target: whether to keep them aligned, mix it up randomly, or order them as non-nones followed by nones (to replicate bug setting). one of [aligned, random, mixed]",
        )

        parser.add_argument(
            "--input_context_format",
            default="full",
            type=str,
            help="format of context that will be used for DST task",
        )

        parser.add_argument(
            "--retrieval_corpus_context",
            default="icdst",
            type=str,
            help="portion of each examples to be considered for comparing bm25 score",
        )

        parser.add_argument(
            "--use_incontext_examples",
            action="store_true",
            help="whether to use incontext examples",
        )

        parser.add_argument(
            "--example_topk",
            type=int,
            default=1,
            help="how many in context examples to use",
        )

        parser.add_argument(
            "--train_example_ranking_metric",
            type=str,
            default="scs-bm25",
            help="what metric to use for ranking: one of [bm25, state_change_sim, all, random]",
        )

        parser.add_argument(
            "--dev_example_ranking_metric",
            type=str,
            default="scs-bm25",
            help="what metric to use for ranking: one of [bm25, state_change_sim, all, random]",
        )

        parser.add_argument(
            "--test_example_ranking_metric",
            type=str,
            default="scs-bm25",
            help="what metric to use for ranking: one of [bm25, state_change_sim, all, random]",
        )

        parser.add_argument(
            "--retrieval_data_budget",
            type=int,
            default=0,
            help="the number of conversations from each domain to restrict the database size for retrieval. default=0, which means no restriction.",
        )
        
        parser.add_argument(
            "--custom_icdst_evaluator",
            type=str,
            default="triplet",
            help="one of [triplet, emb_sim]"
        )

        parser.add_argument(
            "--precompute_only",
            action="store_true",
            help="whether to only run script for precomputing state change similarities and example pairings and storing them as cache",
        )

        parser.add_argument(
            "--save_input_outputs",
            action="store_true",
            help="whether to save input outputs",
        )

        return parent_parser

    @staticmethod
    def add_training_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Training")

        parser.add_argument("--dataset", default='SGD', help="One of MultiWOZ_2.4, MultiWOZ_2.1 or SGD")  # train test
        parser.add_argument(
            "-tts", '--train_test_setting', default='train'
        )  # train test

        parser.add_argument(
            "--debug",
            action="store_true",
            help="whether to use a small set for debugging purposes",
        )

        parser.add_argument(
            "--small",
            action="store_true",
            help="whether to use a small set for testing purposes",
        )

        parser.add_argument(
            "--cumulate_dev_set",
            action="store_true",
            help="whether to cumulate the dev set as training proceeds. Setting this to True is KPN's setting.",
        )

        parser.add_argument(
            "--gradient_clip_val",
            default=1.0,
            type=float,
            help="gradient clipping value to use.",
        )

        parser.add_argument(
            "--use_cuda",
            type=util.strtobool,
            default=torch.cuda.is_available(),
            help="whether to use gpu",
        )

        parser.add_argument(
            "--precision", type=str, default="32", help="one of bf16, 16, 32, 64"
        )

        parser.add_argument(
            "-lr",
            "--learning_rate",
            default=1e-4,
            type=float,
            help="learning rate to use.",
        )

        parser.add_argument(
            "-s", "--seed", default=40, type=int, help="number of epochs to train."
        )

        parser.add_argument(
            "-ep", "--epochs", default=10, type=int, help="number of epochs to train."
        )

        parser.add_argument(
            "-ws",
            "--warmup_steps",
            default=500,
            type=int,
            help="number of steps for warmup.",
        )
        parser.add_argument(
            "-vt",
            "--validation_target",
            default="dev",
            type=str,
            help="one of [train dev test]. set to 'train' to see overfitting",
        )

        parser.add_argument(
            "--domain_not_known",
            action="store_true",
            help="Whether the test domains are known or unknown ahead of time.",
        )

        parser.add_argument(
            "-bs",
            "--batch_size",
            default=16,
            type=int,
            help="batch size to use for training",
        )

        parser.add_argument(
            "-vbs",
            "--eval_batch_size",
            default=32,
            type=int,
            help="batch size to use for validation",
        )

        parser.add_argument(
            "-tbs",
            "--test_batch_size",
            default=32,
            type=int,
            help="batch size to use for testing",
        )

        parser.add_argument(
            "--regularize",
            action="store_true",
            help="whether to regularize model",
        )

        parser.add_argument(
            "-ms",
            "--memory_strategy",
            default="none",
            type=str,
            help="one of [none random prototype]",
        )

        parser.add_argument(
            "-mn",
            "--memory_num",
            default=0,
            type=int,
            help="number of conversations to keep in the memory replay buffer",
        )

        parser.add_argument(
            "--upperbound_idx",
            default=None,
            type=int,
            help="specify the DOMAINS[:upperbound_idx] and get CL upperbound by learning with all current + previous domain data",
        )

        parser.add_argument(
            "--multitask_all",
            action="store_true",
            help="whether to multitask with all available training data",
        )

        parser.add_argument(
            "-mp",
            "--modelpath",
            default=None,
            type=str,
            help="set to a directory to evaluate checkpoints",
        )

        parser.add_argument(
            "-rp",
            "--result_path",
            default=None,
            type=str,
            help="provide custom result path",
        )

        return parent_parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Seq2SeqDST_model")

        parser.add_argument(
            "-m",
            "--model_name",
            default="t5-small",
            type=str,
            help="huggingface model to use. one of [t5-base t5-large t5-3b t5-11b, google/t5-v1_1-base google/t5-v1_1-large facebook/bart-large facebook/bart-base]",
        )
        parser.add_argument(
            "-op",
            "--optimizer",
            default="adamw",
            type=str,
            help="optimizer to use. one of [adamw, adam, adafactor]",
        )

        parser.add_argument(
            "--max_new_tokens",
            default=256,
            type=int,
            help="maximum number of new tokens to generate response.",
        )

        return parent_parser

    def __init__(self, args):
        # set namespaces with args from ArgumentParser
        for key, value in vars(args).items():
            setattr(self, key, value)

        # set dataset specific configurations
        self.init_handler(self.dataset)
        if self.result_path is None:
            self.set_result_path()

        # evaluate when modelpath is set
        if self.modelpath:
            self.result_path = self.modelpath
            self.train_test_setting = "test"
            self.precision = "32"
            
            config_fn = list(Path(self.modelpath).glob("config*.json"))[0]
            config = json.load(open(config_fn))
            
            if "t5-base" in self.modelpath: 
                self.model_name = "t5-base"
            if "t5-small" in self.modelpath: 
                self.model_name = "t5-small"
            
            self.domain_order_key = config["domain_order_key"]
            self.train_example_ranking_metric = config["train_example_ranking_metric"]
            self.dev_example_ranking_metric = config["dev_example_ranking_metric"]

        # if in debug mode,
        if self.debug:
            logger.warning(f'Currently running in debug mode: using smaller dataset')
            self.epochs = 1

        # add index if path already exists
        if self.train_test_setting == "train":
            idx = 0
            orig_path = self.result_path
            while os.path.isdir(self.result_path):
                self.result_path = orig_path + f"_{idx}"
                idx += 1

        config_string = self.get_configuration_string(
            mode=self.train_test_setting, verbose=True
        )

        self.log_path = str(Path(self.result_path) / f"log_{config_string}.txt")
        self.key_results_path = str(
            Path(self.result_path) / f"key_results_{config_string}.json"
        )
        self.test_predictions_raw_path = str(
            Path(self.result_path) / f"test_predictions_raw_{config_string}.json"
        )

        # keep files separate for upperbound files
        if self.upperbound_idx is not None:
            self.log_path = str(
                Path(self.result_path)
                / f"log_{self.train_test_setting}_ubidx{self.upperbound_idx}.txt"
            )
            self.key_results_path = str(
                Path(self.result_path) / f"key_results_ubidx{self.upperbound_idx}.json"
            )
            self.test_predictions_raw_path = str(
                Path(self.result_path)
                / f"test_predictions_raw_ubidx{self.upperbound_idx}.json"
            )

    def get_configuration_string(self, mode: str = None, verbose: bool = False) -> str:

        """get string format summary of configuration

        Args:
            mode (str): one of train, dev, test
            for_sample (bool): set to True if we're setting the string to use for sample files

        Returns:
            str: string format summary of configuration
        """
        config_abbreviation_mapping = {
            "input_format": "if",
            "input_context_format": "icf",
            "example_topk": "ex",
            "retrieval_corpus_context": "bm25cf",
            "train_example_ranking_metric": "train-erm",
            "dev_example_ranking_metric": "dev-erm",
            "test_example_ranking_metric": "test-erm",
            "custom_icdst_evaluator": "icdst-eval",
            "transferqa_order": "to",
            "transferqa_none_ratio": "tnr",
        }

        if mode is None:
            mode = self.train_test_setting

        configs_of_interest = [
            # "input_format",
            "input_context_format",
        ]
        if self.use_incontext_examples:
            configs_of_interest += [
                "example_topk",
                "retrieval_corpus_context",
                "train_example_ranking_metric",
                "dev_example_ranking_metric", 
                "test_example_ranking_metric"
            ]
            
        if self.train_example_ranking_metric == "custom_icdst" or self.dev_example_ranking_metric =="custom_icdst" or self.test_example_ranking_metric =="custom_icdst": 
            configs_of_interest += ["custom_icdst_evaluator"]

        if verbose:
            configs_of_interest = ["input_format"] + configs_of_interest
            if self.input_format == "transferqa" and self.use_incontext_examples:
                configs_of_interest += ["transferqa_order", "transferqa_none_ratio"]

        config_string = "_".join(
            [
                f"{config_abbreviation_mapping[cfg]}:{getattr(self, cfg)}"
                for cfg in configs_of_interest
            ]
        )

        return config_string

    def init_handler(self, m):
        init_method = {
            'MultiWOZ_2.4': self._MultiWOZ_init,
            'MultiWOZ_2.1': self._MultiWOZ_init,
            'SGD': self._SGD_init,
        }
        init_method[m]()
        
        # tests for incompatible hyperparameters
        assert not (
            self.debug == self.small == True
        ), "Only one of --debug or --small should be set to true"

        if self.input_format == "transferqa" and self.transferqa_detailed_format == "simple": 
            assert self.transferqa_order == "aligned"

    def set_result_path(self):
        curr_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        self.result_path = f"results/{curr_time}_{self.model_name}_{self.dataset}_ep{self.epochs}_seed{self.seed}_lr{self.learning_rate}"

    def _MultiWOZ_init(self):
        self.data_path = os.path.join(DATA_DIR, f'{self.dataset}/lifelong')
        self.valid_domains = MWOZ_VALID_DOMAINS

    def _SGD_init(self):
        self.data_path = os.path.join(DATA_DIR, "dstc8-schema-guided-dialogue/lifelong_cpt")
        self.valid_domains = SGD_DOMAINS_OF_INTEREST

    def save(self, save_path=None):
        Path(self.result_path).mkdir(parents=True, exist_ok=True)
        if save_path is None:
            if self.upperbound_idx is not None:
                save_path = (
                    Path(self.result_path) / f"config_ubidx{self.upperbound_idx}.json"
                )
            else:
                save_path = Path(self.result_path) / "config.json"

        if self.train_test_setting != "test": 
            with open(save_path, "w") as f:
                json.dump(vars(self), f, indent=4, sort_keys=True)

    def __str__(self):
        return json.dumps(vars(self), indent=4, sort_keys=True)
