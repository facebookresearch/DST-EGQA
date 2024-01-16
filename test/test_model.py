# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import json
import os
import re

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
)

from egqa import (
    DOMAIN_ORDERS,
    MWOZ_SLOTS,
    Config,
    CL_Dataset,
    Seq2SeqDST_model,
    flatten_nested_list,
    get_ingredients_list,
    load_raw_dataset,
)

MODEL_NAME = "t5-base"
parser = argparse.ArgumentParser()
parser = Config.add_training_specific_args(parser)
parser = Config.add_model_specific_args(parser)
parser = Config.add_data_specific_args(parser)
args = parser.parse_args()
global_param = Config(args)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MULTIWOZ_DATASET_PROCESSOR = CL_Dataset(global_param, tokenizer=TOKENIZER)


def load_all_data():
    """Get all samples, including train, valid, and test sets

    Returns:
        _type_: _description_
    """
    DOMAINS = DOMAIN_ORDERS['default']
    train_data = load_raw_dataset(
        global_param.data_path, domains=DOMAINS, data_type='train'
    )
    dev_data = load_raw_dataset(
        global_param.data_path, domains=DOMAINS, data_type='dev'
    )
    test_data = load_raw_dataset(
        global_param.data_path, domains=DOMAINS, data_type='test'
    )

    all_data = train_data + dev_data + test_data

    all_dialogue_ingredients = [get_ingredients_list(d) for d in all_data]
    all_samples = MULTIWOZ_DATASET_PROCESSOR.form_dialogue_samples(
        all_dialogue_ingredients=all_dialogue_ingredients
    )

    return all_samples


def test_same_model():
    automodel = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    t5model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    automodel_state_dicts = automodel.state_dict()
    t5model_state_dicts = t5model.state_dict()
    assert automodel_state_dicts.keys() == t5model_state_dicts.keys()

    for state_key in t5model_state_dicts.keys():
        assert torch.equal(
            t5model_state_dicts[state_key], automodel_state_dicts[state_key]
        )


def test_same_tokenizer():
    """make sure that the autotokenizer returns an expected tokenizer"""
    autotokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t5tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    t5tokenizer_fast = T5TokenizerFast.from_pretrained(MODEL_NAME)

    # check that the class name is the same
    # t5fast and t5 (original) gets similar outputs: https://github.com/huggingface/transformers/issues/16334
    assert (
        autotokenizer.__class__.__name__ == t5tokenizer.__class__.__name__
        or autotokenizer.__class__.__name__ == t5tokenizer_fast.__class__.__name__
    )


def test_hf_generate():
    """test that all the encoded output example is shorter than the max generation length set in the config file"""

    global_param.input_format = "simpletod"

    all_samples = load_all_data()
    flattened_all_samples = flatten_nested_list(all_samples)
    data_loader = DataLoader(
        flattened_all_samples,
        batch_size=64,
        shuffle=False,
        collate_fn=MULTIWOZ_DATASET_PROCESSOR.collate_fn(),
        num_workers=16,
    )

    for example in tqdm(data_loader):
        encoded_output_length = example['decoder_output'].shape[-1]
        assert encoded_output_length < global_param.max_new_tokens, (
            encoded_output_length,
            global_param.max_new_tokens,
            example,
        )


def test_encode_decode():
    """Test that the encode -> decode output results in the same text as the input"""

    global_param.input_format = "simpletod"

    all_samples = load_all_data()
    flattened_all_samples = flatten_nested_list(all_samples)

    all_slot_labels = [sample["output_text"] for sample in flattened_all_samples]

    dataloader = DataLoader(
        all_slot_labels, batch_size=16, shuffle=False, num_workers=16
    )
    for batch in tqdm(dataloader):
        encode_decode_results = TOKENIZER.batch_decode(
            TOKENIZER(batch)['input_ids'], skip_special_tokens=True
        )

        for orig, res in zip(batch, encode_decode_results):
            # skip some exceptions
            if not orig.isascii():
                continue
            assert re.sub(" ", "", orig) == re.sub(
                " ", "", res
            ), f"orig: {orig}\nres:{res}"


if __name__ == "__main__":
    test_same_model()
    test_hf_generate()
    test_same_tokenizer()
    test_encode_decode()
