# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import json
import os
import pickle
import random
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import multiprocess as mpp
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from rank_bm25 import BM25Okapi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from egqa.data_utils.general_constants import * 
from egqa.data_utils.sgd_constants import SGD_DOMAIN_ORDERS, SGD_DOMAINS_OF_INTEREST

from egqa.data_utils.slots2questions import SLOTS2QUESTIONS
from egqa.retrieval.retrieval_utils import (
    get_state_change_similarity_matrix,
    retrieve_top_example_indices_and_scores,
)
from egqa.data_utils.dataloader_utils import align
from egqa.utils import (
    LATEST_GITHASH,
    create_dialogue_context,
    extract_state_change,
    flatten_nested_list,
    format_simpletod_output_seq,
    get_atomic_domains,
    get_filtered_slots2questions,
    load_raw_dataset,
    normalize_slot_key,
    sample_transferqa_none_seqs,
)
from egqa.retrieval.retrieval_db import GPTEmbeddings_DB, SentBERT_DB

DOMAIN_ORDERS = {
    "SGD": SGD_DOMAIN_ORDERS
}

def format_domains_string(domains: List[str]):

    domains_string = "_".join(sorted(domains)) if len(domains) != 10 else "all"

    return domains_string


def get_ingredients_list(dialog: Dict[str, Any]):
    ingredients_list = []
    dialogue_history, last_belief_state = [], []
    for turn_idx, turn in enumerate(dialog['turns']):
        user_utterance = turn["user_utterance"].lower().strip()
        system_utterance = turn["system_utterance"].lower().strip()
        # filter out "none" slot values as they are equivalent to not predicting anything.
        current_belief_state = [
            [normalize_slot_key(bs[0]), bs[1]]
            for bs in turn['belief_state']
            if bs[1].strip() != "none"
        ]
        turn_id = turn['turn_id']

        state_change = extract_state_change(last_belief_state, current_belief_state)

        # create dictionary that contains all the ingredients for formatting the input & output
        ingredients = {
            'dialogue_history': copy.deepcopy(dialogue_history),
            'user_utterance': user_utterance,
            'system_utterance': system_utterance,
            'last_belief_state': last_belief_state,
            'current_belief_state': current_belief_state,
            'state_change': state_change,
            'turn_id': turn_id,
            'domain': dialog['domains'],
        }

        ingredients_list.append(ingredients)

        # update last belief state
        last_belief_state = current_belief_state

        # add utterances to dialogue history
        if not system_utterance:
            # system utterance should only be empty for the first turn
            if turn_idx == 0:
                dialogue_history += [user_utterance]
            else:
                logger.error(
                    "It is not expected for the system utterance to be empty except for the last turn"
                )
                import pdb

                pdb.set_trace()
        else:
            dialogue_history += [system_utterance, user_utterance]

    return ingredients_list


def transform_to_transferqa_context(
    question_ingredient: Dict[str, str], context: str, sample_type: str
):
    # set up question format
    if question_ingredient["values"]:
        q_choices = " ".join(
            [f"{OPTIONS_SEPARATOR} {v}" for v in question_ingredient["values"]]
        )
        question_context = f"{question_ingredient['transferqa']} {q_choices}"
    else:
        question_context = f"{question_ingredient['transferqa']}"

    if sample_type == "target":
        input_seq = f"{TARGET_TAG} {question_context} Context: {context} Answer: "
    elif sample_type =="example": 
        input_seq = f"{EXAMPLE_TAG} {question_context} Context: {context} Answer: "
    elif sample_type =="question_only": 
        input_seq = question_context
    elif sample_type == "no_question": 
        input_seq = f"Context: {context} Answer: "
    else: 
        raise NotImplementedError
        

    return input_seq


def create_transferqa_inputs_outputs(
    turn_ingredient: Dict[str, Any],
    input_context_format: str,
    filtered_slots2questions: Dict[str, str],
    format_type: str,
):

    target_context = create_dialogue_context(
        per_turn_ingredient=turn_ingredient,
        context_configuration=input_context_format,
    )

    transferqa_input_outputs = []

    for slot in turn_ingredient['ordered_full_belief_states']:
        slot_key, slot_value = slot[0], slot[1]
        
        q_ingred = filtered_slots2questions[slot_key]
        input_seq = transform_to_transferqa_context(
            question_ingredient=q_ingred,
            context=target_context,
            sample_type=format_type,
        )
        output_seq = slot_value

        # for examples, add the answer to the sequence
        if format_type == "example":
            input_seq = f"{input_seq}{output_seq}"

        # transferqa_input_outputs.append([input_seq, output_seq, slot])
        transferqa_input_outputs.append(
            {"input_seq": input_seq, "output_seq": output_seq, "slot": slot_key}
        )

    return transferqa_input_outputs




class InvalidChoiceError(Exception):
    """Raise for my specific kind of exception"""


# from https://github.com/facebookresearch/Zero-Shot-DST/blob/main/T5DST/data_loader.py
def transferqa_collate_fn(
    data: Union[List[Dict[str, str]], List[Dict[str, List[str]]]],
    tokenizer: AutoTokenizer,
    max_src_length: int,
    max_tgt_length: int,
):
    batch_data = {}
    for key in data[0]:
        try: 
            batch_data[key] = [d[key] for d in data]    
        except: 
            import pdb; pdb.set_trace() 

    input_batch = tokenizer(
        batch_data["input_text"],
        padding="longest",
        max_length=max_src_length,
        return_tensors="pt",
        verbose=False,
        truncation=True,
    )
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(
        batch_data["output_text"],
        padding="longest",
        max_length=max_tgt_length,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
    )
    # replace the padding id to -100 for cross-entropy
    # output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    output_batch['input_ids'][
        output_batch['input_ids'] == tokenizer.pad_token_id
    ] = -100
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


class CL_Dataset(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data_memory = {}
        self.samples = {}
        self.dataloaders = {}

    def save_samples(self):

        domain = self.current_domain
        for key in self.samples:
            config_string = self.config.get_configuration_string(
                data_split=key, for_sample=True
            )
            dir_path = os.path.join(os.environ["DATA_DIR"], "samples")
            os.makedirs(dir_path, exist_ok=True)
            save_path = os.path.join(
                os.environ["DATA_DIR"],
                "samples",
                f"samples:{LATEST_GITHASH}_{domain}:{key}_{config_string}.json",
            )

            with open(save_path, "w") as f:
                json.dump(self.samples[key], f, indent=4)


    def format_simpletod_input_output_seq(
        self, 
        turn_ingredients: Dict[str, Any],
        input_context_format: str = "full",
    ) -> Tuple[List[str]]:

        target_context = create_dialogue_context(
            per_turn_ingredient=turn_ingredients,
            context_configuration=input_context_format,
        )

        target_input_seq = f"{TARGET_TAG} {target_context} [domain={', '.join(turn_ingredients['domain'].split('-'))}] {DS_TAG} "
        if self.config.domain_not_known: 
            target_input_seq = f"{TARGET_TAG} {target_context} {DS_TAG} "
           
        target_output_seq = format_simpletod_output_seq(
            turn_ingredients['current_belief_state']
        )

        example_context_seqs = []
        for example_ingredient in turn_ingredients.get('retrieved_example_ingredients', []):
            example_context = create_dialogue_context(
                per_turn_ingredient=example_ingredient,
                context_configuration=input_context_format,
            )
            ex_output_seq = format_simpletod_output_seq(
                example_ingredient['current_belief_state']
            )
            example_context_seqs.append(
                f"{EXAMPLE_TAG} {example_context} {DS_TAG} {ex_output_seq}"
            )

        input_seq = f"{' '.join(example_context_seqs)} {target_input_seq}".lstrip()
        output_seq_list = [target_output_seq]
        input_seq_list = [input_seq]

        return input_seq_list, output_seq_list

    def format_transferqa_input_output_seq_simple(
        self, 
        turn_ingredient: Dict[str, Any],
        schema_domains: List[str],
        data_split: str,
    ) -> Tuple[List[str]]:

        transferqa_none_ratio = self.config.transferqa_none_ratio 
        input_context_format = self.config.input_context_format

        raw_belief_state = turn_ingredient['current_belief_state']
        n_non_none_slots = len(raw_belief_state)

        assert schema_domains is not None, f"schema_domains: {schema_domains}"
        assert data_split is not None, f"data_split: {data_split}"

        # get domains to ask questions for
        granular_domains = get_atomic_domains(schema_domains)
        filtered_slots2questions = get_filtered_slots2questions(granular_domains, dataset=self.config.dataset)

        target_ios = create_transferqa_inputs_outputs(
            turn_ingredient=turn_ingredient,
            input_context_format=input_context_format,
            filtered_slots2questions=filtered_slots2questions,
            format_type="no_question",
        )
        
        all_example_ios = []
        for ex_ingred in turn_ingredient.get('retrieved_example_ingredients', []):
            per_example_ios = create_transferqa_inputs_outputs(
                turn_ingredient=ex_ingred,
                input_context_format=input_context_format,
                filtered_slots2questions=filtered_slots2questions,
                format_type="no_question",
            )

            assert (
                len(per_example_ios)
                == len(filtered_slots2questions.keys())
                == len(target_ios)
            )
            all_example_ios.append(per_example_ios)

        target_sample_inputs = [_io["input_seq"] for _io in target_ios]
        target_sample_outputs = [_io["output_seq"] for _io in target_ios]
        slot_order = [_io["slot"] for _io in target_ios]

        none_seqs_binary_list = []  # 1 if none answer else 0 (track for optional sampling)
        input_output_seqs = []
        for idx, slot in enumerate(slot_order):
            
            question = filtered_slots2questions[slot]['transferqa']
            sin, sout = target_sample_inputs[idx], target_sample_outputs[idx]

            # get examples
            example_context_seqs = [
                f"{per_ex_ios[idx]['input_seq']}{per_ex_ios[idx]['output_seq']}" for per_ex_ios in all_example_ios
            ]

            # create full input sequences using examples if any
            full_input_context = [question] + example_context_seqs + [sin]
            input_seq = "\n".join(full_input_context).lstrip()

            seq_tuple = [input_seq, [slot, sout]]
            input_output_seqs.append(seq_tuple)

            if sout == "none":
                none_seqs_binary_list.append(1)
            else:
                none_seqs_binary_list.append(0)

        # check that the number of created samples are correct
        n_none_samples = np.sum(none_seqs_binary_list)
        n_all_samples = len(filtered_slots2questions)
        try:
            assert (
                n_none_samples == n_all_samples - n_non_none_slots
            ), f"{n_none_samples}!= {n_all_samples} - {n_non_none_slots}"
        except:
            import pdb

            pdb.set_trace()
        assert (
            len(input_output_seqs)
            == len(filtered_slots2questions)
            == len(none_seqs_binary_list)
        )

        # for training, optionally sample "none" slot instances to speed up training
        if data_split == "train" and transferqa_none_ratio != -1:
            target_n_none_samples = int(
                (len(input_output_seqs) - n_none_samples) * transferqa_none_ratio
            )

            # if there are more none samples than desired, sample
            if n_none_samples > target_n_none_samples:
                non_none_seqs, none_seqs = sample_transferqa_none_seqs(
                    input_output_seqs,
                    none_seqs_binary_list,
                    transferqa_none_ratio,
                )
                input_output_seqs = non_none_seqs + none_seqs
            # otherwise use all

        input_seq_list = [io_seq[0] for io_seq in input_output_seqs]
        output_seq_list = [io_seq[1] for io_seq in input_output_seqs]

        return input_seq_list, output_seq_list

    def format_transferqa_input_output_seq_old(
        self, 
        turn_ingredient: Dict[str, Any],
        schema_domains: List[str],
        data_split: str,
    ) -> Tuple[List[str]]:

        transferqa_none_ratio = self.config.transferqa_none_ratio 
        input_context_format = self.config.input_context_format

        raw_belief_state = turn_ingredient['current_belief_state']
        n_non_none_slots = len(raw_belief_state)

        assert schema_domains is not None, f"schema_domains: {schema_domains}"
        assert data_split is not None, f"data_split: {data_split}"

        # get domains to ask questions for
        granular_domains = get_atomic_domains(schema_domains)
        filtered_slots2questions = get_filtered_slots2questions(granular_domains, dataset=self.config.dataset)

        target_ios = create_transferqa_inputs_outputs(
            turn_ingredient=turn_ingredient,
            input_context_format=input_context_format,
            filtered_slots2questions=filtered_slots2questions,
            format_type="target",
        )
        
        all_example_ios = []
        for ex_ingred in turn_ingredient.get('retrieved_example_ingredients', []):
            per_example_ios = create_transferqa_inputs_outputs(
                turn_ingredient=ex_ingred,
                input_context_format=input_context_format,
                filtered_slots2questions=filtered_slots2questions,
                format_type="example",
            )

            assert (
                len(per_example_ios)
                == len(filtered_slots2questions.keys())
                == len(target_ios)
            )
            all_example_ios.append(per_example_ios)

        target_sample_inputs = [_io["input_seq"] for _io in target_ios]
        target_sample_outputs = [_io["output_seq"] for _io in target_ios]
        slot_order = [_io["slot"] for _io in target_ios]

        none_seqs_binary_list = []  # 1 if none answer else 0 (track for optional sampling)
        input_output_seqs = []
        for idx, slot in enumerate(slot_order):

            sin, sout = target_sample_inputs[idx], target_sample_outputs[idx]

            # get examples
            example_context_seqs = [
                per_ex_ios[idx]["input_seq"] for per_ex_ios in all_example_ios
            ]

            # create full input sequences using examples if any
            input_seq = f"{' '.join(example_context_seqs)} {sin}".lstrip()

            seq_tuple = [input_seq, [slot, sout]]
            input_output_seqs.append(seq_tuple)

            if sout == "none":
                none_seqs_binary_list.append(1)
            else:
                none_seqs_binary_list.append(0)

        # check that the number of created samples are correct
        n_none_samples = np.sum(none_seqs_binary_list)
        n_all_samples = len(filtered_slots2questions)
        try:
            assert (
                n_none_samples == n_all_samples - n_non_none_slots
            ), f"{n_none_samples}!= {n_all_samples} - {n_non_none_slots}"
        except:
            import pdb

            pdb.set_trace()
        assert (
            len(input_output_seqs)
            == len(filtered_slots2questions)
            == len(none_seqs_binary_list)
        )

        # for training, optionally sample "none" slot instances to speed up training
        if data_split == "train" and transferqa_none_ratio != -1:
            target_n_none_samples = int(
                (len(input_output_seqs) - n_none_samples) * transferqa_none_ratio
            )

            # if there are more none samples than desired, sample
            if n_none_samples > target_n_none_samples:
                non_none_seqs, none_seqs = sample_transferqa_none_seqs(
                    input_output_seqs,
                    none_seqs_binary_list,
                    transferqa_none_ratio,
                )
                input_output_seqs = non_none_seqs + none_seqs
            # otherwise use all

        input_seq_list = [io_seq[0] for io_seq in input_output_seqs]
        output_seq_list = [io_seq[1] for io_seq in input_output_seqs]

        return input_seq_list, output_seq_list

    def format_input_output_seq(
        self,
        turn_ingredients: Dict[str, str],
        input_format: str = "transferqa",
        schema_domains: List[str] = None,
        data_split: str = None,
    ) -> Tuple[List[str], List[str]]:
        """Main function for formatting the input and output sequences based on the informations from the current turn

        Args:
            ingredients (Dict[str, str]): information about the current turn provided as a dictionary
            input_format (str): one of [simpletod, transferqa]
            schema_domains (List[str]): list of atomic domains. only needed for transferqa
            data_split: one of [train, test, example].

        Returns:
            Tuple[List[str], List[str]]: a tuple of input and output sequences, either as a single string or a list of strings depending on specified format
        """

        # format input sequence
        # SimpleTOD appraoch
        if input_format == "simpletod":
            input_seq_list, output_seq_list = self.format_simpletod_input_output_seq(
                turn_ingredients=turn_ingredients,
                input_context_format=self.config.input_context_format,
            )

        # low priority TODO: choice indexing vs generating choice for categorical values
        elif input_format == "transferqa":
            
            if self.config.transferqa_detailed_format == "simple": 
                format_transferqa_input_output_seq = self.format_transferqa_input_output_seq_simple
            else:
                format_transferqa_input_output_seq = self.format_transferqa_input_output_seq_old
                
            input_seq_list, output_seq_list = format_transferqa_input_output_seq(
                turn_ingredient=turn_ingredients,
                schema_domains=schema_domains,
                data_split=data_split,
            )
        else:
            raise InvalidChoiceError

        assert len(input_seq_list) == len(output_seq_list)

        return [input_seq_list, output_seq_list]

    def get_retrieved_ingredients_for_target(
        self, target_ingredients, retrieval_ingredients, data_split: str
    ):

        # scores to use for ranking examples to retrieve
        scores_for_ranking_map = {
            "train": self.config.train_example_ranking_metric,
            "dev": self.config.dev_example_ranking_metric,
            "test": self.config.test_example_ranking_metric,
        }

        # retrieve random examples
        if (
            (data_split == "train" and self.config.train_example_ranking_metric == "random")
            or (data_split == "dev" and self.config.dev_example_ranking_metric == "random")
            or (data_split == "test" and self.config.test_example_ranking_metric == "random")
        ):
            # randomly sample indices of turns to use for random examples
            indices = range(
                len(retrieval_ingredients['flattened_retrieval_candidate_ingredients'])
            )
            example_indices = random.sample(indices, self.config.example_topk)
            # format it in the same way as when non-random examples are retrievd
            example_indices_and_scores = [
                {"index": ex_idx}
                for ex_idx in example_indices
            ]
        else:
            target_index = retrieval_ingredients['turn_id2target_idx'][
                target_ingredients['turn_id']
            ]
            state_change_similarity_scores = retrieval_ingredients['similarity_matrix'][
                target_index
            ].tolist()

            query = create_dialogue_context(
                per_turn_ingredient=target_ingredients,
                context_configuration=self.config.retrieval_corpus_context,
            )
            tokenized_query = self.tokenizer.tokenize(query)
            # use bm25 score

            if (
                (
                    data_split == "test"
                    and self.config.test_example_ranking_metric in ["scs-bm25", "bm25"]
                )
                or (
                    data_split == "dev"
                    and self.config.dev_example_ranking_metric in ["scs-bm25", "bm25"]
                )
                or (
                    data_split == "train"
                    and self.config.train_example_ranking_metric in ["scs-bm25", "bm25"]
                )
            ):
                bm25_doc_scores = retrieval_ingredients['bm25'].get_scores(
                    tokenized_query
                )
            else: 
                bm25_doc_scores = [0] * len(state_change_similarity_scores)
                
            if (
                (data_split == "train" and self.config.train_example_ranking_metric == "sentbert")
                or (data_split == "dev" and self.config.dev_example_ranking_metric == "sentbert")
                or (data_split == "test" and self.config.test_example_ranking_metric == "sentbert")
            ):
                sentbert_scores = retrieval_ingredients["sentbert_db"].get_scores(query).tolist()
            else: 
                sentbert_scores = [0] * len(state_change_similarity_scores)
                
            if (
                (data_split == "train" and self.config.train_example_ranking_metric == "icdst")
                or (data_split == "dev" and self.config.dev_example_ranking_metric == "icdst")
                or (data_split == "test" and self.config.test_example_ranking_metric == "icdst")
            ):  
                icdst_scores = retrieval_ingredients["icdst_db"].get_scores(query).tolist()
            else: 
                icdst_scores = [0] * len(state_change_similarity_scores)
                
                
            if (
                (data_split == "train" and self.config.train_example_ranking_metric == "custom_icdst")
                or (data_split == "dev" and self.config.dev_example_ranking_metric == "custom_icdst")
                or (data_split == "test" and self.config.test_example_ranking_metric == "custom_icdst")
            ):  
                custom_icdst_scores = retrieval_ingredients["custom_icdst_db"].get_scores(query).tolist()
            else: 
                custom_icdst_scores = [0] * len(state_change_similarity_scores)
                
            if (
                (data_split == "train" and self.config.train_example_ranking_metric == "gpt")
                or (data_split == "dev" and self.config.dev_example_ranking_metric == "gpt")
                or (data_split == "test" and self.config.test_example_ranking_metric == "gpt")
            ):
                gpt_emb_scores = retrieval_ingredients["gpt_db"].get_scores(query).tolist()
            else: 
                gpt_emb_scores = [0] * len(state_change_similarity_scores)
                
            # import pdb; pdb.set_trace() 
                
            retrieval_scores = {
                "state_change_similarity": state_change_similarity_scores,
                "bm25": bm25_doc_scores, 
                "sentbert": sentbert_scores, 
                "icdst": icdst_scores, 
                "custom_icdst": custom_icdst_scores,
                "gpt": gpt_emb_scores
            }

            # retrieve different examples based on different scoring methods
            example_indices_and_scores = retrieve_top_example_indices_and_scores(
                retrieval_scores = retrieval_scores, 
                topk=self.config.example_topk,
                data_split=data_split,
                scores_for_ranking=scores_for_ranking_map[data_split],
            )

        # map actual turn ids to their similarity scores
        turn_id2scores_dict = {
            retrieval_ingredients['flattened_retrieval_candidate_ingredients'][
                ex_idx_score['index']
            ]['turn_id']: ex_idx_score
            for ex_idx_score in example_indices_and_scores
        }

        retrieved_example_ingredients = [
            retrieval_ingredients['flattened_retrieval_candidate_ingredients'][
                ex_idx['index']
            ]
            for ex_idx in example_indices_and_scores
        ]

        return retrieved_example_ingredients, turn_id2scores_dict

    def pair_dialogue_examples(
        self,
        all_dialogue_ingredients: List[List[Dict]],
        retrieval_ingredients: Dict[str, object] = None,
        data_split: str = "test",
    ):

        all_paired_examples = []
        for dialogue_ingredients in all_dialogue_ingredients:
            dialogue_paired_examples = []
            for turn_ingredients in dialogue_ingredients:

                domain = turn_ingredients['domain']

                turn_id2scores_dict = {}
                retrieved_example_ingredients = []
                # retrieve in-context examples
                if self.config.use_incontext_examples:
                    (
                        retrieved_example_ingredients,
                        turn_id2scores_dict,
                    ) = self.get_retrieved_ingredients_for_target(
                        target_ingredients=turn_ingredients,
                        retrieval_ingredients=retrieval_ingredients,
                        data_split=data_split,
                    )
                turn_ingredients["turn_id2scores_dict"] = turn_id2scores_dict
                turn_ingredients[
                    "retrieved_example_ingredients"
                ] = retrieved_example_ingredients

                dialogue_paired_examples.append(copy.deepcopy(turn_ingredients))
            all_paired_examples.append(copy.deepcopy(dialogue_paired_examples))

        return all_paired_examples

    def shift_nones(
        self, full_belief_states: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Shift all slots with 'none' to the end and the rest to the front.

        Args:
            full_belief_states (List[Tuple[str,str]]): output of partial_belief_states_to_ordered_full_belief_states()

        Returns:
            List[Tuple[str,str]]: same list but slots with 'none' shifted
        """

        nones = [_io for _io in full_belief_states if _io[1] == "none"]
        non_nones = [_io for _io in full_belief_states if _io[1] != "none"]
        full_belief_states = non_nones + nones

        return full_belief_states

    def partial_belief_states_to_ordered_full_belief_states(
        self, partial_belief_states: List[List[str]], ordered_slots: List[str]
    ) -> List[List[str]]:
        """

        Args:
            partial_belief_states (List[List[str]]): _description_ e.g [['restaurant-food', 'portugese']]

        Returns:
            List[List[str]]: same list but with 'none' slots filled in for empty slots among `ordered_slots` e.g. [['restaurant-area', 'none'], ['restaurant-day', 'none'], ['restaurant-food', 'portugese'], ['restaurant-name', 'none'], ['restaurant-people', 'none'], ['restaurant-pricerange', 'none'], ['restaurant-time', 'none']]
        """
        partial_belief_states_dict = {bs[0]: bs[1] for bs in partial_belief_states}

        return [
            [slot, partial_belief_states_dict[slot]]
            if slot in partial_belief_states_dict
            else [slot, 'none']
            for slot in ordered_slots
        ]

    def configure_belief_state_alignment(
        self, turn_ingredients, sorted_slots: List[str], alignment: str
    ) -> Tuple[Any]:

        has_examples = bool(
            turn_ingredients.get('retrieved_example_ingredients', False)
        )
        # get partial belief states and transform to full belief states (set empty slot value to 'none')
        target_belief_state = turn_ingredients['current_belief_state']
        target_full_belief_states = (
            self.partial_belief_states_to_ordered_full_belief_states(
                target_belief_state, sorted_slots
            )
        )

        # alignment only matters if there are examples
        if has_examples:
            all_ex_bs = [
                rei['current_belief_state']
                for rei in turn_ingredients['retrieved_example_ingredients']
            ]
            all_example_full_belief_states = [
                self.partial_belief_states_to_ordered_full_belief_states(
                    ex_belief_state, sorted_slots
                )
                for ex_belief_state in all_ex_bs
            ]
            assert len(target_full_belief_states) == len(
                all_example_full_belief_states[0]
            ), f"Fully formed belief states don't have the same number of slots: \ntarget: {target_full_belief_states} \nexample: {all_example_full_belief_states[0]}"

            # align according to different configurations
            if alignment == "aligned":
                # already aligned by creating full belief states with sorted slots
                pass
            elif alignment == "mixed":
                target_full_belief_states = self.shift_nones(target_full_belief_states)
                all_example_full_belief_states = [
                    self.shift_nones(per_ex_bs)
                    for per_ex_bs in all_example_full_belief_states
                ]
            elif alignment == "random":
                random.shuffle(target_full_belief_states)
                if has_examples:
                    for per_ex_bs in all_example_full_belief_states:
                        # apply the same shuffle for all examples
                        random.seed(47)
                        random.shuffle(per_ex_bs)

            elif "fixed" in alignment:
                self.target_alignment_pct = float(alignment.split(":")[-1]) / 100
                combs, self.global_alignment_ct, self.global_total = align(
                    copy.deepcopy(sorted_slots),
                    copy.deepcopy(sorted_slots),
                    self.target_alignment_pct,
                    self.global_alignment_ct,
                    self.global_total,
                )

                target_order = [key_pair[0] for key_pair in combs]
                example_order = [key_pair[1] for key_pair in combs]

                target_full_belief_states = (
                    self.partial_belief_states_to_ordered_full_belief_states(
                        target_belief_state, target_order
                    )
                )
                all_example_full_belief_states = [
                    self.partial_belief_states_to_ordered_full_belief_states(
                        ex_bs, example_order
                    )
                    for ex_bs in all_ex_bs
                ]

            else:
                raise NotImplementedError
        else:
            all_example_full_belief_states = None

        return target_full_belief_states, all_example_full_belief_states

    def form_dialogue_samples(
        self,
        all_dialogue_ingredients: List[List[Dict]],
        schema_domains: List[str] = None,
        data_split: str = "test",
    ) -> List[List[Dict[str, str]]]:
        """Use the configurations and data to prepare dataset

        Args:
            data (List[Dict]): loaded json data from KPN continual learning set up
            memory (List[Dict]): memory data from previous domains

        Returns:
            List[List[Dict[str, str]]]: input and output sequences grouped by conversations
        """

        # turn_ingredients = flatten_nested_list(all_dialogue_ingredients)

        # initialize variables for target-example alignment
        self.global_alignment_ct = 0
        self.global_total = 1e-10  # prevent division by zero at the beginning

        # import pdb; pdb.set_trace()
        
        all_dialogue_samples = [] 
        for turn_ingredients in all_dialogue_ingredients: 
            turn_samples = []                    
            for per_turn_ingredients in turn_ingredients:
                domain = per_turn_ingredients['domain']
                # format everything as seq2seq generation task
                input_format = self.config.input_format

                # based on the data_split (train or test, sample different number of "none" answer questions)
                input_seq, output_seq = self.format_input_output_seq(
                    per_turn_ingredients,
                    input_format=input_format,
                    schema_domains=schema_domains,
                    data_split=data_split,
                )

                for each_inseq, each_outseq in zip(input_seq, output_seq):

                    # for TransferQA format, there are multiple input outputs for each turn
                    if len(each_outseq) == 2:
                        output_text = each_outseq[1]
                        slot = each_outseq[0]
                    # for others, there is 1:1 mapping for each sample for each turn
                    else:
                        output_text = each_outseq
                        slot = None

                    turn_samples.append(
                        {
                            "input_text": each_inseq,
                            "output_text": output_text,
                            "idx": per_turn_ingredients['turn_id'],
                            "domain": domain,
                            "slot": slot,  # for mapping back the output: None for SimpleTOD
                            "example_turn_ids_scores": per_turn_ingredients.get(
                                'turn_id2scores_dict', {}
                            ),
                        }
                    )
            all_dialogue_samples.append(turn_samples)

        logger.info(f"Total number of {data_split} per-turn samples: {len(flatten_nested_list(all_dialogue_samples))}")
        assert len(all_dialogue_ingredients) == len(all_dialogue_samples)

        return all_dialogue_samples

    def collate_fn(self):

        collate_fn = partial(
            transferqa_collate_fn,
            tokenizer=self.tokenizer,
            max_src_length=self.config.max_source_length,
            max_tgt_length=self.config.max_target_length,
        )

        return collate_fn

    def _transpose_batch(self, samples, batch):
        """

        Args:
            batch (List[Tuple[int, int]]): a list of tuples with dial idx and its length

        Returns:
            _type_:
        """
        batch_turns = []
        num_turns = len(samples[batch[0][0]])
        for per_batch in batch:
            assert num_turns == per_batch[1]

        # iterate through number of turns
        for turn_idx in range(num_turns):
            # batch_sample[0]: dial idx
            # basically retrieving each utterance
            per_turn_batch = [
                samples[batch_sample[0]][turn_idx] for batch_sample in batch
            ]
            batch_turns.append(per_turn_batch)
        return batch_turns

    def sequential_prediction_dataloader(self, samples) -> List[List[Dict]]:
        """Return a mini batch iterator (data loader) that returns dialogue batches.
        Each dialogue batch consists of turn batches, which have the size specified by 'test_batch_size' in parameters.
        The turns are grouped by the same turn_idx from dialogues that have the same number of turns.

        Yields:
            (List[List[Dict]]): dialogue batch ->
        """

        # create tuples with samples by each of their index and number of turns
        all_samples = [[dial_idx, len(dial)] for dial_idx, dial in enumerate(samples)]

        all_batches, batch = [], {}
        for dial in all_samples:
            # create dictionary keys by number of turns
            if dial[1] not in batch.keys():
                batch[dial[1]] = []
            # add entire dialogue to such that batch[len(dial)] = [dialogues with len(dial)]
            batch[dial[1]].append(dial)

            # iterate through dictionary and if any of them have reached the desired batch size,
            # add them to batches and empty out the dictionary
            for bk, bv in batch.items():
                if len(bv) == self.config.test_batch_size:
                    all_batches.append(bv)
                    batch[bk] = []

        # handle any remaining batches that haven't reached the desired batch size after completing iteration
        for bk, bv in batch.items():
            if len(bv) != 0:
                all_batches.append(bv)

        collate_fn = self.collate_fn()
        for batch_dialogue in tqdm(all_batches, total=len(all_batches)):
            # form batches
            yield [
                collate_fn(batch_turn)
                for batch_turn in self._transpose_batch(samples, batch_dialogue)
            ]

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['dev']

    def test_dataloader(self):
        return self.dataloaders['test']

    def _reduce_for_debugging_and_testing(self, samples, data_domains, data_split):

        domain_samples = {
            domain: [sample for sample in samples if domain == sample['domains']]
            for domain in data_domains
        }

        if self.config.debug:
            if data_split != "test":
                # sample only 10 conversations
                samples = samples[:10]
            else:
                samples = []
                for domain in domain_samples.keys():
                    samples += domain_samples[domain][:5]

        if self.config.small:
            if data_split != "test":
                # sample only 50 conversations
                samples = samples[:50]
            else:
                samples = []
                for domain in domain_samples.keys():
                    samples += domain_samples[domain][:10]
        return samples

    def load_retrieval_ingredients(
        self, domains: List[str], data_split: str
    ) -> Dict[str, Any]:
        """Load data ingredients that will be used for in-context examples retrieval

        Args:
            data_domains (List[str]): domains
            data_split (str): train dev test

        Returns:
            Dict[str, Any]: dictionary that contains ingredients for forming retrieval results to use as in-context examples
        """
        
        size = ""
        if self.config.debug: 
            size = "debug_"
        if self.config.small: 
            size = "small_"
        domains_string = '_'.join(sorted(domains)) if len(domains) != 10 else "all"
        
        if self.config.use_incontext_examples:

            target_data = load_raw_dataset(
                data_path=self.config.data_path, domains=domains, data_type=data_split
            )

            # always only retrieve from train data
            retrieval_data = load_raw_dataset(
                data_path=self.config.data_path, domains=domains, data_type="train"
            )
                        
            all_corpus =  {}
            for d_split in ["train", "dev", "test"]: 
                raw_data = load_raw_dataset(
                    data_path=self.config.data_path, domains=domains, data_type=d_split
                )
                ingreds = flatten_nested_list([get_ingredients_list(per_dialog) for per_dialog in raw_data])
                all_corpus[d_split] = [create_dialogue_context(per_turn_ingredient, self.config.retrieval_corpus_context) for per_turn_ingredient in ingreds]
            
            cache_config =  f"{self.config.dataset}_{size}domains:{domains_string}_ecf:{self.config.retrieval_corpus_context}"
            sentbert_db = SentBERT_DB(all_corpus, db_name=cache_config)
            icdst_db = SentBERT_DB(all_corpus, db_name=cache_config, model="icdst")
            custom_icdst_db = SentBERT_DB(all_corpus, db_name=cache_config, model="custom_icdst", evaluator=self.config.custom_icdst_evaluator, domain=DOMAIN_ORDERS[self.config.dataset][self.config.domain_order_key][0])
            gpt_db = GPTEmbeddings_DB(all_corpus, db_name=cache_config)

            # restrict retrieval data budget
            if data_split != "train" and self.config.retrieval_data_budget:
                retrieval_data = []
                for domain in domains:
                    retrieval_domain_data = load_raw_dataset(
                        data_path=self.config.data_path,
                        domains=[domain],
                        data_type="train",
                    )
                    retrieval_data += random.sample(
                        retrieval_domain_data, k=self.config.retrieval_data_budget
                    )

            # reduce data size only for debugging and testing
            retrieval_data = self._reduce_for_debugging_and_testing(
                retrieval_data, domains, data_split
            )

            flattened_retrieval_data_turns = flatten_nested_list(
                retrieval_data, target_key="turns"
            )


            # For training
            if data_split == "train":
                examples_db = retrieval_data
                indices_to_score = []
            # For validation and test time
            else:
                
                examples_db = retrieval_data + target_data
                flattened_domain_turns = flatten_nested_list(
                    target_data, target_key="turns"
                )
                retrieval_data_len = len(flattened_retrieval_data_turns)
                test_data_len = len(flattened_domain_turns)
                indices_to_score = list(
                    range(retrieval_data_len, retrieval_data_len + test_data_len)
                )

            retrieval_dialogue_ingredients = [
                get_ingredients_list(per_dialog) for per_dialog in retrieval_data
            ]

            flattened_retrieval_candidate_ingredients = flatten_nested_list(
                retrieval_dialogue_ingredients
            )

            # no need to measure any scores for random retrieval
            if (
                data_split == "dev" and self.config.dev_example_ranking_metric == "random"
            ) or (
                data_split == "test" and self.config.test_example_ranking_metric == "random"
            ):
                similarity_matrix = None
                bm25_obj = None
                turn_id2target_idx = None
            else:
                # calculating similarity matrix requires all items (target + retrieval) in the list
                flattened_example_turns = flatten_nested_list(
                    examples_db, target_key="turns"
                )

                # appropriately set up matrix such that there is only a mapping from target <-> retrieval and no target <-> target (unless at train time).
                # `indices_to_score` contains indices for which we should get similarity scores for at test time.
                # final matrix size is no. target turns X no. retrieval turns
                cache_config_name = f"{self.config.dataset}_{size}domains:{domains_string}_mode:{data_split}"
                
                cache_path = os.path.join(
                    os.environ.get("DATA_DIR"),
                    f"cached_sim_matrix_{cache_config_name}.pkl",
                )

                if os.path.isfile(cache_path):
                    logger.info(f"Loading similarity matrix from: {cache_path}")
                    with open(cache_path, "rb") as f:
                        similarity_matrix = pickle.load(f)
                else:
                    
                    # import pdb; pdb.set_trace()
                    ingredients_list = [get_ingredients_list(per_dialog) for per_dialog in examples_db]
                    flattened_example_turns = flatten_nested_list(ingredients_list)
                    
                    similarity_matrix = get_state_change_similarity_matrix(
                        flattened_turns=flattened_example_turns,
                        indices_to_score=indices_to_score,
                    )

                    with open(cache_path, "wb") as f:
                        pickle.dump(similarity_matrix, f)
                    logger.info(f"Saved similarity matrix to: {cache_path}")

                # map index to turn_id to make it easier to retrieve similarity score for each target
                flattened_target_data_turns = flatten_nested_list(
                    target_data, target_key="turns"
                )
                turn_id2target_idx = {
                    turn['turn_id']: idx
                    for idx, turn in enumerate(flattened_target_data_turns)
                }

                retrieval_corpus = [
                    create_dialogue_context(
                        per_turn_ingredient=per_turn_ingredients,
                        context_configuration=self.config.retrieval_corpus_context,
                    )
                    for per_dialogue in retrieval_dialogue_ingredients
                    for per_turn_ingredients in per_dialogue
                ]
                
                bm25_obj = BM25Okapi(
                    corpus=retrieval_corpus, tokenizer=self.tokenizer.tokenize
                )
                assert len(flattened_retrieval_data_turns) == len(retrieval_corpus)


                
            retrieval_ingredients = {
                "similarity_matrix": similarity_matrix,
                "flattened_retrieval_candidate_ingredients": flattened_retrieval_candidate_ingredients,
                "turn_id2target_idx": turn_id2target_idx,
                "bm25": bm25_obj,
                "sentbert_db": sentbert_db, 
                "icdst_db": icdst_db, 
                "custom_icdst_db": custom_icdst_db,
                "gpt_db": gpt_db, 
                "indices_to_score": indices_to_score,
            }

        else:
            retrieval_ingredients = None
        return retrieval_ingredients

    def form_example_target_paired_ingredients(
        self,
        data_domains: List[str],
        example_domains: List[str],
        data_split: str,
        save_path: str,
    ):

        logger.info(f"Forming paired examples for {save_path}...")

        if example_domains is None:
            example_domains = data_domains

        data_type = data_split
        if data_split == "dev":
            data_type = self.config.validation_target

        target_data = load_raw_dataset(
            data_path=self.config.data_path,
            domains=data_domains,
            data_type=data_type,
        )

        # reduce only for debugging and testing
        target_data = self._reduce_for_debugging_and_testing(
            target_data, data_domains, data_split
        )

        # use in context examples
        retrieval_ingredients = self.load_retrieval_ingredients(
            domains=data_domains, data_split=data_split
        )

        all_dialogue_ingredients = [
            get_ingredients_list(per_dialog) for per_dialog in target_data
        ]

        # pair with examples
        paired_all_dialogue_ingredients = self.pair_dialogue_examples(
            all_dialogue_ingredients=all_dialogue_ingredients,
            retrieval_ingredients=retrieval_ingredients,
            data_split="train" if data_split == "train" else "test",
        )

        return paired_all_dialogue_ingredients

    def prepare_all_paired_dialogue_ingredients(self, data_domains, example_domains, schema_domains, data_split): 
        config_string = self.config.get_configuration_string(data_split)

        domains_string = f"tgt-doms:{format_domains_string(data_domains)}_ex-doms:{format_domains_string(example_domains)}"

        size = ""
        if self.config.debug: 
            size = "debug_"
        if self.config.small: 
            size = "small_"

        save_path = f"hash:{LATEST_GITHASH}_mode:{self.config.dataset}_{data_split}_{size}{domains_string}_{config_string}.json"

        save_path = os.path.join(os.environ["DATA_DIR"], save_path)

        paired_all_dialogue_ingredients = None
        # avoid repeated calculations for full dataset
        if os.path.isfile(save_path):
            # logger.info(f"Loading pre-computed paired examples from: {save_path} ...")
            with open(save_path, "r") as f:
                try: 
                    paired_all_dialogue_ingredients = json.load(f)
                except Exception as e: 
                    logger.error(e)
                    logger.error(f"faulty file: {save_path}")
                    Path(save_path).unlink()
            # logger.info(f"Loaded pre-computed paired examples from: {save_path}")

        if paired_all_dialogue_ingredients is None:
            paired_all_dialogue_ingredients = (
                self.form_example_target_paired_ingredients(
                    data_domains=data_domains,
                    example_domains=example_domains,
                    data_split=data_split,
                    save_path=save_path,
                )
            )
            with open(save_path, "w") as f:
                json.dump(paired_all_dialogue_ingredients, f, indent=4)
            logger.info(f"Saved paired examples to {save_path}.")

        if self.config.small: 
            paired_all_dialogue_ingredients = paired_all_dialogue_ingredients[:10]
    
        if schema_domains:
            atomic_domains = get_atomic_domains(schema_domains)
            filtered_slots2questions = get_filtered_slots2questions(atomic_domains, dataset=self.config.dataset)
            sorted_slots = sorted(list(filtered_slots2questions.keys()))


            turn_ingredients = flatten_nested_list(paired_all_dialogue_ingredients)
            
            for per_turn_ingredients in turn_ingredients: 
                # set up questions alignment between target and example for transferqa
                (
                    ordered_target_full_belief_states,
                    ordered_all_example_full_belief_states,
                ) = self.configure_belief_state_alignment(
                    per_turn_ingredients,
                    sorted_slots,
                    alignment=self.config.transferqa_order,
                )
                per_turn_ingredients[
                    "ordered_full_belief_states"
                ] = ordered_target_full_belief_states
                if ordered_all_example_full_belief_states:
                    for ordered_ex_full_bs, example_ingredients in zip(
                        ordered_all_example_full_belief_states,
                        per_turn_ingredients['retrieved_example_ingredients'],
                    ):
                        example_ingredients[
                            "ordered_full_belief_states"
                        ] = ordered_ex_full_bs
            
        return paired_all_dialogue_ingredients

    def set_data_samples(
        self,
        data_domains: List[str],
        schema_domains: Set[str],
        example_domains: List[str] = None,
        memory_samples: List[Dict[str, object]] = None,
        data_split: str = "test",
    ):

        all_samples = []

        for domain in data_domains:
            
            paired_dialogue_ingredients = self.prepare_all_paired_dialogue_ingredients(
                data_domains=[domain], 
                example_domains=[domain], 
                schema_domains=[domain], 
                data_split=data_split
            )
            
            logger.info(f"Forming dialogue samples for {domain}'s {data_split} set...")

            all_samples += self.form_dialogue_samples(
                all_dialogue_ingredients=paired_dialogue_ingredients,
                schema_domains=[domain],  # known
                data_split="train" if data_split == "train" else "test",
            )
                
        # save as attribute for testing and updating memory
        self.samples[data_split] = copy.deepcopy(all_samples)

        per_turn_samples = flatten_nested_list(all_samples)

        # add memory
        if memory_samples:
            logger.debug(f"Adding {len(memory_samples)} memory samples")
            per_turn_samples += copy.deepcopy(memory_samples)

        return per_turn_samples

    def set_data_loader(
        self,
        data_domains: List[str],
        schema_domains: Set[str],
        example_domains: List[str] = None,
        memory_samples: Dict[str, object] = None,
        data_split: str = "test",
    ):

        flattened_samples = self.set_data_samples(
            data_domains=data_domains,
            schema_domains=schema_domains,
            example_domains=example_domains,
            memory_samples=memory_samples,
            data_split=data_split,
        )

        data_loader = DataLoader(
            flattened_samples,
            batch_size=self.config.batch_size
            if data_split == "train"
            else self.config.eval_batch_size,
            shuffle=True if data_split == "train" else False,
            collate_fn=self.collate_fn(),
            num_workers=16,
        )

        return data_loader

    def configure_all_data_loaders(self, domains, current_domain,  data_split="train"):
        # necessary variables for updating memory
        
        self.current_domain = current_domain
        self.current_domain_idx = domains.index(current_domain)

        ### set domains for train / dev / test
        # load all previous domains for multitasking upperbound
        if self.config.upperbound_idx is not None:
            train_domains = domains[: self.config.upperbound_idx + 1]
            val_domains = train_domains
            test_domains = train_domains

        else:
            train_domains = [current_domain]
            if self.config.cumulate_dev_set:  # KPN setup
                val_domains = domains[: self.current_domain_idx + 1]
            else:
                val_domains = [current_domain]  # AdapterCL setup
            # only load training data for new domain
            # unless upperbound is set, test with all domains for FWT and BWT metrics, and compute JGA for each individual domain
            test_domains = domains[self.current_domain_idx:self.current_domain_idx+2]
            
            # for BWT metric calculation, evaluate on all domains 
            if self.current_domain_idx +1 == len(domains): 
                test_domains = domains 

        train_schema_domains = get_atomic_domains(train_domains)
        assert all(
            [
                schema_dom in self.config.valid_domains
                for schema_dom in train_schema_domains
            ]
        ), f"{train_schema_domains} must have all of its elements be in {self.config.valid_domains}"

        # skip when only interested in testing
        if data_split != "test":
            logger.info(
                f"Train domains: {train_domains}, Train schema domains: {train_schema_domains}"
            )
            
            train_data_memory_samples = flatten_nested_list([
                samples[1] for samples in self.data_memory.items()
            ])
            train_data_loader = self.set_data_loader(
                data_domains=train_domains,
                schema_domains=train_schema_domains,
                memory_samples=train_data_memory_samples,
                data_split="train",
            )

            validation_schema_domains = train_schema_domains
            logger.info(
                f"Validation domains: {val_domains}, Validation schema domains: {validation_schema_domains}"
            )
            dev_data_loader = self.set_data_loader(
                data_domains=val_domains,
                schema_domains=validation_schema_domains,
                memory_samples=[],
                data_split="dev",
            )

            # for debugging by making sure that we can overfit to the training data and get high JGA
            if self.config.validation_target == "train":
                assert self.samples["train"] == self.samples["dev"]
        else:
            train_data_loader = None
            dev_data_loader = None

        # configure for two different settings
        if self.config.upperbound_idx is not None:
            test_schema_domains = train_schema_domains
        if self.config.domain_not_known: 
            test_schema_domains = get_atomic_domains(domains)
        else:
            test_schema_domains = get_atomic_domains(test_domains)

        logger.info(
            f"Test domains: {test_domains}, Test schema domains: {test_schema_domains}"
        )

        test_data_loader = self.set_data_loader(
            data_domains=test_domains,
            schema_domains=test_schema_domains,
            memory_samples=[],
            data_split="test",
        )

        # TODO: later, use separate batch loader for prediction (only makes a difference for models that require sequential predictions for a dialogue)
        # test_data_loader = self.sequential_prediction_dataloader()

        self.dataloaders = {
            "train": train_data_loader,
            "dev": dev_data_loader,
            "test": test_data_loader,
        }

    def set_data_memory(self, memory):
        self.data_memory = memory

    def update_memory(self):

        if self.config.memory_strategy == "random":
            logger.info(f'Update Memory: {self.current_domain}')

            num_current_domain_samples = self.config.memory_num 
            per_turn_samples = flatten_nested_list(self.samples["train"])
            self.data_memory[self.current_domain] = random.sample(
                per_turn_samples, min(num_current_domain_samples, len(per_turn_samples))
            )
            
        elif self.config.memory_strategy == "dialogue": 
            dialogue_samples = random.sample(self.samples["train"], min(self.config.memory_num, len(self.samples["train"])))
            per_turn_samples = flatten_nested_list(dialogue_samples)
            self.data_memory[self.current_domain] = per_turn_samples 