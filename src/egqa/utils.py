# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import codecs
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger

from egqa.data_utils.mwoz_constants import (
    MWOZ_SLOTS,
    MWOZ_NAMED_ENTITY_SLOTS,
    MWOZ_SLOT_VAL_CONVERSION, 
    MWOZ_VALID_DOMAINS
)

from egqa.data_utils.general_constants import EXAMPLE_TAG, SYSTEM_TAG, USER_TAG

from egqa.data_utils.slots2questions import SLOTS2QUESTIONS
from egqa.data_utils.sgd_constants import COMBINED_SGD_SCHEMA

# LATEST_GITHASH = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
LATEST_GITHASH = "aefb994"

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


# writh json
def write_json(_dataset, _path, intent=4, separate_store=False):
    if separate_store:
        for data_idx, data in enumerate(_dataset):
            write_json(data, _path + '_' + str(data_idx) + '.json', intent)
    else:
        with codecs.open(_path, 'w', 'utf-8') as file_write:
            json.dump(_dataset, file_write, indent=intent)
            file_write.close()


# read json
def read_json(_path, file_num=0):
    if _path.endswith('json'):
        with codecs.open(_path, 'r', 'utf-8') as file_read:
            _dataset = json.load(file_read)
            file_read.close()
    else:
        _dataset = [read_json(_path + '_' + str(i) + '.json') for i in range(file_num)]
    return _dataset


def load_raw_dataset(data_path, domains: List[str], data_type: str):
    """load lifelong training data from json

    Args:
        args (_type_): parameters in parameters.py
        domains (List[str]): list of domains from DOMAINS
        data_type (str): one of [train, dev, test]

    Returns:
        _type_: _description_
    """
    from egqa import read_json

    data = []
    for current_domain in domains:
        data += read_json(
            os.path.join(data_path, current_domain + '_' + data_type + '.json')
        )
    return data


def extract_slot_from_string(
    slots_string: str, valid_domains: List[str], dataset: str = "SGD"
) -> Tuple[List[str]]:
    """
    Either ground truth or generated result should be in the format:
    "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
    and this function would reformat the string into list:
    ["dom--slot_type--slot_val", ... ]
    """

    slots_list = []

    if not slots_string:
        return [], {}, [] 
    
    per_domain_slot_lists = {}
    named_entity_slot_lists = []

    if dataset == "SGD": 
        str_split = slots_string.split(",")
        for slot_ in str_split: 
            if slot_.strip() == "" or len(slot_.strip().split()) < 2: 
                continue 
            slot_type = slot_.strip().split()[0]
            slot_value = " ".join(slot_.split()[1:])
            slots_list.append(f"{slot_type}--{slot_value}")



    if dataset == "MultiWOZ": 

        # # # remove start and ending token if any
        try:
            str_split = slots_string.strip().split()
        except Exception as e:
            logger.error(str(e))
            import pdb

            pdb.set_trace()

        if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
            str_split = str_split[1:]
        if "</bs>" in str_split:
            str_split = str_split[: str_split.index("</bs>")]

        str_split = " ".join(str_split).split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
                    
            # ignore cases without proper format and valid domains or only domains of interest
            if len(slot) > 2 and slot[0] in valid_domains:
                domain = slot[0]
                # handle cases where slot key contains "book"
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])

                # any normalizations
                slot_val = MWOZ_SLOT_VAL_CONVERSION.get(slot_val, slot_val)

                # may be problematic to skip these cases
                # if not slot_val == "dontcare":
                slots_list.append(domain + "--" + slot_type + "--" + slot_val)

                # divide by domains and categorize as named entities
                if domain in per_domain_slot_lists:
                    per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
                else:
                    per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
                if domain + "--" + slot_type in MWOZ_NAMED_ENTITY_SLOTS:
                    named_entity_slot_lists.append(
                        domain + "--" + slot_type + "--" + slot_val
                    )

        # for slot in slots_list:
        #     assert is_proper_slot_format(slot), f"Slot: {slot} is not in proper format."

    return (slots_list, per_domain_slot_lists, named_entity_slot_lists)


def is_proper_slot_format(slot):
    return len(slot.split("--")) == 3


def compute_jga(predicted_ds, golden_ds):
    return int(set(predicted_ds) == set(golden_ds))


def get_atomic_domains(domain_list: List[str]):
    if isinstance(domain_list, str): 
        domain_list = [domain_list]
    atomic_domains = []
    for dom in domain_list:
        atomic_domains += dom.split("-")

    return set(atomic_domains)


def create_belief_state_dictionary(belief_state: List[List[str]]):
    """transform belief state in List[List[slot, slot value]] format to Dict[slot, slot value] format

    Args:
        belief_state (List[List[str]]): original belief state format

    Returns:
        _type_: belief state in dictionary format
    """
    if belief_state and isinstance(belief_state[0], str): 
        dict_belief_state ={} 
        for bs in belief_state: 
            splits = bs.split()
            slot_key, slot_val = splits[0], ' '.join(splits[1:])
            dict_belief_state[slot_key] = slot_val 
    else: 
        dict_belief_state = {bs[0]: bs[1] for bs in belief_state}
    return dict_belief_state


def format_dialogue_history(dialogue_history: List[str]) -> str:
    """Format dialogue history with the speaker tags

    Args:
        dialogue_history (List[str]): list of utterances

    Returns:
        str: single string with utterances combined with corresponding speaker tags
    """

    dialogue_history_strings = []
    if len(dialogue_history) > 2:
        for turn_idx, turn in enumerate(dialogue_history):
            speaker = SYSTEM_TAG if turn_idx % 2 else USER_TAG
            dialogue_history_strings.append(f"{speaker} {turn}")

        return " ".join(dialogue_history_strings)
    elif len(dialogue_history) == 2:
        return f"{SYSTEM_TAG} {dialogue_history[0]} {USER_TAG} {dialogue_history[1]}"
    elif len(dialogue_history) == 1:
        return f"{USER_TAG} {dialogue_history[0]}"


def extract_state_change(
    previous_belief_state: List[List[str]], current_belief_state: List[List[str]]
):
    # three types: insert (none->value), update (value->value), delete (value->none)
    # result: {slot: [state_change_type, slot_value being inserted/updated/deleted]}

    # TODO: write an example
    # create dictionaries out of both belief states
    prev_bs_dict = create_belief_state_dictionary(previous_belief_state)
    curr_bs_dict = create_belief_state_dictionary(current_belief_state)

    state_change = {}
    for prev_bs_slot_key, prev_bs_slot_value in prev_bs_dict.items():
        # if there has been a deletion
        if prev_bs_slot_key not in curr_bs_dict:
            state_change[prev_bs_slot_key] = ["DELETE", prev_bs_slot_value]

    for curr_bs_slot_key, curr_bs_slot_value in curr_bs_dict.items():
        # if there has been an update
        if (
            curr_bs_slot_key in prev_bs_dict
            and curr_bs_slot_value != prev_bs_dict[curr_bs_slot_key]
        ):
            state_change[curr_bs_slot_key] = ["UPDATE", curr_bs_dict[curr_bs_slot_key]]
        # if there has been an insertion
        elif curr_bs_slot_key not in prev_bs_dict:
            state_change[curr_bs_slot_key] = ["INSERT", curr_bs_dict[curr_bs_slot_key]]

    return state_change


def create_dialogue_context(
    per_turn_ingredient: Dict[str, object], context_configuration: str
) -> str:
    # only: [previous DS] [last system utt.] [last user utt.]
    if context_configuration == "icdst":
        if per_turn_ingredient['system_utterance']:
            utterance_list = [
                per_turn_ingredient['system_utterance'],
                per_turn_ingredient['user_utterance'],
            ]
        else:
            assert per_turn_ingredient['dialogue_history'] == []
            utterance_list = [per_turn_ingredient['user_utterance']]
        dialogue_history = format_dialogue_history(utterance_list)
        formatted_belief_states = ", ".join(
            [
                format_simpletod_belief_state(bs)
                for bs in per_turn_ingredient['last_belief_state']
            ]
        )
        context = f"[{formatted_belief_states}] <context> {dialogue_history}"
    # all utterances or last pair of utterances 
    elif context_configuration == "full" or context_configuration =="last":
        if per_turn_ingredient['system_utterance']:
            utterance_list = per_turn_ingredient['dialogue_history'] + [
                per_turn_ingredient['system_utterance'],
                per_turn_ingredient['user_utterance'],
            ]
        else:
            assert per_turn_ingredient['dialogue_history'] == []
            utterance_list = [per_turn_ingredient['user_utterance']]
        if context_configuration == "last": 
            utterance_list = utterance_list[-2:]
        context = format_dialogue_history(utterance_list)        
    else:
        raise NotImplementedError

    return context


def format_simpletod_output_seq(raw_belief_state):
    # e.g. [['restaurant-name', 'pizza hut city centre']]
    # --> restaurant name pizza hut city center, ...

    formatted_and_sorted_bs = sorted(
        [format_simpletod_belief_state(bs) for bs in raw_belief_state],
        key=lambda x: x,
    )
    output_seq = ', '.join(formatted_and_sorted_bs)

    return output_seq


def format_simpletod_belief_state(belief_state: List[str]):
    """
    Format belief states such that
        ["hotel-internet", "yes"] ==> "hotel internet yes"
        ["hotel-book stay", "3"] ==> "hotel stay 3"
    """

    slot_key = normalize_slot_key(belief_state[0])
    slot_value = belief_state[1]
    domain = slot_key.split("-")[0]
    # remove 'book', dashes, and whitespaces
    pieces = slot_key.split("-") + [slot_value]

    formatted_belief_state = " ".join(pieces)

    if domain in MWOZ_VALID_DOMAINS and slot_key not in MWOZ_SLOTS:
        logger.error(f"'{slot_key}' is not one of the expected slots: {MWOZ_SLOTS}")
        raise NotImplementedError

    return formatted_belief_state


def normalize_slot_key(slot_key: str) -> str:
    """Keep dash between domain and slot key and remove 'book'

    e.g.
    "hotel-internet" --> "hotel-internet"
    "hotel-book stay" --> "hotel-stay"

    """

    normalized_slot_key = re.sub("book", "", slot_key).replace(" ", "")

    return normalized_slot_key


def get_filtered_slots2questions(
    atomic_domains: List[str],
    dataset: str
) -> Dict[str, Dict[str, str]]:
    """_summary_

    Args:
        atomic_domains (List[str]): list of atomic domains (e.g. only restaurant, train, etc. No restaurant-train etc.)

    Returns:
        Dict[str, Dict[str, str]]: filtered set of question templates for each atomic domain
    """

    if "MultiWOZ" in dataset: 
        filtered_slots2questions = {
            slot: q_ingred
            for slot, q_ingred in SLOTS2QUESTIONS.items()
            if slot.split("-")[0] in atomic_domains
        }
    elif dataset == "SGD": 
        domain = list(atomic_domains)[0]        
        assert len(atomic_domains) ==1, len(atomic_domains) 
        
        filtered_slots2questions = {
            slot: q_ingred for slot, q_ingred in COMBINED_SGD_SCHEMA[domain].items()
        }
    else: 
        raise NotImplementedError
    
    return filtered_slots2questions


def flatten_nested_list(
    dialogue_samples: List[List[Dict]], target_key: str = None
) -> List[Dict]:
    """unravel a list of lists of dicts to a single list of dicts"""

    # flatten samples
    if target_key:
        flat_samples = [turn for dial in dialogue_samples for turn in dial[target_key]]
    else:
        flat_samples = [turn for dial in dialogue_samples for turn in dial]

    return flat_samples


def sample_transferqa_none_seqs(input_output_seqs, none_seqs_binary_list, none_ratio):
    # keep only a certain ratio to that of non-none slots

    # limit max to the total number of none slots
    n_none_samples = np.sum(none_seqs_binary_list)
    n_non_none_samples = len(input_output_seqs) - n_none_samples

    target_n_none_samples = min(n_none_samples, n_non_none_samples * none_ratio)

    # split between non-none and none answer cases
    non_none_answer_samples = []
    none_answer_samples = []
    for io_seq, is_none in zip(input_output_seqs, none_seqs_binary_list):
        if is_none:
            none_answer_samples.append(io_seq)
        else:
            non_none_answer_samples.append(io_seq)

    assert len(input_output_seqs) == len(none_answer_samples) + len(
        non_none_answer_samples
    )

    if n_none_samples > target_n_none_samples:
        none_answer_samples = random.sample(none_answer_samples, target_n_none_samples)

    return non_none_answer_samples, none_answer_samples


def load_dataset_information(data_path: str, data_type: str, domains: List[str]):
    """Load test set data and relevant information

    Args:
        data_path (str): folder path of the testsets
        domains (List[str]): domains to load

    Returns:
        Dict: dictionary
    """

    datasets = {
        domain: load_raw_dataset(data_path, domains=[domain], data_type=data_type)
        for domain in domains
    }

    n_dialogues = {domain: len(dialogues) for domain, dialogues in datasets.items()}

    turns = {
        domain: [turn for dialogue in dialogues for turn in dialogue['turns']]
        for domain, dialogues in datasets.items()
    }

    n_turns = {domain: len(cases) for domain, cases in turns.items()}

    logger.debug(f"Loaded [{data_type:<10}] dataset information: ")
    logger.debug(f"{'DOMAIN':<30s} {'# DIALOGUES':>10s} {'# TURNS':>10s}")
    for domain in n_dialogues.keys():
        logger.debug(
            f"{domain:<30s} {str(n_dialogues[domain]):>10s} {str(n_turns[domain]):>10s}"
        )

    return {
        "datasets": datasets,
        "n_dialogues": n_dialogues,
        "turns": turns,
        "n_turns": n_turns,
    }


def filter_slots(slots: List[str], domains: List[str]) -> List[str]:
    """Filter out slots if they are not relevant to the current set of domains

    Args:
        slots (List[str]): list of slots
        domains (List[str]): domains to keep

    Returns:
        List[str]: filtered slots
    """

    filtered_slots = []
    for slot in slots:
        if len(slot.split()) < 1:
            if slot != "":
                import pdb

                pdb.set_trace()
            else:
                continue
        # for predictions in format of domain-slotkey slotvalue (transferqa)
        if "-" in slot:
            if slot.split()[0].split("-")[0] in domains:
                filtered_slots.append(slot)
        # for predictions in format of domain slotkey slotvalue (simpletod)
        else:
            if slot.split()[0] in domains:
                filtered_slots.append(slot)

    return filtered_slots


def compute_filtered_jgas(
    prediction_list, dataset:str, eval_domains: List[str], strategy: str
) -> List[int]:
    """Given a model's list of predictions that contain the raw outputs, recompute JGA based on any filters
    to simulate the known / unknown test setting

    Args:
        prediction_list (_type_): _description_
        eval_domains (List[str]): _description_
        strategy (str): _description_

    Returns:
        List[int]: _description_
    """

    jgas = []
    same_diff_ct = {}
    slot_f1 = {} 
    
    for each_pred in prediction_list:
        slot_preds = each_pred['pred']
        if 'label' in each_pred:
            slot_gold = each_pred['label']
        else:
            slot_gold = each_pred['gold']

        if strategy == "transferqa":
            if isinstance(slot_preds, str):
                slot_preds = slot_preds.split(",")
                slot_gold = slot_gold.split(",")

            if dataset != "SGD": 
                filtered_preds = filter_slots(slot_preds, eval_domains)
                # apply same mapping as done for simpletod
                filtered_preds = [
                    MWOZ_SLOT_VAL_CONVERSION.get(pred_slot, pred_slot)
                    for pred_slot in filtered_preds
                ]
                filtered_gold = filter_slots(slot_gold, eval_domains)
                
                pred_bs_dict = create_belief_state_dictionary(filtered_preds)
                gold_bs_dict = create_belief_state_dictionary(filtered_gold)
                
                if pred_bs_dict.keys() != gold_bs_dict.keys(): 
                    import pdb; pdb.set_trace() 
            else: 
                filtered_gold = slot_gold 
                filtered_preds = slot_preds 
        else:
            filtered_preds, _, _ = extract_slot_from_string(slot_preds, eval_domains)
            filtered_gold, _, _ = extract_slot_from_string(slot_gold, eval_domains)

        jga = int(filtered_preds == filtered_gold)
        jgas.append(jga)

    return jgas


def compute_cl_metrics(
    predictions,
    dataset: str, 
    trained_domain_order: List[str],
    strategy: str,
    eval_mode: str = "known",
):

    if dataset=="SGD": 
        datapath = os.path.join(os.environ["DATA_DIR"], "dstc8-schema-guided-dialogue/lifelong_cpt")
    else: 
        datapath = os.path.join(os.environ["DATA_DIR"], "MultiWOZ_2.4/lifelong_cpt")

    testset_info = load_dataset_information(
        datapath, data_type="test", domains=trained_domain_order
    )

    # JGA
    complete_jga_matrix = defaultdict(dict)
    step_jgas = []
    for trained_dom in trained_domain_order:
        predictions_dict = predictions.get(trained_dom, None)
        if predictions_dict is None:
            continue

        # track the CL jga for each step
        cl_jga = []
        # all the domains that the model has been trained with so far
        all_trained_domains = trained_domain_order[
            : trained_domain_order.index(trained_dom) + 1
        ]
        all_trained_atomic_domains = get_atomic_domains(all_trained_domains)
        for pred_dom, prediction_list in predictions_dict.items():

            pred_dom_atomic_domains = get_atomic_domains([pred_dom])
            # only evaluate JGA with the schema of the test domain: filter out predictions to simulate setting of knowing target domains ahead of time
            if eval_mode == "known":
                eval_domains = pred_dom_atomic_domains
            # otherwise assume test domain is not known by asking all question from previously trained domains
            else:
                eval_domains = all_trained_atomic_domains

            # check that the number of predictions made for the current prediction domain is correct
            if len(prediction_list) != len(testset_info['turns'][pred_dom]):
                logger.warning(
                    f"Mismatch for trained dom:{trained_dom} and pred dom: {pred_dom} -- # predictions {len(prediction_list)} vs # test cases: {len(testset_info['turns'][pred_dom])}"
                )

            jgas = compute_filtered_jgas(
                prediction_list, dataset=dataset, eval_domains=eval_domains, strategy=strategy
            )

            if pred_dom in all_trained_domains:
                cl_jga += jgas

            complete_jga_matrix[trained_dom][pred_dom] = (np.mean(jgas), len(jgas))

        if cl_jga:
            # print(f"{trained_dom},{np.mean(cl_jga):.4f},{len(cl_jga)}")
            step_jgas.append(np.mean(cl_jga))

    cl_jgas = {
        domain: step_jga for domain, step_jga in zip(trained_domain_order, step_jgas)
    }
    final_jga = step_jgas[-1]
    final_trained_dom = trained_domain_order[-1]
    average_jga = np.mean([results[0] for pred_dom, results in complete_jga_matrix[final_trained_dom].items()])
    
    print(f"final_jga, {final_jga:.4f}")
    print(f"average_jga, {average_jga:.4f}")
    
    return {
        "cl_jgas": cl_jgas,
        "final_jga": final_jga,
        "average_jga": average_jga,
        "complete_jga_matrix": complete_jga_matrix,
    }


def compute_upperbound_metrics(
    predictions, dataset:str, trained_domains: List[str], strategy: str, eval_mode="known"
):

    UB_JGA = []
    DOMAIN_STEP_JGA = defaultdict(list)

    if dataset=="SGD": 
        datapath = os.path.join(os.environ["DATA_DIR"], "dstc8-schema-guided-dialogue/lifelong_cpt")
    else: 
        datapath = os.path.join(os.environ["DATA_DIR"], "MultiWOZ_2.4/lifelong_cpt")

    testset_info = load_dataset_information(
        datapath, data_type="test", domains=trained_domains
    )

    all_trained_atomic_domains = get_atomic_domains(trained_domains)

    for train_domain in trained_domains:
        prediction_list = predictions[trained_domains[0]][train_domain]
        pred_dom_atomic_domains = get_atomic_domains([train_domain])

        if eval_mode == "known":
            eval_domains = pred_dom_atomic_domains
        else:
            eval_domains = all_trained_atomic_domains

        if len(prediction_list) != len(testset_info['turns'][train_domain]):
            logger.warning(
                f"Mismatch. # predictions: {len(prediction_list)} vs # test cases: {len(testset_info['turns'][train_domain])}"
            )

        jgas = compute_filtered_jgas(
            prediction_list, dataset=dataset, eval_domains=eval_domains, strategy=strategy
        )

        UB_JGA.extend(jgas)
        DOMAIN_STEP_JGA[train_domain].extend(jgas)

    # print(f"Total JGA after multitasking with {trained_domains}: {np.mean(UB_JGA)}")
    # print(f"{trained_domains[-1]},{np.mean(UB_JGA)}")
    per_domain_jgas = {}
    for domain, jgas in DOMAIN_STEP_JGA.items():
        # print(f"{domain}: {np.mean(jgas):.4f}")
        per_domain_jgas[domain] = np.mean(jgas)
    # upperbound_jga = np.mean(UB_JGA)
    upperbound_avg_jga = np.mean([jga for jga in per_domain_jgas.values()])
    print(f"Multitask final JGA: {np.mean(UB_JGA):.4f}")
    print(f"Multitask avg JGA: {upperbound_avg_jga:.4f}")

    return {
        "upperbound_avg_jga": upperbound_avg_jga,
        "trained_domains": trained_domains,
        "per_domain_jgas": per_domain_jgas,
    }


def compute_forward_transfer(
    complete_jga_matrix: Dict[str, Dict[str, float]], trained_domain_order: List[str]
) -> Dict[str, float]:
    """how much training on current domain helps/harms perf. on future domains

    Args:
        complete_jga_matrix (Dict[str, Dict[str, float]]): jga matrix that contains per domain JGA of each trained domains
        trained_domain_order (List[str]): order of domains used for training

    Returns:
        Dict[str, float]: forward transfer values in dictionary format
    """

    modified_fwt = []  # track all a_{i-1, j} where j>i-1
    original_fwt = []  # track only a_{i-1, i}
    for i, trained_dom_i in enumerate(trained_domain_order):
        if trained_dom_i not in complete_jga_matrix:
            logger.warning(
                f"{trained_dom_i} not in jga matrix: {complete_jga_matrix.keys()}"
            )
            continue

        for j, trained_dom_j in enumerate(trained_domain_order):
            # ignore backward transfer
            if j <= i:
                continue

            if trained_dom_j not in complete_jga_matrix[trained_dom_i]:
                logger.warning(
                    f"{trained_dom_j} not in {complete_jga_matrix[trained_dom_i].keys()}"
                )
                continue
            a_ij = complete_jga_matrix[trained_dom_i][trained_dom_j][0]
            if j == i + 1:
                original_fwt.append(a_ij)
            modified_fwt.append(a_ij)

    print(f"original fwt,{np.mean(original_fwt):.4f}")
    print(f"modified fwt,{np.mean(modified_fwt):.4f}")
    fwt_result = {
        "original_fwt": np.mean(original_fwt),
        "modified_fwt": np.mean(modified_fwt),
    }

    return fwt_result


def compute_backward_transfer(
    complete_jga_matrix: Dict[str, Dict[str, float]], trained_domain_order: List[str]
) -> Dict[str, float]:
    """how much training on current domain helps/harms perf. future domains
    mean of a_{T, i} - a_{i,i} where T is the index of the final task

    Args:
        complete_jga_matrix (Dict[str, Dict[str, float]]): jga matrix that contains per domain JGA of each trained domains
        trained_domain_order (List[str]): order of domains used for training

    Returns:
        Dict[str, float]: backward transfer values in dictionary format
    """

    backward_transfer = []
    final_domain = trained_domain_order[-1]
    for trained_dom_i in trained_domain_order[:-1]:
        a_ii = complete_jga_matrix[trained_dom_i][trained_dom_i][0]
        a_Ti = complete_jga_matrix[final_domain][trained_dom_i][0]
        backward_transfer.append(a_Ti - a_ii)

    print(f"original_bwt,{np.mean(backward_transfer):.4f}")
    return {"original_bwt": np.mean(backward_transfer)}
