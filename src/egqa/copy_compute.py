# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# analyze how many cases are instances of copying from the retrieved example. 
from egqa import (
    DOMAIN_ORDERS,
    load_raw_dataset,
    flatten_nested_list,
    create_belief_state_dictionary,
    get_ingredients_list,
    CL_Dataset, 
    Config, 
    Seq2SeqDST_model
)
from collections import defaultdict
import argparse
import json 
import os 
from loguru import logger 

parser = argparse.ArgumentParser()
parser = Config.add_training_specific_args(parser)
parser = Config.add_model_specific_args(parser)
parser = Config.add_data_specific_args(parser)
args = parser.parse_args()
global_param = Config(args)

current_dst_model = Seq2SeqDST_model(global_param)
multiwoz_dataset_processor = CL_Dataset(
    global_param, tokenizer=current_dst_model.tokenizer
)

class CopyCounter: 
    pass 

def compute_copy():

    total_slot_level_copy = 0
    total_slot_level_copy_wo_none = 0
    total_slot_ct = 0
    total_slot_wo_none_ct = 0

    total_jga_level_copy = 0
    total_jga_ct = 0
    data_path = os.path.join(os.environ['DATA_DIR'], "MultiWOZ_2.1/", "lifelong")

    # load all train_data
    train_data = load_raw_dataset(
        data_path=data_path, domains=DOMAIN_ORDERS['default'], data_type="train"
    )

    examples_ingredients = [get_ingredients_list(dialogue) for dialogue in train_data]
    examples_flat_ingredients = flatten_nested_list(examples_ingredients)
    id2examples = {ingred['turn_id']: ingred for ingred in examples_flat_ingredients}

    test_data = load_raw_dataset(
        data_path=data_path, domains=DOMAIN_ORDERS['default'], data_type="test"
    )
    target_ingredients = [get_ingredients_list(dialogue) for dialogue in test_data]
    target_flat_ingredients = flatten_nested_list(target_ingredients)
    id2targets = {ingred['turn_id']: ingred for ingred in target_flat_ingredients}

    alignment = ["mixed", "aligned", "fixed:0", "fixed:33", "fixed:50", "fixed:66", "random"]
    retrieval = ["bm25", "all"]
    splits = ["train", "dev", "test"]
    
    ingredients_by_config = {}
    multiwoz_dataset_processor.config.small = True 
    multiwoz_dataset_processor.config.use_incontext_examples = True 
    
    columns = ["domain", "retrieval", "split", "alignment", "jga", "st-st (same)", "none-none", "st-none", "none-st", "st-st (diff)"]
    print(",".join(columns))
    
    show_sample = False
    
    for curr_domain in DOMAIN_ORDERS["default"]: 
        ingredients_by_config[curr_domain] = {}
        if show_sample: 
            retrieval = ["bm25"]
        for ret in retrieval: 
            ingredients_by_config[curr_domain][ret] = defaultdict(dict)
            multiwoz_dataset_processor.config.train_example_ranking_metric = ret 
            multiwoz_dataset_processor.config.dev_example_ranking_metric = ret 
            multiwoz_dataset_processor.config.test_example_ranking_metric = ret 
            if show_sample: 
                splits = ["test"]
            for data_split in splits: 
                for al in alignment: 
                    multiwoz_dataset_processor.global_alignment_ct = 0
                    multiwoz_dataset_processor.global_total = 1e-10  # prevent division by zero at the beginning
                    
                    multiwoz_dataset_processor.config.transferqa_order = al 
                                    
                    paired_dialogue_ingredients = multiwoz_dataset_processor.prepare_all_paired_dialogue_ingredients([curr_domain], [curr_domain], [curr_domain], data_split)
                    turn_ingredients = flatten_nested_list(paired_dialogue_ingredients)
                    ingredients_by_config[curr_domain][ret][data_split][al] = turn_ingredients
                    
                    # same: st-st (same), none-none 
                    total_slot_none_none_copy_ct = 0                    
                    total_slot_st_st_copy_ct = 0 
                    
                    # diff: none - st, st - none, st-st (diff)
                    total_slot_st_st_diff_ct = 0 
                    total_slot_none_st_diff_ct = 0 
                    total_slot_st_none_diff_ct =0 
                
                    total_slot_copy_ct = 0
                    total_slot_ct = 0
                    
                    total_jga_level_copy_ct = 0
                    total_jga_ct = 0
                    
                    for turn_ingred in turn_ingredients: 
                        target_gold_belief_states = turn_ingred["ordered_full_belief_states"]
                        example_gold_belief_states = turn_ingred["retrieved_example_ingredients"][0]["ordered_full_belief_states"]
                        
                        if show_sample: 
                            print(al)
                            print("tgt\tex")
                            for tgt_bs, ex_bs in zip(target_gold_belief_states, example_gold_belief_states): 
                                print(tgt_bs, ex_bs)
                            break 
                        
                        total_jga_ct += 1 
                        if al == "mixed": 
                            return 
                        if [bs[1] for bs in target_gold_belief_states] == [bs[1] for bs in example_gold_belief_states]: 
                            total_jga_level_copy_ct += 1 
                            
                        for tgt_bs, ex_bs in zip(target_gold_belief_states, example_gold_belief_states): 
                            total_slot_ct += 1
                            
                            if tgt_bs[1] == ex_bs[1]: 
                                total_slot_copy_ct += 1 
                                
                                if tgt_bs[1] != "none": 
                                    total_slot_st_st_copy_ct += 1 
                                else: 
                                    total_slot_none_none_copy_ct += 1 
                            else: 
                                if tgt_bs[1] != "none" and ex_bs[1] != "none": 
                                    total_slot_st_st_diff_ct += 1 
                                elif tgt_bs[1] != "none" and ex_bs[1] =="none": 
                                    total_slot_st_none_diff_ct += 1 
                                elif tgt_bs[1] == "none" and ex_bs[1] !="none": 
                                    total_slot_none_st_diff_ct += 1 
                                else: 
                                    logger.error(f"Unaccounted cases: {tgt_bs} vs {ex_bs}")


                    if show_sample: 
                        continue 
                    st_st_diff_pct = total_slot_st_st_diff_ct / total_slot_ct * 100
                    st_none_pct = total_slot_st_none_diff_ct / total_slot_ct * 100 
                    none_st_pct = total_slot_none_st_diff_ct / total_slot_ct * 100 
                    st_st_same_pct = total_slot_st_st_copy_ct / total_slot_ct * 100
                    none_none_same_pct = total_slot_none_none_copy_ct / total_slot_ct * 100  
                    
                    jga_copy_pct = total_jga_level_copy_ct / total_jga_ct * 100
                    
                    # columns = ["domain", "retrieval", "split", "alignment", "jga", "st-st (same)", "none-none", "st-none", "none-st", "st-st (diff)"]                    
                    data_columns = [curr_domain, ret, data_split, al, jga_copy_pct, st_st_same_pct, none_none_same_pct, st_none_pct, none_st_pct, st_st_diff_pct]
                    data_columns = [f"{dc:.2f}%" if isinstance(dc, float) else dc for dc in data_columns]
                    print(",".join(data_columns))
                
                if show_sample: 
                    print("\n\n")
                    continue 
                
    if show_sample: 
        return 
                           
                    
    # assert that when alignment is the only difference, the belief states for the examples and the belief states for the targets are the same 
    for curr_domain in DOMAIN_ORDERS["default"]: 
        for ret in retrieval: 
            for data_split in splits: 
                _al = alignment[0]
                turn_ingredients = ingredients_by_config[curr_domain][ret][data_split][_al] 
                
                for idx in range(len(turn_ingredients)): 
                    target_gold_belief_states = sorted(turn_ingredients[idx]["ordered_full_belief_states"])
                    example_gold_belief_states = sorted(turn_ingredients[idx]["retrieved_example_ingredients"][0]["ordered_full_belief_states"])
                    
                    for al in alignment[1:]: 
                        compare_target_gold_belief_states = ingredients_by_config[curr_domain][ret][data_split][al][idx]["ordered_full_belief_states"]
                        compare_example_gold_belief_states= ingredients_by_config[curr_domain][ret][data_split][al][idx]["retrieved_example_ingredients"][0]["ordered_full_belief_states"]
                                                
                        assert target_gold_belief_states == sorted(compare_target_gold_belief_states), (target_gold_belief_states, compare_target_gold_belief_states)
                        assert example_gold_belief_states == sorted(compare_example_gold_belief_states), (example_gold_belief_states, compare_example_gold_belief_states)



def compute_copy_from_pred(pred_fn):
    
    total_slot_level_copy = 0
    total_slot_level_copy_wo_none = 0
    total_slot_ct = 0
    total_slot_wo_none_ct = 0

    total_jga_level_copy = 0
    total_jga_ct = 0
    
    with open(pred_fn) as f: 
        predictions = json.load(pred_fn)

    for trained_last_domain, trained_last_domain_predictions in predictions.items():

        if trained_last_domain_predictions is None:
            continue

        for test_domain, test_domain_preds in trained_last_domain_predictions.items():

            for pred in test_domain_preds:
                gold = pred['gold']

                if (
                    'example_turn_ids_scores' in pred
                    and pred['example_turn_ids_scores']
                ):
                    example_turn_id = list(pred['example_turn_ids_scores'].keys())[0]
                    example_ingred = id2examples[example_turn_id]
                    example_belief_state = example_ingred['current_belief_state']
                    example_bs_dict = create_belief_state_dictionary(
                        example_belief_state
                    )
                else:
                    example_bs_dict = {}

                slot_level_copy = 0
                for slot_key_value in gold:
                    splits = slot_key_value.split()
                    slot_key = splits[0]
                    slot_value = ' '.join(splits[1:])
                    if slot_value == example_bs_dict.get(slot_key, 'none'):
                        slot_level_copy += 1
                    if (
                        slot_value != "none"
                        and example_bs_dict.get(slot_key, 'none') != "none"
                    ):
                        total_slot_wo_none_ct += 1
                        if slot_value == example_bs_dict.get(slot_key, 'none'):
                            total_slot_level_copy_wo_none += 1

                total_slot_level_copy += slot_level_copy
                total_slot_ct += len(gold)

                if slot_level_copy == len(gold):
                    total_jga_level_copy += 1
                total_jga_ct += 1

        break

    slot_level_copy_pct = total_slot_level_copy / total_slot_ct
    slot_level_copy_wo_none_pct = total_slot_level_copy_wo_none / total_slot_wo_none_ct
    jga_level_copy_pct = total_jga_level_copy / total_jga_ct

    logger.info(
        f"slot level copy: {total_slot_level_copy}/{total_slot_ct}={slot_level_copy_pct*100:.3f}%"
    )
    logger.info(
        f"slot level copy without nones: {total_slot_level_copy_wo_none}/{total_slot_wo_none_ct}={slot_level_copy_wo_none_pct*100:.3f}%"
    )
    logger.info(
        f"jga level copy: {total_jga_level_copy}/{total_jga_ct}={jga_level_copy_pct*100:.3f}%"
    )

    return

if __name__ == "__main__": 
    compute_copy()