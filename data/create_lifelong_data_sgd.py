# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os
from mtc.utils.dataloader_utils import read_json, write_json
from mtc.dataloader.constants.sgd import SGD_DOMAINS_OF_INTEREST
from collections import defaultdict
import glob
from loguru import logger
from tabulate import tabulate
import numpy as np
import random

DATA_DIR= os.environ.get("DATA_DIR", "")

def formalize_schema():
    schema = read_json(args.input_dir+'train/schema.json')
    schema_all = {}
    for domain in schema:
        state_slots = set()
        for intent in domain[ "intents" ]:
            state_slots.update([s.lower().strip() for s in intent[ "required_slots" ]])
            state_slots.update([s.lower().strip() for s in intent[ "optional_slots" ]])

        domain_name = domain['service_name'].lower().strip()
        slot_all = []
        for slot in domain['slots']:
            slot_name = slot['name'].lower().strip()
            if slot['name'] in state_slots:
                slot_all.append(domain_name+'-'+slot_name)
        schema_all[domain_name] = sorted(slot_all)
    return schema_all

def read_json_list(file_list):
    per_data = []
    for per_file in file_list:
        if 'schema.json' not in per_file:
            per_data += read_json(per_file)
        else:
            logger.debug('ignore schema file')
    return per_data

def main():
    schema = formalize_schema()
    sgd_data_dir = os.path.join(DATA_DIR, "dstc8-schema-guided-dialogue/")
    
    data_by_domain = defaultdict(list)
    for split in ["train", "dev", "test"]: 
        files_list = glob.glob(f"{sgd_data_dir}{split}/*.json")
        per_data = read_json_list(files_list)
        for per_dialog in per_data:
            turns, get_domain, oov_domain = process_dialog(per_dialog, schema)
            
            if get_domain == '':
                continue
            
            # keep only single service conversations
            if len(get_domain.split("-")) >=2: 
                continue

            domains = get_domain
            per_dialog_normal = {'domains':domains, 'dial_id': per_dialog["dialogue_id"], 'turns':turns}
            data_by_domain[domains].append(per_dialog_normal)
                
    n_domains = len(data_by_domain.keys()) 
    assert n_domains == 44 # same as CPT setup. 
    logger.debug(f"# SINGLE DOMAIN: {n_domains}")

    table = []
    total_train, total_dev, total_test = 0, 0, 0 
    for lifelong_domain in data_by_domain.keys():
        
        domain_data = data_by_domain[lifelong_domain]
        random.shuffle(domain_data)
        train_data, dev_data, test_data = np.split(domain_data, [int(len(domain_data)*0.7), int(len(domain_data)*0.8)])
        
        table.append({"dom":lifelong_domain, "train":len(train_data), "valid":len(dev_data), "test":len(test_data)})
        total_train += len(train_data)
        total_dev += len(dev_data)
        total_test += len(test_data)
        
        if not os.path.isdir(args.output_dir): 
            os.makedirs(args.output_dir, exist_ok=True)
            
        write_json(train_data.tolist(), args.output_dir+'/'+lifelong_domain+'_train.json')
        write_json(dev_data.tolist(), args.output_dir+'/'+lifelong_domain+'_dev.json')
        write_json(test_data.tolist(), args.output_dir+'/'+lifelong_domain+'_test.json')

    table.append({"dom": "SGD TOTAL", "train": total_train, "valid": total_dev, "test": total_test})
    print(tabulate(table, headers="keys"))
    
    return 

def process_dialog(dialog, schema):
    temp_turns, get_domain, oov_domain = [], [], False
    dialogue_id = dialog['dialogue_id']
    for per_turn_idx, per_turn in enumerate(dialog['turns']):
        if per_turn_idx % 2 == 0:
            assert per_turn['speaker'] == 'USER'
            user_utterance = per_turn['utterance']
            if per_turn_idx == 0:
                system_utterance = ''
            else:
                system_utterance = dialog['turns'][per_turn_idx-1]['utterance']

            belief_state = []
            for per_frames in per_turn['frames']:
                domain = per_frames['service'].lower().strip()
                get_domain.append(domain)
                for s, v in per_frames['state']['slot_values'].items():
                    s, v = s.lower().strip(), v[0].lower().strip()
                    if v == 'none':
                        continue
                    # if domain not in schema.keys():
                    #     oov_domain = True
                    #     continue

                    # ds = domain + '-' + s
                    belief_state.append([s, v])
                    
                    # if ds in schema[domain]:
                        # belief_state.append([ds, v])
                    # else:
                        # print('1')

            temp_belief_state = sorted(belief_state, key=lambda x:x[0])

            temp_turns.append({'user_utterance':user_utterance,
                               'system_utterance': system_utterance,
                               'belief_state': temp_belief_state,
                               'turn_id': f"{dialogue_id}-{per_turn_idx}"
                               })
    return temp_turns, '-'.join(sorted(list(set(get_domain)))), oov_domain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=os.path.join(DATA_DIR, 'dstc8-schema-guided-dialogue/'))
    parser.add_argument("--output_dir", type=str, default=os.path.join(DATA_DIR, 'dstc8-schema-guided-dialogue/lifelong_cpt'))
    args = parser.parse_args()
    main()
