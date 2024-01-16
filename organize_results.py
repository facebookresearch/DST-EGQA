# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json 
from pathlib import Path 
import pandas as pd 
import re
from tabulate import tabulate
from caq.parameters import RESULTS_DIR

results_path = RESULTS_DIR

result_dirs = Path(results_path).glob("*")

# TODO include these into the key results file directly. 
def extract_retrieval_method(fn_name: str, dir_name:str, data_split:str)-> str: 
        
    pattern = f"{data_split}-erm:([a-zA-Z0-9\-]*)"
    search_result = re.search(pattern, fn_name)

    ret_method = "vanilla"
    if search_result:
        ret_method = search_result[1]
    else: 
        retrieval_methods = ["scs-bm25", "oracle", "all", "bm25", "random", "vanilla"]
        for method in retrieval_methods: 
            if f"_{method}_" in dir_name: 
                ret_method = method 
                
    if ret_method == "custom": 
        if "triplet" in fn_name: 
            ret_method = "custom_icdst_triplet"
        elif "emb_sim" in fn_name: 
            ret_method = "custom_icdst_embsim"
        else: 
            ret_method = "custom_icdst"
            
    if ret_method == "all" or ret_method == "oracle": 
        return "scs-bm25"
    return ret_method

def extract_example_count(str_:str) -> str: 
    pattern = f"_ex:([0-9])_"
    search_result = re.search(pattern, str_)
    if search_result: 
        return search_result[1]
    return "0"

def extract_memory_method(str_:str) -> str: 
    
    dialogue_memory_search_pattern = "dialogue_memory[0-9]*"
    if re.search(dialogue_memory_search_pattern, str_): 
        memory_method = re.search(dialogue_memory_search_pattern, str_)[0]
        return f"_{memory_method}"
    
    turn_memory_search_pattern = "memory[0-9]*"
    if re.search(turn_memory_search_pattern, str_): 
        memory_method = re.search(turn_memory_search_pattern, str_)[0]
        return f"_{memory_method}"
    
    return ""

def get_memory_method_from_config(config) -> str: 
    
    memory_count = config["memory_num"]
    if config.get("memory_strategy") == "dialogue": 
        return f"_dialogue_memory{memory_count}"        
    
    if int(memory_count) == 0: 
        return ""
    
    return f"_memory{memory_count}"
    
def extract_dataset(str_): 
    if "SGD" in str_: 
        return "SGD"
    else: 
        return "MultiWOZ"
    
def extract_alignment_method(str_: str)-> str: 
    
    alignments = ["aligned", "mixed", "fixed", "random"]
    for al in alignments: 
        if al != "fixed": 
            if al in str_: 
                return al
        else: 
            if "fixed" in str_: 
                start_index = str_.index("fixed")
                return str_[start_index: start_index + len("fixed") + 2].replace("_","")

def extract_domain_ordering(str_: str) -> str: 
    
    order = re.search("ord:([0-9])", str_)
    if order: 
        return order[1]
    

def quick_tests(): 
    fn_name = "key_results_if:transferqa_icf:full_ex:3_bm25cf:last_train-erm:scs-bm25_dev-erm:scs-bm25_test-erm:bm25_to:aligned_tnr:-1.0.json"
    assert "scs-bm25" == extract_retrieval_method(fn_name, dir_name="", data_split="train")
    
    dir_name = "2023-02-11_17:23:05_transferqa_oracle_aligned_t5-small_ex:1_sd:40_SGD_ord:1"
    fn_name = "key_results_if:transferqa_icf:full_ex:1_bm25cf:icdst_test-erm:bm25_to:aligned_tnr:-1.0.json"
    
    assert "bm25" ==   extract_retrieval_method(fn_name, dir_name=dir_name, data_split="test")
    assert "scs-bm25" ==   extract_retrieval_method(fn_name, dir_name=dir_name, data_split="train")

def get_organized_results(): 
    quick_tests()

    all_results = []  
    column_name = ["config", "domain_order", "average_jga", "fwt", "bwt", "time", "fp"]
    all_results.append(column_name)
    print(",".join(column_name))
    for rdir in result_dirs: 
        result_fns = sorted(rdir.glob("key*.json"))

        # import pdb ; pdb.set_trace()        
        # print(rdir) 
        config_fn = list(rdir.glob("config*.json"))[0]
        with open(config_fn, "r") as f: 
            config = json.load(f)
        
        for rfn in result_fns: 
            with open(rfn, "r") as f: 
                result = json.load(f)
                        
            strategy = "stod"  if "simpletod" in rdir.name  else "tqa"
            # import pdb; pdb.set_trace() 
            result_file_name = rfn.name
            if "multitask" in result: 
                average_jga = result["multitask"]["upperbound_avg_jga"]
                fwt, bwt = 0, 0
                elapsed_time = result["multitask"]["elapsed_time"]
                
                example_type = extract_retrieval_method("", rdir.name, data_split="train")
                alignment = extract_alignment_method(rdir.name)
                memory_setting = get_memory_method_from_config(config)
                
                if alignment: 
                    config_name = f"{strategy}_upperbound_{example_type}:{alignment}"
                else:             
                    config_name = f"{strategy}_upperbound_{example_type}"
                domain_order = "-"
                result_row = [
                    config_name, 
                    domain_order, 
                    # f"{average_jga*100:.2f}%",
                    average_jga,
                    fwt, 
                    bwt, 
                    elapsed_time, 
                    str(Path(rdir.name) / result_file_name)
                ] 
                # print(",".join(result_row))
                all_results.append(result_row)
                
            if "cl" in result: 
                train_config = rdir.name
                test_config = rfn.name  
                
                elapsed_time = result["cl"].get("elapsed_time", "-")
                
                cl_results = result["cl"]
                average_jga = cl_results["average_jga"]
                fwt = cl_results['fwt']['original_fwt']
                bwt = cl_results['bwt']['original_bwt']
                domain_order = extract_domain_ordering(train_config)
                memory_setting = extract_memory_method(train_config) 
                    
                if "train_example_ranking_metric" in config:
                    train_example = config["train_example_ranking_metric"] 
                else: 
                    train_example = extract_retrieval_method(test_config,train_config, data_split="train")

                dev_example = config["dev_example_ranking_metric"]

                if not config["use_incontext_examples"]: 
                    train_example_count = "0" 
                    train_example = "vanilla"
                else: 
                    train_example_count = config.get("example_topk")
                
                train_alignment = config.get("transferqa_order", "aligned")
                
                
                # example_count = config['example_topk']
                test_example_count = extract_example_count(test_config)
                test_example = extract_retrieval_method(test_config, train_config, data_split="test")
                test_alignment = extract_alignment_method(test_config)
                
                
                if train_example == "vanilla": 
                    config_name = f"{strategy}_vanilla{memory_setting}_test:{test_example}:{test_example_count}"
                else: 
                    config_name = f"{strategy}_train:{train_example}:{train_example_count}_dev:{dev_example}:{train_example_count}_test:{test_example}:{test_example_count}{memory_setting}"
                result_row = [
                    config_name, 
                    domain_order, 
                    # f"{average_jga*100:.2f}%", 
                    # f"{fwt:.2f}",
                    # f"{bwt:.2f}",
                    average_jga, 
                    fwt, 
                    bwt,
                    elapsed_time,
                    str(Path(rdir.name) / result_file_name)
                ] 
                # print(",".join(result_row))
                all_results.append(result_row)

    # for row in sorted(all_results, key=lambda x: x[0]): 
    #     print("\t\t".join([f"{el*100:.2f}%" if isinstance(el, float) or isinstance(el, int) else el for el in row]))
                
    return all_results            

# format results 
def format_number(num): 
    if num < 0: 
        return "\\text{-}" + f"{abs(num)*100:.1f}"
    else:     
        return f"{num*100:.1f}"

def print_formatted_output(agg_result): 
    for idx, row in agg_result.iterrows(): 
        fjga_mean = format_number(row["average_jga"]["mean"])
        fjga_std = "{" + format_number(row["average_jga"]["std"]) + "}"
        
        fwt_mean = format_number(row["fwt"]["mean"])
        fwt_std = "{" + format_number(row["fwt"]["std"]) + "}"

        bwt_mean = format_number(row["bwt"]["mean"])
        bwt_std = "{" + format_number(row["bwt"]["std"]) + "}"
        
        # config = row["config"]
        result_str =  f"${fjga_mean}_{fjga_std}$ & ${fwt_mean}_{fwt_std}$ & ${bwt_mean}_{bwt_std}$ & \t{idx}"
        print(result_str)
    

if __name__ == "__main__": 
    pd.set_option('display.max_colwidth', None)
    
    all_results = get_organized_results()
    df = pd.DataFrame.from_records(all_results[1:], columns=all_results[0]).sort_values(["fp"])     
    print(tabulate(df.sort_values(by="average_jga")))         
    agg_result = df[["config", "average_jga", "fwt", "bwt"]].groupby("config").agg(["mean", "std", "count"]).sort_values(("average_jga", "mean"))
    
    print(agg_result)
    print_formatted_output(agg_result)