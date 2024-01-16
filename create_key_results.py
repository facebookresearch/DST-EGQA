# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from pathlib import Path 
import json
from egqa import compute_backward_transfer, compute_forward_transfer, compute_cl_metrics, compute_upperbound_metrics, DOMAIN_ORDERS
from organize_results import extract_domain_ordering, extract_dataset
import os 
from loguru import logger 


results_dir = "/project/jonmay_231/hjcho/cldst/CAQ/results/"

result_paths = Path(results_dir).glob("*")

domain_key = None 
for path in result_paths: 

    pred_files = list(Path(path).glob("test_predictions_raw*"))

    print(f"{len(pred_files)} files found in {str(path)}")

    for pred in pred_files:             
        results_file = str(pred).replace("test_predictions_raw", "key_results")
        elapsed_time = "-"
        if Path(results_file).is_file(): 
            # remove file and recalculate 
            with open(results_file, "r") as f: 
                prev_results = json.load(f) 
            elapsed_time = list(prev_results.values())[0].get("elapsed_time", "-")
            Path(results_file).unlink()
        
        if not Path(results_file).is_file(): 
            
            with open(pred, "r") as f: 
                predictions = json.load(f) 

            dataset = extract_dataset(str(path))
            domain_key = extract_domain_ordering(str(path)) 
            if "upperbound" in str(path): 
                domain_key = 1 
            
            if int(domain_key) in DOMAIN_ORDERS[dataset]: 
                domain_key = int(domain_key)
                
            domain_order = DOMAIN_ORDERS[dataset][domain_key]
            strategy = "transferqa" if "transferqa" in str(path) else "simpletod"
            if dataset=="SGD": 
                datapath = os.path.join(os.environ.get("DATA_DIR"), "dstc8-schema-guided-dialogue/lifelong_cpt")
            else: 
                datapath = os.path.join(os.environ.get("DATA_DIR"), "MultiWOZ_2.4/lifelong")
            
            if "upperbound" not in str(path): 
                main_results = compute_cl_metrics(predictions, dataset=dataset, trained_domain_order=domain_order, strategy=strategy)
                fwt = compute_forward_transfer(main_results['complete_jga_matrix'], domain_order)
                bwt = compute_backward_transfer(main_results['complete_jga_matrix'], domain_order)
                main_results['fwt'] = fwt
                main_results['bwt'] = bwt
                eval_config = 'cl'

            else: 
                main_results = compute_upperbound_metrics(predictions, dataset=dataset, trained_domains=domain_order, strategy=strategy)
                main_results['fwt'] = 0 
                main_results['bwt'] = 0 
                eval_config = 'multitask'
            main_results["elapsed_time"] = elapsed_time 

            key_results ={
                eval_config: main_results
            }
                        
            with open(results_file, "w") as f: 
                json.dump(key_results, f, indent=4)
        

            
            

