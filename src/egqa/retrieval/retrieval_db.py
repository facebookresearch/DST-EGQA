# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# make it similar to BM25 model setup 
from sentence_transformers import SentenceTransformer, util 
import pickle 
import torch 
import openai 
from openai.embeddings_utils import get_embedding, get_embeddings, cosine_similarity
import os 
from typing import List, Dict 
from loguru import logger 
import time 
import torch 

ICDST_MODELPATH = os.path.join(os.environ["PROJECT_DIR"], 'mw21_5p_v2')
CUSTOM_ICDST_MODELPATH = os.path.join(os.environ["RESULTS_DIR"], f"custom_SGD_icdst_")

class Retrieval_DB: 
    def __init__(self): 
        pass 
    
    def encode(self, query): 
        pass 
    
    def precompute_scores(self): 
        
        self.scores = {
            "train": util.cos_sim(self.corpus_map["train"]["embeddings"], self.corpus_map["train"]["embeddings"]), 
            "dev": util.cos_sim(self.corpus_map["dev"]["embeddings"], self.corpus_map["train"]["embeddings"]), 
            "test": util.cos_sim(self.corpus_map["test"]["embeddings"], self.corpus_map["train"]["embeddings"]) 
        }
        
    def search_query_data_split_and_index(self, query): 
        for data_split, text_embeddings in self.corpus_map.items(): 
            if query in text_embeddings["text"]: 
                idx = text_embeddings["text"].index(query)
                return data_split, idx 
        return None, None 
        
    
    def get_scores(self, query:str): 

        query_embedding = None  
        found_data_split, idx = self.search_query_data_split_and_index(query)

        if found_data_split: 
            try: 
                # retrieve precomputed scores
                return self.scores[found_data_split][idx]
            except Exception as e: 
                logger.warning(e)       
                query_embedding = self.corpus_map[found_data_split]["embeddings"][idx]
                  
        if query_embedding is None:
            logger.warning(f"{query} not found in any of the pre-embedded texts.")
            query_embedding= self.encode(query)

        cos_scores = util.cos_sim(query_embedding, self.corpus_map["train"]["embeddings"])[0]        
        return cos_scores 

class SentBERT_DB(Retrieval_DB): 
    
    def __init__(self, corpus: Dict[str,List[str]], db_name:str =None, model="all-mpnet-base-v2", evaluator:str = None, domain:str = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_cache_path = os.path.join(os.environ.get("DATA_DIR"), f"cached_{model}_db_{db_name}.pkl")
        if evaluator and domain: 
            self.db_cache_path = os.path.join(os.environ.get("DATA_DIR"), f"cached_{model}_{evaluator}_{domain}_db_{db_name}.pkl")
            
        if model=="icdst": 
            model = ICDST_MODELPATH
        if model == "custom_icdst": 
            model = CUSTOM_ICDST_MODELPATH + f"{evaluator}_{domain}"
            
        try: 
            self.sentbert_model = SentenceTransformer(model, device=device)
            self.corpus_map = {} 
            
            if db_name and os.path.isfile(self.db_cache_path): 
                with open(self.db_cache_path, "rb") as f:       
                    self.corpus_map = pickle.load(f)
            else: 
                for data_split, texts in corpus.items(): 
                    logger.info(f"Encoding {data_split} samples of {len(texts)} texts with SentBERT embeddings using model: {model}")
                    start_time = time.time()
                    corpus_embeddings = self.sentbert_model.encode(texts, convert_to_tensor=True)
                    logger.info(f"Completed in {time.time() - start_time:.2f}s")
                    self.corpus_map[data_split] = {
                        "text": texts,
                        "embeddings": corpus_embeddings
                    }
                    
            if db_name and not os.path.isfile(self.db_cache_path): 
                with open(self.db_cache_path, "wb") as f: 
                    pickle.dump(self.corpus_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.precompute_scores()
            
        except Exception as e:
            logger.warning(f"Error while trying to load SnetBERT model: {e}")
            if not os.path.isdir(model):
                logger.warning(f"No SentBERT model found in `{model}`")
        
    def encode(self, query): 
        return self.sentbert_model.encode(query,convert_to_tensor=True)

    
# class ICDST_DB(Retrieval_DB)

class GPTEmbeddings_DB(Retrieval_DB): 
    
    def __init__(self, corpus:Dict[str,List[str]], db_name:str=None, model:str="text-embedding-ada-002"): 
        self.db_cache_path = os.path.join(os.environ.get("DATA_DIR"), f"cached_gpt_db_{db_name}.pkl")
        self.corpus_map = {}
        
        OPENAI_LIMIT = 2048 # maximium batch size for gpt
        
        self.model = model 
        if db_name and os.path.isfile(self.db_cache_path): 
            with open(self.db_cache_path, "rb") as f:            
                self.corpus_map = pickle.load(f)                
        else: 
            for data_split, texts in corpus.items(): 
                logger.info(f"Encoding {data_split} samples of {len(texts)} texts with GPT embeddings using model: {model}")
                start_time = time.time()
                if len(texts) > OPENAI_LIMIT: 
                    corpus_embeddings = [] 
                    for steps in range(0, len(texts), OPENAI_LIMIT): 
                        corpus_embeddings += get_embeddings(texts[steps:steps+OPENAI_LIMIT], engine=model)
                else: 
                    corpus_embeddings = get_embeddings(texts, engine=model)
                logger.info(f"Completed in {time.time() - start_time:.2f}s")
                self.corpus_map[data_split] = {
                    "text": texts,
                    "embeddings": corpus_embeddings
                }
        
        if db_name and not os.path.isfile(self.db_cache_path): 
            with open(self.db_cache_path, "wb") as f: 
                pickle.dump(self.corpus_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.precompute_scores()
        
    def encode(self, query): 
        return get_embedding(query, engine=self.model)
    

if __name__ == "__main__": 
    query = "what's up?"
    
    corpus = {
        "train": ["hello", "good bye", "i like to eat pizza", "can I have a cup of water"], 
        "dev": ["hola"],
        "test": [query]
    }
    
    sentbert_db = SentBERT_DB(corpus)
    sentbert_scores = sentbert_db.get_scores(query)
    
    icdst_db = SentBERT_DB(corpus, model="icdst")
    gpt_db = GPTEmbeddings_DB(corpus) 
    
    icdst_scores = icdst_db.get_scores(query)
    gpt_scores = gpt_db.get_scores(query)
    
    print(sentbert_scores)
    print(icdst_scores)
    print(gpt_scores)
    