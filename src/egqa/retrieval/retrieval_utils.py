# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from tqdm import tqdm


def compute_prf(gold: List[str], pred: List[str]) -> float:
    """Compute f1 score for two lists (without duplicates)
    Referenced from: https://github.com/Yushi-Hu/IC-DST/blob/main/evaluate_metrics.py

    Args:
        gold (List[str]): label list
        pred (List[str]): prediction list

    Returns:
        float: F1 score
    """
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return float(F1)


def reformat_state_change(
    state_change: Dict[str, List[str]], configuration: str
) -> Dict[str, str]:
    """Transform from {slot key: [operation, slot value]} -> {slot key-operation: slot value}

    Args:
        state_change (Dict[str, List[str]]): dictionary containing state changes with key and operation separated
        configuration (str): configuration for reformating. the format to use for the state changes. one of [original, modified]

    Returns:
        Dict[str, str]: dictionary with state changes with keys that combine slot key and operation
    """

    new_state_change = {}

    for key, value in state_change.items():
        operation = value[0]
        slot_value = value[1]
        # original format used by IC-DST
        if configuration == "original":
            if operation == "DELETE":
                new_state_change[key] = operation
            else:
                new_state_change[key] = slot_value
        # our modified format that includes operation into the key
        elif configuration == "modified":
            new_state_change[f"{key}-{operation}"] = slot_value

    return new_state_change


def compute_state_change_similarity(
    src_state_change: Dict[str, List[str]],
    tgt_state_change: Dict[str, List[str]],
    configuration: str = "modified",
    check_commutative: bool = False,
) -> float:
    """_summary_

    Args:
        src_state_change (Dict[str, List[str]]): state change information of the source turn
        tgt_state_change (Dict[str, List[str]]): state chagne information of the target turn
        configuration (str): the format to use for the state changes. one of [original, modified]
        check_commutative (bool, optional): check whether the f1 score is commutative. only use for testing. Defaults to False.

    Returns:
        float: state change similarity value as defined by IC-DST (with modifications to the key value)
    """
    src_state_change = reformat_state_change(
        src_state_change, configuration=configuration
    )
    tgt_state_change = reformat_state_change(
        tgt_state_change, configuration=configuration
    )

    src_slotkeys_only = list(src_state_change.keys())
    tgt_slotkeys_only = list(tgt_state_change.keys())
    f_slot = compute_prf(src_slotkeys_only, tgt_slotkeys_only)
    if check_commutative:
        commutative_f_slot = compute_prf(tgt_slotkeys_only, src_slotkeys_only)
        assert f_slot == commutative_f_slot, (
            f_slot,
            commutative_f_slot,
            src_slotkeys_only,
            tgt_slotkeys_only,
        )

    src_full_slot = [
        [slot_key, slot_value] for slot_key, slot_value in src_state_change.items()
    ]
    tgt_full_slot = [
        [slot_key, slot_value] for slot_key, slot_value in tgt_state_change.items()
    ]

    f_slot_value = compute_prf(src_full_slot, tgt_full_slot)
    if check_commutative:
        commutative_f_slot_value = compute_prf(src_full_slot, tgt_full_slot)
        assert f_slot_value == commutative_f_slot_value, (
            f_slot_value,
            commutative_f_slot_value,
            src_full_slot,
            tgt_full_slot,
        )

    similarity = (f_slot + f_slot_value) / 2
    return similarity


def get_state_change_similarity_matrix(
    flattened_turns: List[Dict[str, object]], indices_to_score: List[int] = None
) -> np.array:
    """_summary_

    Args:
        flattened_turns (List[Dict[str, object]]): _description_
        indices_to_score (List[int], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_turns = len(flattened_turns)
    logger.info("Computing state change similarity matrix...")

    if indices_to_score:

        similarity_matrix = {
            i: {
                j: compute_state_change_similarity(
                    flattened_turns[i]['state_change'],
                    flattened_turns[j]['state_change'],
                    check_commutative=False,
                )
                for j in range(indices_to_score[0])
            }
            for i in tqdm(range(indices_to_score[0], n_turns))
        }

        similarity_matrix = np.array(
            [
                [similarity_matrix[i][j] for j in range(indices_to_score[0])]
                for i in range(indices_to_score[0], n_turns)
            ]
        )

        # import pdb; pdb.set_trace()
        assert similarity_matrix.shape == (len(indices_to_score), indices_to_score[0])

    else:

        similarity_matrix = {
            i: {
                j: compute_state_change_similarity(
                    flattened_turns[i]['state_change'],
                    flattened_turns[j]['state_change'],
                    check_commutative=False,
                )
                for j in range(i, n_turns)
            }
            for i in tqdm(range(n_turns))
        }

        # assign commutative valuse a_ij = a_ji
        for i in range(n_turns):
            for j in range(n_turns):
                similarity_matrix[j][i] = similarity_matrix[i][j]

        # transform to numpy array
        similarity_matrix = np.array(
            [[similarity_matrix[i][j] for j in range(n_turns)] for i in range(n_turns)]
        )
        assert similarity_matrix.shape == (n_turns, n_turns)

    return similarity_matrix


def retrieve_top_example_indices_and_scores(
    retrieval_scores: Dict[str, List[float]], 
    topk: int,
    data_split: str = "train",
    scores_for_ranking: str = "scs-bm25",
) -> List[Dict[str, Union[int, float]]]:
    """_summary_

    Args:
        state_change_similarity_scores (List[float]): state change similarity scores
        bm25_doc_scores (List[floa]): bm25 similarity scores
        topk (int): maximum number of examples to retrieve
        data_split (str, optional): current train/dev/test split. Defaults to "train".
        scores_for_eval (str, optional): scores to use for ranking. Defaults to "scs-bm25".

    Raises:
        NotImplementedError: if an invalid option is given that is not one of [scs-bm25, bm25, state_change_sim]

    Returns:
        List[Dict[str, Union[int, float]]]: list of dictionaries that contain information about the retrieved examples
    """

    state_change_similarity_scores = retrieval_scores['state_change_similarity']
    bm25_doc_scores = retrieval_scores['bm25']
    sentbert_scores = retrieval_scores['sentbert']
    icdst_scores = retrieval_scores['icdst']
    custom_icdst_scores = retrieval_scores['custom_icdst']
    gpt_emb_scores = retrieval_scores['gpt']

    assert len(state_change_similarity_scores) == len(bm25_doc_scores), (len(state_change_similarity_scores), len(bm25_doc_scores))
    assert len(state_change_similarity_scores) == len(icdst_scores), (len(state_change_similarity_scores), len(icdst_scores))

    grouped = [
        {
            "index": j,
            "state_change_sim": state_change_similarity_scores[j],
            "bm25": bm25_doc_scores[j],
            "icdst": icdst_scores[j], 
            "gpt": gpt_emb_scores[j], 
            "sentbert": sentbert_scores[j],
            "custom_icdst": custom_icdst_scores[j],
        }
        for j in range(len(state_change_similarity_scores))
    ]

    if scores_for_ranking == "scs-bm25":
        grouped_sorted = sorted(
            grouped, key=lambda x: (x["state_change_sim"], x["bm25"]), reverse=True
        )
    elif scores_for_ranking in grouped[0].keys(): 
        grouped_sorted = sorted(grouped, key=lambda x: x[scores_for_ranking], reverse=True)
    else:
        logger.error(
            f"{scores_for_ranking} is not a valid option. choose one of [scs-bm25, bm25, state_change_sim]"
        )
        raise NotImplementedError

    if data_split == "train":
        # remove itself as a candidate
        candidate_dict_list = grouped_sorted[1 : (topk + 1)]
    else:
        candidate_dict_list = grouped_sorted[:topk]

    return candidate_dict_list


# reference: https://gist.github.com/bwhite/3726239
def compute_mean_average_precision_from_ranks(gold_ranks, pred_ranks):
    all_relevance_scores = []
    for gold_rank, pred_rank in zip(gold_ranks, pred_ranks):
        all_relevance_scores.append(compute_relevance_score(gold_rank, pred_rank))

    return mean_average_precision(all_relevance_scores)


def compute_relevance_score(gold_rank, pred_rank):
    relevance_scores = []
    for query in pred_rank:
        relevance = 1 if query in gold_rank else 0
        relevance_scores.append(relevance)

    return relevance_scores


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.0


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def compute_mean_reciprocal_rank(
    gold_rankings: List[List[str]], top_predictions: List[str]
) -> float:

    assert len(top_predictions) == len(gold_rankings)

    mrr = 0
    for top_pred, gold_rank in zip(top_predictions, gold_rankings):
        if top_pred in gold_rank:
            mrr += 1 / (gold_rank.index(top_pred) + 1)

    return mrr / len(top_predictions)
