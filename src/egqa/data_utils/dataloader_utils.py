# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random
from typing import List, Tuple


def align(
    x: List[str],
    y: List[str],
    target_alignment_pct: float,
    global_alignment_ct: int,
    global_total: int,
) -> Tuple[List, int, int]:
    """_summary_

    Args:
        x (List[str]): list of slots
        y (List[str]): list of slots
        target_alignment_pct (float): target alignment
        global_alignment_ct (int): number of slot combinations that were aligned so far
        global_total (int): total number of slot combinations

    Returns:
        Tuple[List,int,int]: _description_
    """

    combs = []
    while x:
        # prioritize identical ones if present
        shared = list(set(x).intersection(set(y)))
        if shared:
            random.shuffle(shared)
            shared_chosen = shared.pop()
            x_index = x.index(shared_chosen)
            x_target = x.pop(x_index)
        else:
            random.shuffle(x)
            x_target = x.pop()
        try:
            y_index = y.index(x_target)
        except ValueError:
            y_index = -1

        if global_alignment_ct / global_total < target_alignment_pct:
            y_target = y.pop(y_index)
            combs.append([x_target, y_target])
            if x_target == y_target:
                global_alignment_ct += 1
        else:
            # remove the same one as we want misalignment
            y_exclude = y.pop(y_index)
            if y:
                random.shuffle(y)
                y_target = y.pop()
                combs.append([x_target, y_target])
                y.append(y_exclude)
            else:
                y_target = y_exclude
                combs.append([x_target, y_target])
        global_total += 1

    return combs, global_alignment_ct, global_total
