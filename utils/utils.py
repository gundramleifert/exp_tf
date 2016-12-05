# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import string

def string_target_tensor_to_ctc_tensor(targetTensor, charMap):
    res = []
    for tgt in targetTensor:
        aTgt = []
        for c in tgt:
            aTgt.append(charMap[c])
        res.append(aTgt)

    print res
    return target_list_to_sparse_tensor(res)


def target_list_to_sparse_tensor(targetList):
    """
    Args:
    Inputs:
     List of lists with the MAPPED int representations of the chars
    Returns:
     Tuple necessary to build the sparse tensor
    """
    indices=[]
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
    return (np.array(indices), np.array(vals), np.array(shape))



def get_charmap_lp():
    cm = {str(x): x for x in range(10)}
    for idx, Z in enumerate(string.ascii_uppercase):
        cm[Z] = idx + 10
    cm['Ä'] = len(cm)
    cm['Ü'] = len(cm)
    cm['Ö'] = len(cm)
    finIdx = len(cm)
    cm['-'] = finIdx
    cm['_'] = finIdx
    cm[' '] = finIdx
    return cm



