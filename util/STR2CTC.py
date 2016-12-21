# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import string




def target_string_to_ctc_repr(targetString, charMap):
    aTgt = []
    for c in targetString:
        aTgt.append(charMap[c])
    return aTgt

def target_string_list_to_ctc_tensor_repr(targetList, charMap):
    res = []
    for tgt in targetList:
        # encoded = tgt.decode('utf-8')
        aTgt = []
        for c in tgt:
            try:
                aTgt.append(charMap.get_channel(c))
            except KeyError as error:
                print("KeyError: character {} not in charMap, skipping...".format(error))
        res.append(aTgt)

    #print res
    return target_list_to_sparse_tensor_repr(res)


def target_list_to_sparse_tensor_repr(targetList):
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



# def get_charmap_lp():
#     classes = 0
#     cm = {str(x): x for x in range(10)}
#     classes += 10
#     for idx, Z in enumerate(string.ascii_uppercase):
#         cm[Z] = idx + 10
#         classes += 1
#     cm[u'Ä'] = len(cm)
#     classes += 1
#     cm[u'Ü'] = len(cm)
#     classes += 1
#     cm[u'Ö'] = len(cm)
#     classes += 1
#     finIdx = len(cm)
#     cm['-'] = finIdx
#     cm['_'] = finIdx
#     cm[' '] = finIdx
#     cm[u'.'] = finIdx
#     cm['.'] = finIdx
#     classes += 1
#     return cm, classes

# def get_charmap_dft():
#     cm = {}
#     for idx, Z in enumerate(string.printable):
#         cm[Z] = idx
#     return cm
