import numpy as np
import pygmo as pg

def moves_containing_nd_points(qsets, nd_set):
    """
    Returns an array of booleans where an element is true if the corresponding qset contains an undominated point
    :param qsets: all the qsets for each move
    :param nd_set: ND(U qsets)
    :return: an array containing one if the corresponding move contains non dominated points in the nd set
    """
    if (len(nd_set)) <= 1: # need at least two points to compare fronts
        return __inter_qsets_front(qsets, nd_set)

    negated_ndset = np.array(nd_set) * -1.


    ndf = [nd_set[p] for p in pg.fast_non_dominated_sorting(points=negated_ndset)[0][0]]
    front = set(ndf)

    return __inter_qsets_front(qsets, front)

def __inter_qsets_front(qsets, front):
    non_dominated_moves = np.zeros_like(qsets)

    for a in range(len(qsets)):
        if len(qsets[a].to_set().intersection(front)) > 0:
            non_dominated_moves[a] = True

    return non_dominated_moves