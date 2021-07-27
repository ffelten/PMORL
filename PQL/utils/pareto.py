import numpy as np
from . import Front
import pygmo as pg


def pareto_efficient(points: list[float]) -> Front:
    """
    Filters out the Pareto inefficient points (supposed maximization)
    :param points: the input pareto set to filter
    :return: a set of Pareto efficient points among points
    """
    front = []
    if len(points) >= 2:
        _, dl, _, _ = pg.fast_non_dominated_sorting(points=points)
        for i in range(len(dl)):
            if dl[i].size == 0:
                front.append(points[i])
    elif len(points) == 1:
        front = points
    return list(np.unique(np.array(front), axis=0))
