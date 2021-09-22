import numpy as np

from PQL.utils.pareto import pareto_efficient
import matplotlib.pyplot as plt
import pygmo as pg


def round_set(s):
    tmp = np.around(np.array(s), decimals=2)
    return list(tmp)


class QSet:
    """
    Set of undominated q-vectors
    """

    def __init__(self, qarray=None):
        if qarray is None:
            qarray = []
        self.set = list(qarray)

    def append(self, qset):
        self.set += qset.set
        self.paretoize()

    def td_update(self, gamma, reward):
        self.set = list(map(lambda v: reward + (gamma * v), self.set))
        if not self.set:
            self.set.append(reward)
        self.paretoize()

    def paretoize(self):
        self.set = list(pareto_efficient(round_set(self.set)))

    def clone(self):
        new = QSet()
        new.append(self)
        return new

    def clone_td(self, gamma, reward):
        new = self.clone()
        new.td_update(gamma, reward)
        return new

    def worst_point(self) -> int:
        crowding_distances = pg.crowding_distance(points=self.set)
        return np.argwhere(crowding_distances == np.amin(crowding_distances))[0,0]

    def shrink(self, max_size: int):
        """ Removes the worst points front the current qset """
        if len(self.set) > max_size:
            for i in range(len(self.set) - max_size):
                to_remove = self.worst_point()
                self.set.pop(to_remove)

    def compute_means(self):
        """
        :return: a vector of means per objective
        """
        set_as_array = np.array(self.set)
        return set_as_array.mean(axis=0)

    def compute_mins(self):
        if self.set:
            set_as_array = np.array(self.set)
            return set_as_array.min(axis=0, initial=float("inf"))
        else:
            return []

    def compute_maxs(self):
        if self.set:
            set_as_array = np.array(self.set)
            return set_as_array.max(axis=0, initial=float("-inf"))
        else:
            return []

    def __str__(self):
        return str(self.set)

    def to_set(self) -> set:
        return set([tuple(arr) for arr in self.set])

    def draw_front_2d(self):
        treasure, time = zip(*self.set)
        plt.scatter(time, treasure)
        plt.show()

