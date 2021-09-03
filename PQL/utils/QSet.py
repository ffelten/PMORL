import numpy as np

from PQL.utils.pareto import pareto_efficient
import matplotlib.pyplot as plt

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
        self.set = list(pareto_efficient(self.set))

    def td_update(self, gamma, reward):
        self.set = list(map(lambda v: reward + (gamma * v), self.set))
        if not self.set:
            self.set.append(reward)
        self.set = list(pareto_efficient(self.set))

    def clone_td(self, gamma, reward):
        new = QSet()
        new.append(self)
        new.td_update(gamma, reward)
        return new

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

