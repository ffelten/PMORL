import numpy as np

from PQL.utils.pareto import pareto_efficient
import matplotlib.pyplot as plt

class QSet:
    """
    Set of undominated q-vectors
    """

    def __init__(self):
        self.set = []

    def append(self, qset):
        self.set += qset.set
        self.set = list(pareto_efficient(self.set))

    def td_update(self, gamma, reward):
        self.set = list(map(lambda v: reward + (gamma * v), self.set))
        if not self.set:
            self.set.append(reward)
        self.set = list(pareto_efficient(self.set))

    def __str__(self):
        return str(self.set)

    def draw_front_2d(self):
        treasure, time = zip(*self.set)
        plt.scatter(time, treasure)
        plt.show()

