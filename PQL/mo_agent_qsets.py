from numpy import ndarray

from mo_agent import MOGridWorldAgent
from numpy.typing import NDArray
import numpy as np
from utils.QSet import QSet


class MOGridWorldQSets(MOGridWorldAgent):
    """
    MAXIMIZATION IS ASSUMED
    """
    def heuristic(self, obs: NDArray[int]) -> int:
        """
            Heuristic based on intervals
        """
        ## QSets of all actions
        actions_qsets = self.qsets(obs)

        ## For each action objective, we form an interval (min, max) to be able to compare the qsets
        ## This array is in the form (action_index, interval_index, objective_index)
        actions_intervals = np.zeros((self.env.actions, 2, self.num_objectives))

        for i in range(len(actions_qsets)):
            action_qset: QSet = actions_qsets[i]
            ## Those are vectors, for each objective, I have the min & max
            ## [min_obj1, ..., min_obj_n]
            mins = action_qset.compute_mins()
            if np.all(mins == 0):
                mins = np.zeros_like(mins)
                # mins.fill(float("-inf")) ## Making unvisited state interesting
            ## [max_obj1, ..., max_obj_n]
            maxs = action_qset.compute_maxs()
            if np.all(maxs == 0):
                maxs = np.empty(self.num_objectives, dtype=np.float)
                maxs.fill(float("inf")) ## Making unvisited state interesting

            actions_intervals[i] = np.vstack([mins, maxs])

        dominance_matrix = self.dominance(actions_intervals)

        dominated_scores = np.sum(dominance_matrix, axis=0)

        less_dominated_sets = np.argwhere(dominated_scores == np.amin(dominated_scores)).flatten()
        if len(less_dominated_sets) == 1:
            return less_dominated_sets[0]
        else:
            # Use pythagorean measure to include diversity
            # axis_lengths = actions_intervals[:, 1, :] - actions_intervals[:, 0, :]
            # pythagorean_actions_values = np.sum(np.square(axis_lengths), axis=1)
            # # Replaces infinity with zeroes
            # # pythagorean_actions_values[pythagorean_actions_values == float("inf")] = 0.
            # # Include only the dominating actions
            # pythagorean_actions_values = pythagorean_actions_values[less_dominated_sets]
            # biggest_pyth = np.argwhere(pythagorean_actions_values == np.nanmax(pythagorean_actions_values)).flatten()
            # if len(biggest_pyth) == 1:
            #     return less_dominated_sets[biggest_pyth[0]]
            # else:
                # If there are equalities, randomly chooses among the equal pareto fronts
                return less_dominated_sets[self.rng.integers(low=0, high=len(less_dominated_sets))]

    def dominance(self, actions_intervals: ndarray):
        """
        Computes the indices of the strongly dominant actions
        :param actions_intervals: array of shape (#actions, 2, #objectives) containing the min and max for each objective of each action
        :return: a domination matrix s.t. (i,j) is True when action i dominates j
        """

        ## Contains true in i,j if i strong dominates j
        domination_matrix = np.zeros((len(actions_intervals), len(actions_intervals)), dtype=np.bool)

        for i in range(len(actions_intervals)):
            for j in range(len(actions_intervals)):
                if i!=j:
                    ## Contains an array of shape (2, #obj)
                    first_action_intervals = actions_intervals[i]
                    second_action_intervals = actions_intervals[j]

                    zipped = list(zip(first_action_intervals.T, second_action_intervals.T))

                    certain_domination = list(map(lambda x: certain_dominates(x[0], x[1]),
                                                  zipped))
                    uncertain_strong_domination = list(map(lambda x: uncertain_strong_dominates(x[0], x[1]),
                                                           zipped))
                    uncertain_weak_domination = list(map(lambda x: uncertain_weak_dominates(x[0], x[1]),
                                                         zipped))

                    ## Case 1, all certain dominates
                    case1 = np.all(certain_domination)

                    ## Case 2, all us dominate
                    case2 = np.all(uncertain_strong_domination)

                    ## Case 3, one uncertain, all other certain
                    uncertain = np.argmax(uncertain_strong_domination)
                    case3 = np.sum(uncertain_strong_domination) == 1 and \
                            np.sum(certain_domination) == (self.num_objectives - 1) and \
                            certain_domination[uncertain] == False

                    ## Case 4, weak dominance, one uncertain weak and all the others certain or uncertain strong
                    uncertain_weak = np.argmax(uncertain_weak_domination)
                    certain_or_strong_domination = np.logical_or(certain_domination, uncertain_strong_domination)
                    case4 = np.sum(uncertain_weak_domination) == 1 and \
                            np.sum(certain_or_strong_domination) == (self.num_objectives - 1) and \
                            certain_or_strong_domination[uncertain_weak] == False

                    if (case1 or case2 or case3 or case4):
                        domination_matrix[i, j] = True

        return domination_matrix


def certain_dominates(interval1, interval2):
    """
    Checks if interval1 certain dominates interval2. That is, min(interval1) > max(interval2).
    :return: True if interval1 certain dominates interval2
    """
    return interval1[0] > interval2[1]


def uncertain_strong_dominates(interval1, interval2):
    """
    Checks if interval1 uncertain strong dominates interval2. That is,
    min(interval1) <= max(interval2) & min(interval1) >= min(interval2) & max(interval1) >= max(interval2).
    :return: True if interval1 certain dominates interval2
    """
    return (interval1[0] <= interval2[1]) and (interval1[0] >= interval2[0]) and (interval1[1] >= interval2[1])


def uncertain_weak_dominates(interval1, interval2):
    """
    Checks if interval1 uncertain weak dominates interval2. That is,
    min(interval1) <= max(interval2) & [
        (max(interval1) < max(interval2) & min(interval1) >= min(interval2)) |
        (max(interval1) >= max(interval2) & min(interval1) < min(interval2))
    ].
    :return: True if interval1 certain dominates interval2
    """
    return (interval1[0] <= interval2[1]) and (
            ((interval1[1] < interval2[1]) and (interval1[0] >= interval2[0])) or
            ((interval1[1] >= interval2[1]) and (interval1[0] < interval2[0]))
    )
