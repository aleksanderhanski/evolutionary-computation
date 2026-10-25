import math

import numpy as np


def greedy_cycle_weighted(D, costs, start: int, w_regret=1, w_best=1) -> list[int]:
    # the path is treated as a cycle all the time, so the closing edge counts too
    # picks the node with the highest w_regret * regret - w_best * best_delta
    # and inserts it at its best position

    n = len(D)
    k = math.ceil(n / 2)
    path = [start]

    while len(path) < k:
        free = np.array([v for v in range(n) if v not in path])

        # deltas[p][i] = change of the objective after inserting free[i]
        # between path[p - 1] and path[p] (p = 0 is the closing edge last -> first)
        deltas = []
        for p in range(len(path)):
            a, b = path[p - 1], path[p]
            deltas.append(D[a, free] + D[free, b] - D[a, b] + costs[free])
        deltas = np.array(deltas)

        best_position = deltas.argmin(axis=0)
        sorted_deltas = np.sort(deltas, axis=0)
        best = sorted_deltas[0]
        # a one-node cycle has a single place to insert, so nothing to regret
        second = sorted_deltas[1] if len(path) > 1 else best

        score = w_regret * (second - best) - w_best * best
        # highest score, ties (e.g. the first step, where all regrets are 0) go to the smaller delta
        i = np.lexsort((best, -score))[0]
        path.insert(best_position[i], free[i])

    return path


def greedy_cycle_regret(D, costs, start: int) -> list[int]:
    # pure 2-regret: the best delta itself is ignored
    return greedy_cycle_weighted(D, costs, start, w_regret=1, w_best=0)
