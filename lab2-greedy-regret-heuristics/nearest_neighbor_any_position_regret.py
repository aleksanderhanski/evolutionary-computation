import math

import numpy as np


def nn_any_position_weighted(D, costs, start: int, w_regret=1, w_best=1) -> list[int]:
    # the path is open while building, the closing edge only appears at the end
    # picks the node with the highest w_regret * regret - w_best * best_delta
    # and inserts it at its best position

    n = len(D)
    k = math.ceil(n / 2)
    path = [start]

    while len(path) < k:
        free = np.array([v for v in range(n) if v not in path])

        # deltas[p][i] = change of the objective after inserting free[i] at position p
        deltas = []
        for p in range(len(path) + 1):
            if p == 0:
                delta = D[free, path[0]]
            elif p == len(path):
                delta = D[path[-1], free]
            else:
                a, b = path[p - 1], path[p]
                delta = D[a, free] + D[free, b] - D[a, b]
            deltas.append(delta + costs[free])
        deltas = np.array(deltas)

        best_position = deltas.argmin(axis=0)
        sorted_deltas = np.sort(deltas, axis=0)
        best, second = sorted_deltas[0], sorted_deltas[1]

        score = w_regret * (second - best) - w_best * best
        # highest score, ties (e.g. the first step, where all regrets are 0) go to the smaller delta
        i = np.lexsort((best, -score))[0]
        path.insert(best_position[i], free[i])

    return path


def nn_any_position_regret(D, costs, start: int) -> list[int]:
    # pure 2-regret: the best delta itself is ignored
    return nn_any_position_weighted(D, costs, start, w_regret=1, w_best=0)
