import math

import numpy as np


def insertion_deltas(D, costs, path) -> np.ndarray:
    # the path is open while building, the closing edge only appears at the end
    # row p = change of the objective after inserting each node at position p
    # (p = 0 before the first node, p = len(path) after the last)
    a, b = np.array(path[:-1], dtype=np.int64), np.array(path[1:], dtype=np.int64)
    first = D[:, path[0]]
    inside = D[a] + D[:, b].T - D[a, b][:, None]
    last = D[path[-1]]
    return np.vstack([first, inside, last]) + costs


def nn_any_position_weighted(D, costs, start: int, w_regret=1.0, w_best=1.0) -> list[int]:
    # picks the node maximizing w_regret * regret - w_best * best_delta,
    # where regret = second best delta - best delta of that node,
    # and inserts it at its best position

    n = len(D)
    k = math.ceil(n / 2)

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    path = [start]

    while len(path) < k:
        free = np.flatnonzero(~visited)
        deltas = insertion_deltas(D, costs, path)[:, free].astype(np.float64)

        best_position = deltas.argmin(axis=0)
        best = deltas[best_position, np.arange(len(free))]
        second = np.partition(deltas, 1, axis=0)[1]
        regret = second - best

        score = w_regret * regret - w_best * best
        # ties (e.g. the first step, where all regrets are 0) go to the smaller delta
        chosen = np.lexsort((best, -score))[0]

        node = int(free[chosen])
        visited[node] = True
        path.insert(int(best_position[chosen]), node)

    return path


def nn_any_position_regret(D, costs, start: int) -> list[int]:
    # pure 2-regret: the best delta itself is ignored
    return nn_any_position_weighted(D, costs, start, w_regret=1.0, w_best=0.0)
