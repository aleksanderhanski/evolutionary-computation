import math

import numpy as np


def insertion_deltas(D, costs, path) -> np.ndarray:
    # the path is treated as a cycle all the time, so the closing edge counts too
    # row p = change of the objective after inserting each node at position p,
    # i.e. between path[p - 1] and path[p]; row 0 is the closing edge (last, first)
    b = np.array(path, dtype=np.int64)
    a = np.roll(b, 1)
    return D[a] + D[:, b].T - D[a, b][:, None] + costs


def greedy_cycle_weighted(D, costs, start: int, w_regret=1.0, w_best=1.0) -> list[int]:
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
        if len(path) > 1:
            second = np.partition(deltas, 1, axis=0)[1]
        else:
            # one-node cycle has a single place to insert, so nothing to regret
            second = best
        regret = second - best

        score = w_regret * regret - w_best * best
        # ties (e.g. the first step, where all regrets are 0) go to the smaller delta
        chosen = np.lexsort((best, -score))[0]

        node = int(free[chosen])
        visited[node] = True
        path.insert(int(best_position[chosen]), node)

    return path


def greedy_cycle_regret(D, costs, start: int) -> list[int]:
    # pure 2-regret: the best delta itself is ignored
    return greedy_cycle_weighted(D, costs, start, w_regret=1.0, w_best=0.0)
