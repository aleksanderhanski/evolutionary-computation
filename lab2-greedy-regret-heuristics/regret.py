import math

import numpy as np


def insertion_deltas_path(D, costs, path) -> np.ndarray:
    # row p = change of the objective after inserting each node at position p
    # of the open path (p = 0 before the first node, p = len(path) after the last)
    a, b = np.array(path[:-1], dtype=np.int64), np.array(path[1:], dtype=np.int64)
    first = D[:, path[0]]
    inside = D[a] + D[:, b].T - D[a, b][:, None]
    last = D[path[-1]]
    return np.vstack([first, inside, last]) + costs


def insertion_deltas_cycle(D, costs, path) -> np.ndarray:
    # row p = change of the objective after inserting each node at position p,
    # i.e. between path[p - 1] and path[p]; row 0 is the closing edge (last, first)
    b = np.array(path)
    a = np.roll(b, 1)
    return D[a] + D[:, b].T - D[a, b][:, None] + costs


def regret_heuristic(D, costs, start: int, deltas_fn, w_regret=1.0, w_best=0.0) -> list[int]:
    # picks the node maximizing w_regret * regret - w_best * best_delta,
    # where regret = second best delta - best delta of that node,
    # and inserts it at its best position
    # w_regret=1, w_best=0 -> pure 2-regret; w_regret=1, w_best=1 -> equal weights

    n = len(D)
    k = math.ceil(n / 2)

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    path = [start]

    while len(path) < k:
        deltas = deltas_fn(D, costs, path).astype(np.float64)
        free = np.flatnonzero(~visited)
        deltas = deltas[:, free]

        best_position = deltas.argmin(axis=0)
        best = deltas[best_position, np.arange(len(free))]
        if len(deltas) > 1:
            second = np.partition(deltas, 1, axis=0)[1]
        else:
            # only one place to insert, so there is nothing to regret
            second = best
        regret = second - best

        score = w_regret * regret - w_best * best
        # ties (e.g. the first step, where all regrets are 0) go to the smaller delta
        chosen = np.lexsort((best, -score))[0]

        node = int(free[chosen])
        visited[node] = True
        path.insert(int(best_position[chosen]), node)

    return path


def nn_any_position_regret(D, costs, start: int) -> list[int]:
    return regret_heuristic(D, costs, start, insertion_deltas_path, 1.0, 0.0)


def nn_any_position_weighted(D, costs, start: int, w_regret=1.0, w_best=1.0) -> list[int]:
    return regret_heuristic(D, costs, start, insertion_deltas_path, w_regret, w_best)


def greedy_cycle_regret(D, costs, start: int) -> list[int]:
    return regret_heuristic(D, costs, start, insertion_deltas_cycle, 1.0, 0.0)


def greedy_cycle_weighted(D, costs, start: int, w_regret=1.0, w_best=1.0) -> list[int]:
    return regret_heuristic(D, costs, start, insertion_deltas_cycle, w_regret, w_best)
