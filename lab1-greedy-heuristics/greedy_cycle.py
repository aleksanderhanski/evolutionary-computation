import math

import numpy as np


def greedy_cycle(D, costs, start: int) -> list[int]:
    # the path is treated as a cycle all the time, so the closing edge counts too

    n = len(D)
    k = math.ceil(n / 2)

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    path = [start]

    while len(path) < k:
        best_delta, best_node, best_position = np.inf, None, None

        node_costs = np.where(visited, np.inf, costs)

        for i in range(len(path)):
            a, b = path[i], path[(i + 1) % len(path)]
            delta = D[a] + D[b] - D[a, b] + node_costs
            node = int(delta.argmin())

            if delta[node] < best_delta:
                best_delta, best_node, best_position = delta[node], node, i + 1

        visited[best_node] = True
        path.insert(best_position, best_node)

    return path
