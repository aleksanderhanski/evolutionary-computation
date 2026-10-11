import math

import numpy as np


def nearest_neighbor_any_position(D, costs, start: int) -> list[int]:
    # the path is open while building, the closing edge only appears at the end

    n = len(D)
    k = math.ceil(n / 2)

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    path = [start]

    while len(path) < k:
        best_delta, best_node, best_position = np.inf, None, None

        for i in range(len(path) + 1):
            if i == 0:
                delta = D[:, path[0]] + costs
            elif i == len(path):
                delta = D[path[-1]] + costs
            else:
                # between a and b, edge a->b replaced by a->node->b
                a, b = path[i - 1], path[i]
                delta = D[a] + D[:, b] - D[a][b] + costs

            delta = delta.astype(np.float64)
            delta[visited] = np.inf
            node = int(delta.argmin())

            if delta[node] < best_delta:
                best_delta, best_node, best_position = delta[node], node, i

        visited[best_node] = True
        path.insert(best_position, best_node)

    return path
