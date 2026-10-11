import math

import numpy as np


def nearest_neighbor_path_end_only(D, costs, start: int) -> list[int]:
    # finding nearest neighbor only for the last node in the path

    n = len(D)
    k = math.ceil(n / 2)

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    path = [start]
    curr_node = start

    while len(path) < k:
        # cost of going to each node = edge length + that node's cost
        step_cost = (D[curr_node] + costs).astype(np.float64)
        step_cost[visited] = np.inf
        next_node = int(step_cost.argmin())

        visited[next_node] = True
        path.append(next_node)
        curr_node = next_node

    return path
