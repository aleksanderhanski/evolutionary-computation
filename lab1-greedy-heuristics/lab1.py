import sys
import math
from pathlib import Path

import numpy as np

# helpers.py lives one level up, shared by all labs
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helpers import DATA_DIR, load_data, distance_matrix, objective, check_solution


def random_solution(D, costs, rng) -> list[int]:
    # k distinct nodes, already in random order
    n = len(D)
    k = math.ceil(n / 2)
    return [int(node) for node in rng.choice(n, size=k, replace=False)]


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


def greedy_cycle(D, costs, start: int) -> list[int]:
    # the path is treated as a cycle all the time, so the closing edge counts too
    
    n = len(D)
    k = math.ceil(n / 2)

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    path = [start]

    while len(path) < k:
        best_delta, best_node, best_position = np.inf, None, None

        for i in range(len(path)):
            a, b = path[i], path[(i + 1) % len(path)]
            delta = D[a] + D[:, b] - D[a][b] + costs

            delta = delta.astype(np.float64)
            delta[visited] = np.inf
            node = int(delta.argmin())

            if delta[node] < best_delta:
                best_delta, best_node, best_position = delta[node], node, i + 1

        visited[best_node] = True
        path.insert(best_position, best_node)

    return path


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "TSPA"
    df = load_data(DATA_DIR / f"{name}.csv")

    D = distance_matrix(df)
    costs = df["cost"].to_numpy(dtype=np.int64)
    n = len(D)

    rng = np.random.default_rng(42)
    methods = {
        "random": lambda start: random_solution(D, costs, rng),
        "nn_end_only": lambda start: nearest_neighbor_path_end_only(D, costs, start),
        "nn_any_position": lambda start: nearest_neighbor_any_position(D, costs, start),
        "greedy_cycle": lambda start: greedy_cycle(D, costs, start),
    }

    for method_name, method in methods.items():
        results = []
        # 200 solutions: one per start node (random ignores the start)
        for start in range(n):
            path = method(start)
            check_solution(path, n)
            results.append((objective(D, costs, path), path))

        values = [value for value, _ in results]
        best_value, best_path = min(results)
        print(f"{name} {method_name}: min={min(values)} max={max(values)} avg={np.mean(values):.1f}")
        print("best path:", best_path)
