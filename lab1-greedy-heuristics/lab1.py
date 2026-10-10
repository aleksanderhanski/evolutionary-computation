import sys
import math
from pathlib import Path

import numpy as np

# helpers.py lives one level up, shared by all labs
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helpers import DATA_DIR, load_data, distance_matrix, objective, check_solution


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


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "TSPA"
    df = load_data(DATA_DIR / f"{name}.csv")

    D = distance_matrix(df)
    costs = df["cost"].to_numpy(dtype=np.int64)
    n = len(D)


    results = []
    for start in range(n):
        path = nearest_neighbor_path_end_only(D, costs, start)
        check_solution(path, n)
        results.append((objective(D, costs, path), path))

    values = [value for value, _ in results]
    best_value, best_path = min(results)
    print(f"{name}: min={min(values)} max={max(values)} avg={np.mean(values):.1f}")
    print("best path:", best_path)
