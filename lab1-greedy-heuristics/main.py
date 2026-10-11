import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# helpers.py lives one level up, shared by all labs
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helpers import DATA_DIR, load_data, distance_matrix, objective, check_solution, plot_solution
from random_solution import random_solution
from nearest_neighbor_path_end_only import nearest_neighbor_path_end_only
from nearest_neighbor_any_position import nearest_neighbor_any_position
from greedy_cycle import greedy_cycle

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


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

    best = {}
    for method_name, method in methods.items():
        results = []
        # 200 solutions: one per start node (random ignores the start)
        for start in range(n):
            path = method(start)
            check_solution(path, n)
            results.append((objective(D, costs, path), path))

        values = [value for value, _ in results]
        best_value, best_path = min(results)
        best[method_name] = (best_value, best_path)
        print(f"{name} {method_name}: min={min(values)} max={max(values)} avg={np.mean(values):.1f}")
        print("best path:", best_path)

    # best solution of each method, one panel per method
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    for ax, (method_name, (best_value, best_path)) in zip(axes.flat, best.items()):
        points = plot_solution(df, best_path, ax, f"{method_name}  (objective {best_value})")
    colorbar = fig.colorbar(points, ax=axes, shrink=0.6, label="node cost")
    colorbar.outline.set_visible(False)
    fig.suptitle(f"{name}: best solution per method (hollow = not selected, bigger/darker = higher cost)", x=0.02, ha="left")

    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=150)
    print("saved plot:", PLOTS_DIR / f"{name}.png")
