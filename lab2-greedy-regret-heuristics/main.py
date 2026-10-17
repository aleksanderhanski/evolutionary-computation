import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# helpers.py lives one level up, shared by all labs; lab1 methods are reused for the summary
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "lab1-greedy-heuristics"))

from helpers import DATA_DIR, load_data, distance_matrix, objective, check_solution, plot_solution
from random_solution import random_solution
from nearest_neighbor_path_end_only import nearest_neighbor_path_end_only
from nearest_neighbor_any_position import nearest_neighbor_any_position
from greedy_cycle import greedy_cycle
from nearest_neighbor_any_position_regret import nn_any_position_regret, nn_any_position_weighted
from greedy_cycle_regret import greedy_cycle_regret, greedy_cycle_weighted

LAB_DIR = Path(__file__).resolve().parent
PLOTS_DIR = LAB_DIR / "plots"
RESULTS_DIR = LAB_DIR / "results"

LAB2_METHODS = ["nn_any_position_regret", "nn_any_position_weighted",
                "greedy_cycle_regret", "greedy_cycle_weighted"]

# (w_regret, w_best) pairs for the extra weight experiment
WEIGHTS = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1)]


def run(method, n):
    # 200 solutions: one per start node (random ignores the start)
    results = []
    for start in range(n):
        path = method(start)
        check_solution(path, n)
        results.append((objective(D, costs, path), path))
    values = [value for value, _ in results]
    best_value, best_path = min(results)
    return {"min": min(values), "max": max(values), "avg": float(np.mean(values)),
            "best_path": [int(v) for v in best_path]}


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
        "nn_any_position_regret": lambda start: nn_any_position_regret(D, costs, start),
        "nn_any_position_weighted": lambda start: nn_any_position_weighted(D, costs, start),
        "greedy_cycle_regret": lambda start: greedy_cycle_regret(D, costs, start),
        "greedy_cycle_weighted": lambda start: greedy_cycle_weighted(D, costs, start),
    }

    summary = {}
    for method_name, method in methods.items():
        summary[method_name] = run(method, n)
        r = summary[method_name]
        print(f"{name} {method_name}: min={r['min']} max={r['max']} avg={r['avg']:.1f}")
        print("best path:", r["best_path"])

    # weighted sum criterion with other weights
    weights = {}
    for w_regret, w_best in WEIGHTS:
        for base, fn in [("nn_any_position", nn_any_position_weighted),
                         ("greedy_cycle", greedy_cycle_weighted)]:
            key = f"{base}_w{w_regret}_{w_best}"
            r = run(lambda start: fn(D, costs, start, w_regret, w_best), n)
            weights[key] = {k: r[k] for k in ("min", "max", "avg")}
            print(f"{name} {key}: min={r['min']} max={r['max']} avg={r['avg']:.1f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({"methods": summary, "weights": weights}, f, indent=1)

    # best solution of each new method, one plot per method
    PLOTS_DIR.mkdir(exist_ok=True)
    for method_name in LAB2_METHODS:
        best_value = summary[method_name]["min"]
        best_path = summary[method_name]["best_path"]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        points = plot_solution(df, best_path, ax, f"{name} {method_name}  (objective {best_value})")
        colorbar = fig.colorbar(points, ax=ax, shrink=0.6, label="node cost")
        colorbar.outline.set_visible(False)
        fig.suptitle("hollow = not selected, bigger/darker = higher cost", x=0.02, ha="left", fontsize=9)

        plot_path = PLOTS_DIR / f"{name}_{method_name}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print("saved plot:", plot_path)
