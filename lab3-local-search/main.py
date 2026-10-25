import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# helpers.py lives one level up, shared by all labs; lab1 and lab2 methods are reused
# for the starting solutions and for the summary
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "lab1-greedy-heuristics"))
sys.path.append(str(ROOT / "lab2-greedy-regret-heuristics"))

from helpers import DATA_DIR, load_data, distance_matrix, objective, check_solution, plot_solution
from random_solution import random_solution
from nearest_neighbor_path_end_only import nearest_neighbor_path_end_only
from nearest_neighbor_any_position import nearest_neighbor_any_position
from greedy_cycle import greedy_cycle
from nearest_neighbor_any_position_regret import nn_any_position_regret, nn_any_position_weighted
from greedy_cycle_regret import greedy_cycle_regret, greedy_cycle_weighted
from local_search import steepest, greedy

LAB_DIR = Path(__file__).resolve().parent
PLOTS_DIR = LAB_DIR / "plots"
RESULTS_DIR = LAB_DIR / "results"

# best greedy construction heuristic (lowest average objective in lab 2) for each instance
BEST_GREEDY = {"TSPA": greedy_cycle_weighted, "TSPB": nn_any_position_weighted}


def run(method, inputs, n):
    # one solution per input, only the call of the method is timed
    results, times, stats = [], [], []
    for x in inputs:
        t0 = time.perf_counter()
        path, st = method(x)
        times.append(time.perf_counter() - t0)
        check_solution(path, n)
        results.append((objective(D, costs, path), path))
        stats.append(st)

    values = [value for value, _ in results]
    best_value, best_path = min(results)
    times_ms = 1000 * np.array(times)
    summary = {"min": min(values), "max": max(values), "avg": float(np.mean(values)),
               "time_min": float(times_ms.min()), "time_max": float(times_ms.max()),
               "time_avg": float(times_ms.mean()), "best_path": [int(v) for v in best_path]}
    # local search only: applied moves and evaluated moves (deltas) per run
    for key in stats[0]:
        summary[f"{key}_avg"] = float(np.mean([st[key] for st in stats]))
    return summary


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "TSPA"
    df = load_data(DATA_DIR / f"{name}.csv")

    D = distance_matrix(df)
    costs = df["cost"].to_numpy(dtype=np.int64)
    n = len(D)

    # previous methods: 200 solutions, one per start node (random ignores the start)
    rng = np.random.default_rng(42)
    constructions = {
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
    for method_name, method in constructions.items():
        summary[method_name] = run(lambda start: (method(start), {}), range(n), n)

    # starting solutions, shared by all local search methods of the same start type
    # (same seed as above, so the random starts are exactly the "random" solutions)
    start_rng = np.random.default_rng(42)
    starts = {
        "random": [random_solution(D, costs, start_rng) for _ in range(n)],
        "greedy": [BEST_GREEDY[name](D, costs, start) for start in range(n)],
    }

    ls_rng = np.random.default_rng(123)
    ls_methods = []
    for start_type in ["random", "greedy"]:
        for intra in ["nodes", "edges"]:
            for ls_type in ["steepest", "greedy"]:
                method_name = f"ls_{ls_type}_{intra}_{start_type}"
                if ls_type == "steepest":
                    method = lambda path: steepest(D, costs, path, intra)
                else:
                    method = lambda path: greedy(D, costs, path, intra, ls_rng)
                summary[method_name] = run(method, starts[start_type], n)
                ls_methods.append(method_name)

    for method_name, r in summary.items():
        print(f"{name} {method_name}: min={r['min']} max={r['max']} avg={r['avg']:.1f} "
              f"time={r['time_avg']:.2f}ms ({r['time_min']:.2f}-{r['time_max']:.2f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({"best_greedy": BEST_GREEDY[name].__name__, "methods": summary}, f, indent=1)

    # best solution of each local search method, one plot per method
    PLOTS_DIR.mkdir(exist_ok=True)
    for method_name in ls_methods:
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
