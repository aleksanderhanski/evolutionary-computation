import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# helpers.py lives one level up, shared by all labs; lab1 random solutions and the best lab2
# greedy heuristics are the starting solutions, lab3 steepest local search is the baseline
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "lab1-greedy-heuristics"))
sys.path.append(str(ROOT / "lab2-greedy-regret-heuristics"))
sys.path.append(str(ROOT / "lab3-local-search"))

from helpers import DATA_DIR, load_data, distance_matrix, objective, check_solution, plot_solution
from random_solution import random_solution
from nearest_neighbor_any_position_regret import nn_any_position_weighted
from greedy_cycle_regret import greedy_cycle_weighted
from local_search import steepest
from candidate_local_search import candidate_lists, steepest_candidates

LAB_DIR = Path(__file__).resolve().parent
PLOTS_DIR = LAB_DIR / "plots"
RESULTS_DIR = LAB_DIR / "results"
LAB3_RESULTS_DIR = ROOT / "lab3-local-search" / "results"

# best greedy construction heuristic (lowest average objective in lab 2) for each instance,
# the same greedy starting solutions as in lab 3
BEST_GREEDY = {"TSPA": greedy_cycle_weighted, "TSPB": nn_any_position_weighted}

N_CAND = 10
# other numbers of candidate neighbors for the extra experiment
N_CAND_EXPERIMENT = [5, 10, 15, 20]


def run(method, starts, n):
    # one local search per starting solution, only the local search is timed
    results, times, stats = [], [], []
    for start_path in starts:
        t0 = time.perf_counter()
        path, st = method(start_path)
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
    # applied moves and evaluated moves (deltas) per run
    for key in stats[0]:
        summary[f"{key}_avg"] = float(np.mean([st[key] for st in stats]))
    return summary


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "TSPA"
    df = load_data(DATA_DIR / f"{name}.csv")

    D = distance_matrix(df)
    costs = df["cost"].to_numpy(dtype=np.int64)
    n = len(D)

    # 200 random starting solutions, same seed as in lab1 and lab3, so they are the same solutions,
    # and 200 greedy starting solutions, one per start node, as in lab3
    start_rng = np.random.default_rng(42)
    starts = {
        "random": [random_solution(D, costs, start_rng) for _ in range(n)],
        "greedy": [BEST_GREEDY[name](D, costs, start) for start in range(n)],
    }

    cand = candidate_lists(D, costs, N_CAND)
    methods = {
        "ls_steepest_edges": lambda path: steepest(D, costs, path, "edges"),
        "ls_steepest_candidates": lambda path: steepest_candidates(D, costs, path, cand),
    }
    summary = {}
    for start_type, start_paths in starts.items():
        for method_name, method in methods.items():
            summary[f"{method_name}_{start_type}"] = run(method, start_paths, n)

    # candidate moves with other numbers of nearest nodes (random starts only)
    n_cand_results = {}
    for n_cand in N_CAND_EXPERIMENT:
        cand_n = candidate_lists(D, costs, n_cand)
        r = run(lambda path: steepest_candidates(D, costs, path, cand_n), starts["random"], n)
        n_cand_results[n_cand] = {key: value for key, value in r.items() if key != "best_path"}

    for method_name, r in summary.items():
        print(f"{name} {method_name}: min={r['min']} max={r['max']} avg={r['avg']:.1f} "
              f"time={r['time_avg']:.2f}ms ({r['time_min']:.2f}-{r['time_max']:.2f})")
        print("best path:", r["best_path"])
    for n_cand, r in n_cand_results.items():
        print(f"{name} candidates n_cand={n_cand}: min={r['min']} max={r['max']} avg={r['avg']:.1f} "
              f"time={r['time_avg']:.2f}ms")

    # previous methods are taken from the lab3 results (summary tables only)
    with open(LAB3_RESULTS_DIR / f"{name}.json") as f:
        previous = json.load(f)["methods"]

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({"methods": summary, "n_cand": n_cand_results, "previous": previous}, f, indent=1)

    # best solution of each method, one plot per method
    PLOTS_DIR.mkdir(exist_ok=True)
    for method_name, r in summary.items():
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        points = plot_solution(df, r["best_path"], ax, f"{name} {method_name}  (objective {r['min']})")
        colorbar = fig.colorbar(points, ax=ax, shrink=0.6, label="node cost")
        colorbar.outline.set_visible(False)
        fig.suptitle("hollow = not selected, bigger/darker = higher cost", x=0.02, ha="left", fontsize=9)

        plot_path = PLOTS_DIR / f"{name}_{method_name}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print("saved plot:", plot_path)
