import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data(path):
    return pd.read_csv(path, sep=";", header=None, names=["x", "y", "cost"])


def distance_matrix(df) -> np.ndarray:
    xy = df[["x", "y"]].to_numpy()
    n = len(xy)
    D = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            dist = math.sqrt((xy[i][0] - xy[j][0])**2 + (xy[i][1] - xy[j][1])**2)
            # round() does banker's rounding, we need mathematical rounding
            D[i][j] = math.floor(dist + 0.5)
    return D


def objective(D, costs, path) -> int:
    path = np.asarray(path)
    next_nodes = np.roll(path, -1) #shifts array one place to the left
    sum_edges = D[path, next_nodes].sum()
    return int(sum_edges + costs[path].sum())


def check_solution(path, n):
    assert len(path) == math.ceil(n / 2), "wrong number of nodes"
    assert len(set(path)) == len(path), "node visited twice"


# one-hue ramp for node cost: light = cheap, dark = expensive
# (starts at a mid step so cheap nodes stay visible on white)
COST_CMAP = LinearSegmentedColormap.from_list(
    "cost", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def plot_solution(df, path, ax, title=""):
    xy = df[["x", "y"]].to_numpy()
    cost = df["cost"].to_numpy()
    selected = np.zeros(len(df), dtype=bool)
    selected[path] = True

    # closed cycle: repeat the first node at the end
    cycle = list(path) + [path[0]]
    ax.plot(xy[cycle, 0], xy[cycle, 1], color="#52514e", linewidth=1, zorder=1)

    # size by cost for every node, same scale for every plot of this instance
    size = 15 + 75 * (cost - cost.min()) / (cost.max() - cost.min())

    # nodes not in the solution: hollow grey
    ax.scatter(xy[~selected, 0], xy[~selected, 1], s=size[~selected], facecolors="none",
               edgecolors="#b5b3ad", linewidths=0.8, zorder=2)

    # selected nodes: also colored by cost
    points = ax.scatter(xy[selected, 0], xy[selected, 1], c=cost[selected],
                        cmap=COST_CMAP, vmin=cost.min(), vmax=cost.max(),
                        s=size[selected], edgecolors="white", linewidths=0.8, zorder=3)

    ax.set_title(title, fontsize=10, loc="left")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, colors="#52514e")
    for spine in ax.spines.values():
        spine.set_color("#d6d4cf")
    return points
