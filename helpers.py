import math
from pathlib import Path

import numpy as np
import pandas as pd

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
