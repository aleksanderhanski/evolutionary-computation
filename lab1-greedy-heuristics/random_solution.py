import math


def random_solution(D, costs, rng) -> list[int]:
    # k distinct nodes, already in random order
    n = len(D)
    k = math.ceil(n / 2)
    return [int(node) for node in rng.choice(n, size=k, replace=False)]
