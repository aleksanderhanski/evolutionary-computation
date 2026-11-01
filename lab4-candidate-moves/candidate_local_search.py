import numpy as np

# kinds of moves
INTER = 0  # node path[i] is replaced by a not selected node v
INTRA = 1  # two-edges exchange of edges (path[i], path[i+1]) and (path[j], path[j+1]), i < j


def candidate_lists(D, costs, n_cand=10) -> list[list[int]]:
    # for each node u the n_cand other nodes v with the smallest D[u][v] + cost[v]
    score = (D + costs[None, :]).astype(np.float64)
    np.fill_diagonal(score, np.inf)
    return np.argsort(score, axis=1, kind="stable")[:, :n_cand].tolist()


def inter_delta(D, costs, path, i, v):
    # path[i] = u is replaced by v, edges p-u-nx become p-v-nx
    k = len(path)
    p, u, nx = path[i - 1], path[i], path[(i + 1) % k]
    return costs[v] - costs[u] + D[p][v] + D[v][nx] - D[p][u] - D[u][nx]


def edges_delta(D, path, i, j):
    # edges a-na and b-nb are replaced by a-b and na-nb (2-opt)
    k = len(path)
    a, na, b, nb = path[i], path[i + 1], path[j], path[(j + 1) % k]
    return D[a][b] + D[na][nb] - D[a][na] - D[b][nb]


def candidate_moves(path, pos, cand):
    # yields only the moves that introduce at least one candidate edge u-v,
    # where u is selected and v is one of the nearest nodes of u
    k = len(path)
    for i, u in enumerate(path):
        for v in cand[u]:
            j = pos[v]
            if j >= 0:
                # v selected: two 2-opt moves add the edge u-v
                # 1) remove u-next(u) and v-next(v), add u-v and next(u)-next(v)
                # 2) remove prev(u)-u and prev(v)-v, add prev(u)-prev(v) and u-v
                for a, b in ((i, j), ((i - 1) % k, (j - 1) % k)):
                    a, b = min(a, b), max(a, b)
                    # neighboring edges share a node, exchanging them changes nothing
                    if b - a > 1 and not (a == 0 and b == k - 1):
                        yield INTRA, a, b
            else:
                # v not selected: v replaces prev(u) (adds v-u) or next(u) (adds u-v)
                yield INTER, (i - 1) % k, v
                yield INTER, (i + 1) % k, v


def apply_move(path, pos, kind, i, x):
    # pos[node] = position of the node in path, -1 if not selected
    if kind == INTER:
        pos[path[i]] = -1
        path[i] = x
        pos[x] = i
    else:
        # the part between the two edges is reversed
        path[i + 1:x + 1] = path[i + 1:x + 1][::-1]
        for t in range(i + 1, x + 1):
            pos[path[t]] = t


def steepest_candidates(D, costs, path, cand):
    # every iteration evaluates only the candidate moves (inter + two-edges exchange)
    # and applies the best one, until no candidate move improves the solution

    D, costs = D.tolist(), costs.tolist()  # plain lists are much faster to index one by one
    path = list(path)
    pos = [-1] * len(D)
    for i, u in enumerate(path):
        pos[u] = i
    iterations = evaluated = 0

    while True:
        best_delta, best_move = 0, None
        for kind, i, x in candidate_moves(path, pos, cand):
            if kind == INTER:
                delta = inter_delta(D, costs, path, i, x)
            else:
                delta = edges_delta(D, path, i, x)
            evaluated += 1

            if delta < best_delta:
                best_delta, best_move = delta, (kind, i, x)

        if best_move is None:  # no improving candidate move
            break
        apply_move(path, pos, *best_move)
        iterations += 1

    return path, {"iterations": iterations, "evaluated": evaluated}
