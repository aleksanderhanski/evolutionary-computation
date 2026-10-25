import numpy as np

# kinds of moves
INTER = 0  # exchange of a selected node path[i] with a not selected node free[j]
INTRA = 1  # two-nodes or two-edges exchange inside the path, positions i < j


def free_nodes(n, path) -> np.ndarray:
    selected = np.zeros(n, dtype=bool)
    selected[path] = True
    return np.flatnonzero(~selected)


def intra_pairs(k, intra):
    # all position pairs i < j of the intra-route moves
    I, J = np.triu_indices(k, 1)
    if intra == "edges":
        # neighboring edges share a node, exchanging them changes nothing
        keep = (J - I > 1) & ~((I == 0) & (J == k - 1))
        I, J = I[keep], J[keep]
    return I, J


def inter_delta(D, costs, path, free, i, j):
    # path[i] = u is replaced by free[j] = v, edges p-u-nx become p-v-nx
    k = len(path)
    p, u, nx, v = path[i - 1], path[i], path[(i + 1) % k], free[j]
    return costs[v] - costs[u] + D[p][v] + D[v][nx] - D[p][u] - D[u][nx]


def nodes_delta(D, path, i, j):
    # a = path[i] and b = path[j] swap places
    k = len(path)
    a, pa, na = path[i], path[i - 1], path[(i + 1) % k]
    b, pb, nb = path[j], path[j - 1], path[(j + 1) % k]
    delta = (D[pa][b] + D[b][na] + D[pb][a] + D[a][nb]
             - D[pa][a] - D[a][na] - D[pb][b] - D[b][nb])
    if j - i == 1 or (i == 0 and j == k - 1):
        # a and b are neighbors: the edge a-b stays, but the formula above
        # removes it twice and adds two self-loops of length 0 instead
        delta += 2 * D[a][b]
    return delta


def edges_delta(D, path, i, j):
    # edges a-na and b-nb are replaced by a-b and na-nb (2-opt)
    k = len(path)
    a, na, b, nb = path[i], path[i + 1], path[j], path[(j + 1) % k]
    return D[a][b] + D[na][nb] - D[a][na] - D[b][nb]


def apply_move(path, free, kind, i, j, intra):
    if kind == INTER:
        path[i], free[j] = free[j], path[i]
    elif intra == "nodes":
        path[i], path[j] = path[j], path[i]
    else:
        # the part between the two edges is reversed
        path[i + 1:j + 1] = path[i + 1:j + 1][::-1]


def all_moves(k, n_free, intra):
    # one (kind, i, j) per move, the positions stay valid after every applied move
    I, J = intra_pairs(k, intra)
    moves = [(INTER, i, j) for i in range(k) for j in range(n_free)]
    moves += [(INTRA, i, j) for i, j in zip(I.tolist(), J.tolist())]
    return moves


def steepest(D, costs, path, intra):
    # every iteration evaluates the whole neighborhood (inter + intra moves)
    # and applies the best move, until no move improves the solution

    D, costs = D.tolist(), costs.tolist()  # plain lists are much faster to index one by one
    path = list(path)
    free = free_nodes(len(D), path).tolist()
    intra_delta = nodes_delta if intra == "nodes" else edges_delta
    moves = all_moves(len(path), len(free), intra)
    iterations = evaluated = 0

    while True:
        best_delta, best_move = 0, None
        for kind, i, j in moves:
            if kind == INTER:
                delta = inter_delta(D, costs, path, free, i, j)
            else:
                delta = intra_delta(D, path, i, j)

            if delta < best_delta:
                best_delta, best_move = delta, (kind, i, j)
        evaluated += len(moves)

        if best_move is None:  # no improving move: local optimum
            break
        apply_move(path, free, *best_move, intra)
        iterations += 1

    return path, {"iterations": iterations, "evaluated": evaluated}


def greedy(D, costs, path, intra, rng):
    # every iteration browses the whole neighborhood (inter + intra moves mixed together)
    # in a new random order and applies the first improving move,
    # until no move improves the solution

    D, costs = D.tolist(), costs.tolist()  # plain lists are much faster to index one by one
    path = list(path)
    free = free_nodes(len(D), path).tolist()
    intra_delta = nodes_delta if intra == "nodes" else edges_delta
    moves = all_moves(len(path), len(free), intra)
    iterations = evaluated = 0

    improved = True
    while improved:
        improved = False
        # new random order of the whole neighborhood, moves are evaluated lazily one by one
        for m in rng.permutation(len(moves)):
            kind, i, j = moves[m]
            evaluated += 1
            if kind == INTER:
                delta = inter_delta(D, costs, path, free, i, j)
            else:
                delta = intra_delta(D, path, i, j)

            if delta < 0:
                apply_move(path, free, kind, i, j, intra)
                iterations += 1
                improved = True
                break

    return path, {"iterations": iterations, "evaluated": evaluated}
