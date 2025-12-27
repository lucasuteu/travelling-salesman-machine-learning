import numpy as np


def greedy_algorithm(coords, distance_fn=None):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n == 0:
        return np.array([], dtype=int), 0.0

    if distance_fn is None:
        def distance_fn(a, b):
            return np.linalg.norm(a - b)

    path = [0]
    used = [False] * n
    used[0] = True
    weight = 0.0

    for _ in range(1, n):
        current = path[-1]
        best_dist = float("inf")
        best_j = -1
        for j in range(n):
            if not used[j]:
                d = distance_fn(coords[current], coords[j])
                if d < best_dist:
                    best_dist = d
                    best_j = j
        weight += best_dist
        path.append(best_j)
        used[best_j] = True

    weight += distance_fn(coords[path[-1]], coords[path[0]])
    return np.array(path, dtype=int), weight
