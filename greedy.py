import h5py

def greedy_algorithm(D):
    
    n = len(D)
    path = [0] * (n+1)
    used = [False] * n

    used[0] = True

    weight = 0

    for i in range(1, n):
        current = path[i - 1]

        best_dist = float("inf")
        best_j = -1

        for j in range(n):
            if not used[j] and D[current][j] < best_dist:
                best_dist = D[current][j]
                best_j = j

        weight += D[current][best_j]
        path[i] = best_j
        used[best_j] = True

    weight += D[path[n-1]][0]
    return (path, weight)


input_file = "data/tsp_300_1000.h5"
with h5py.File(input_file, "r") as f:
    Dset = f["D"]
    num_instances = Dset.shape[0]

    for i in range(10):
        D = Dset[i]

        print(greedy_algorithm(D)[1])