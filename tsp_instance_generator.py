import numpy as np
import h5py

def euclidean_distance_matrix(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

num_instances = 1000 
num_points = 300                
output_file = "data/tsp_" + str(num_points) + "_" + str(num_instances) + ".h5"

with h5py.File(output_file, "w") as f:
    D = f.create_dataset("D", shape=(num_instances, num_points, num_points), dtype="float32")

    rng = np.random.default_rng(0)

    for i in range(num_instances):
        points = rng.random((num_points, 2))
        D[i] = euclidean_distance_matrix(points)
