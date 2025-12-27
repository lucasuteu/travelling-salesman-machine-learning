import numpy as np
import h5py

num_instances = 10000
num_points = 50
MODE = "two_islands"
test = True
output_file = f"{"test_" if test else ""}data/tsp_{"" if MODE == "uniform" else MODE + "_"}{num_points}_{num_instances}.h5"


def sample_points(rng, n, mode="uniform"):
    if mode == "uniform":
        return rng.random((n, 2), dtype=np.float32)

    if mode == "two_islands":
        c1 = np.array([0.25, 0.30], dtype=np.float32)
        c2 = np.array([0.75, 0.70], dtype=np.float32)
        sigma = 0.06

        n1 = n // 3
        n2 = n - n1

        p1 = c1 + sigma * rng.standard_normal((n1, 2)).astype(np.float32)
        p2 = c2 + sigma * rng.standard_normal((n2, 2)).astype(np.float32)

        pts = np.vstack([p1, p2])
        pts = np.clip(pts, 0.0, 1.0)
        rng.shuffle(pts, axis=0)
        return pts

    if mode == "three_columns":
        xs = np.array([0.25, 0.50, 0.75], dtype=np.float32)
        width = 0.04  # column half-width

        n_col = n // 3
        rest = n - 3 * n_col

        pts = []
        for i, x in enumerate(xs):
            ni = n_col + (i < rest)

            x_vals = rng.uniform(x - width, x + width, size=ni)
            y_vals = rng.uniform(0.0, 1.0, size=ni)

            pts.append(np.stack([x_vals, y_vals], axis=1))

        pts = np.vstack(pts).astype(np.float32)
        pts = np.clip(pts, 0.0, 1.0)
        rng.shuffle(pts, axis=0)
        return pts

    raise ValueError(f"Unknown mode: {mode}")


with h5py.File(output_file, "w") as f:
    D = f.create_dataset("D", shape=(num_instances, num_points, 2), dtype="float32")

    rng = np.random.default_rng() if test else np.random.default_rng(0)

    for i in range(num_instances):
        D[i] = sample_points(rng, num_points, MODE)
