import os
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from greedy import greedy_algorithm
from model import AttentionModel, compute_tour_length
from concorde import ConcordeUnavailable, solve_tsp_concorde


def plot_tour(ax, coords, tour, title=None, show_idx=True, metric="euclidean"):
    # coords: (N,2), tour: (N,) indices
    ordered = coords[tour]
    x = ordered[:, 0]
    y = ordered[:, 1]
    if metric == "manhattan":
        path_x = [x[0]]
        path_y = [y[0]]
        for i in range(1, len(x)):
            path_x.extend([x[i], x[i]])
            path_y.extend([y[i - 1], y[i]])
        path_x.extend([x[0], x[0]])
        path_y.extend([y[-1], y[0]])
        ax.plot(path_x, path_y, '-o')
    else:
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), '-o')
    if show_idx:
        for i, (xi, yi) in enumerate(ordered):
            ax.text(xi, yi, str(i), fontsize=8)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')


def load_instance(data_dir, num_points, idx, data_file=None):
    fname = data_file or os.path.join(data_dir, f"tsp_{num_points}_1000.h5")
    with h5py.File(fname, 'r') as f:
        coords = f['D'][idx]
    return coords


def visualize(model_path, data_dir='data', num_points=20, indices=None, n_samples=3, out_dir='visualizations', device='cpu',
              use_concorde=True, concorde_time_bound=-1, concorde_verbose=False, concorde_scale=1000.0, distance_metric="euclidean",
              data_file=None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    loaded = torch.load(model_path, map_location=device)
    # handle both raw state_dict and checkpoint dicts saved by train
    if isinstance(loaded, dict) and "model_state" in loaded:
        sd = loaded["model_state"]
    else:
        sd = loaded

    # detect legacy checkpoint where decoder.w_c expects 3*d input (includes first)
    include_first = False
    if "decoder.w_c.weight" in sd:
        w = sd["decoder.w_c.weight"]
        in_dim = w.shape[1]
        out_dim = w.shape[0]
        if in_dim == 3 * out_dim:
            include_first = True

    model = AttentionModel(d_model=out_dim if 'out_dim' in locals() else 128, include_first=include_first)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    if indices is None:
        indices = list(range(5))

    for idx in indices:
        coords = load_instance(data_dir, num_points, idx, data_file=data_file)
        coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)

        if distance_metric == "manhattan":
            distance_fn = lambda a, b: np.abs(a - b).sum()
        else:
            distance_fn = None
        tour_greedy_heur, _ = greedy_algorithm(coords, distance_fn=distance_fn)
        greedy_heur_len = compute_tour_length(
            coords_t, torch.from_numpy(tour_greedy_heur).unsqueeze(0), metric=distance_metric
        ).item()

        with torch.no_grad():
            tour_greedy_model, _ = model(coords_t, decode_type='greedy', random_start=False)
            tour_greedy_model = tour_greedy_model[0].cpu().numpy()
            greedy_model_len = compute_tour_length(
                coords_t, torch.from_numpy(tour_greedy_model).unsqueeze(0), metric=distance_metric
            ).item()

            sampled_tours = []
            sampled_lens = []
            for s in range(n_samples):
                tour_s, _ = model(coords_t, decode_type='sample', random_start=False)
                tour_s = tour_s[0].cpu().numpy()
                l = compute_tour_length(
                    coords_t, torch.from_numpy(tour_s).unsqueeze(0), metric=distance_metric
                ).item()
                sampled_tours.append(tour_s)
                sampled_lens.append(l)

        concorde_tour = None
        concorde_len = None
        if use_concorde:
            try:
                concorde_norm = "MAN_2D" if distance_metric == "manhattan" else "EUC_2D"
                tour_c, _ = solve_tsp_concorde(
                    coords,
                    norm=concorde_norm,
                    time_bound=concorde_time_bound,
                    verbose=concorde_verbose,
                    scale=concorde_scale,
                )
                tour_c = tour_c.astype(np.int64)
                concorde_tour = tour_c
                concorde_len = compute_tour_length(
                    coords_t, torch.from_numpy(tour_c).unsqueeze(0), metric=distance_metric
                ).item()
            except ConcordeUnavailable as exc:
                print(f"[concorde] instance {idx}: {exc}")
            except Exception as exc:
                print(f"[concorde] instance {idx}: solve failed ({exc})")

        # create plot
        nplots = 2 + n_samples + (1 if concorde_tour is not None else 0)
        fig, axes = plt.subplots(1, nplots, figsize=(4 * nplots, 4))
        if nplots == 1:
            axes = [axes]
        plot_tour(axes[0], coords, tour_greedy_model, title=f"Model Greedy (len={greedy_model_len:.4f})", metric=distance_metric)
        plot_tour(axes[1], coords, tour_greedy_heur, title=f"Heuristic Greedy (len={greedy_heur_len:.4f})", metric=distance_metric)
        for i, (tour_s, l) in enumerate(zip(sampled_tours, sampled_lens)):
            plot_tour(axes[i+2], coords, tour_s, title=f"Sample {i+1} (len={l:.4f})", metric=distance_metric)
        if concorde_tour is not None:
            plot_tour(
                axes[-1],
                coords,
                concorde_tour,
                title=f"Concorde (len={concorde_len:.4f})",
                metric=distance_metric,
            )

        dataset_tag = ""
        if data_file:
            base = os.path.basename(data_file)
            dataset_tag = "_" + os.path.splitext(base)[0]
        metric_tag = f"_{distance_metric}" if distance_metric != "euclidean" else ""
        out_path = os.path.join(out_dir, f"instance_{idx}{dataset_tag}{metric_tag}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved visualization: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model state dict')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--num_points', type=int, default=20)
    parser.add_argument('--indices', default='0,1,2,3,4', help='Comma-separated instance indices')
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--out_dir', default='visualizations')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no_concorde', action='store_true', help='Disable Concorde solver')
    parser.add_argument('--concorde_time_bound', type=int, default=-1, help='Concorde time bound (-1 for no limit)')
    parser.add_argument('--concorde_verbose', action='store_true', help='Show Concorde output')
    parser.add_argument('--concorde_scale', type=float, default=1000.0, help='Scale coords before Concorde (for EUC_2D rounding)')
    parser.add_argument('--distance_metric', default='euclidean', choices=['euclidean', 'manhattan'], help='Tour length metric')
    parser.add_argument('--data_file', default=None, help='Explicit HDF5 dataset path')
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(',')]
    visualize(
        args.model,
        data_dir=args.data_dir,
        num_points=args.num_points,
        indices=indices,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        device=args.device,
        use_concorde=not args.no_concorde,
        concorde_time_bound=args.concorde_time_bound,
        concorde_verbose=args.concorde_verbose,
        concorde_scale=args.concorde_scale,
        distance_metric=args.distance_metric,
        data_file=args.data_file,
    )
