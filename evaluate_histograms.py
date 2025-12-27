import argparse
import os
import re

import h5py
import numpy as np
import torch

from concorde import ConcordeUnavailable, solve_tsp_concorde
from greedy import greedy_algorithm
from model import AttentionModel, compute_tour_length


def load_model(model_path, device):
    loaded = torch.load(model_path, map_location=device)
    if isinstance(loaded, dict) and "model_state" in loaded:
        sd = loaded["model_state"]
    else:
        sd = loaded

    include_first = False
    out_dim = 128
    if "decoder.w_c.weight" in sd:
        w = sd["decoder.w_c.weight"]
        out_dim = w.shape[0]
        if w.shape[1] == 3 * out_dim:
            include_first = True

    model = AttentionModel(d_model=out_dim, include_first=include_first)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def iter_datasets(test_data_dir):
    for fname in sorted(os.listdir(test_data_dir)):
        if fname.endswith(".h5"):
            yield os.path.join(test_data_dir, fname)


def get_all_checkpoint_paths(ckpt_dir):
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if re.match(r"ckpt_ep\\d+\\.pt$", f)
    ]
    if ckpts:
        return sorted(ckpts)
    final = os.path.join(ckpt_dir, "tsp_attention_model.pt")
    return [final] if os.path.exists(final) else []


def dataset_tag_from_checkpoint_dir(ckpt_dir):
    base = os.path.basename(ckpt_dir)
    if base.startswith("checkpoints_") and base != "checkpoints_manhattan":
        return base[len("checkpoints_") :]
    return None


def filter_datasets_by_names(datasets, names):
    if not names:
        return datasets
    return [d for d in datasets if os.path.splitext(os.path.basename(d))[0] in names]


def evaluate_methods(
    coords_np,
    model,
    device,
    metric,
    best_k,
    batch_size,
    concorde_scale,
    do_concorde,
):
    if metric == "manhattan":
        distance_fn = lambda a, b: np.abs(a - b).sum()
        concorde_norm = "MAN_2D"
    else:
        distance_fn = None
        concorde_norm = "EUC_2D"

    model_greedy = []
    model_best = []
    heuristic_greedy = []
    concorde = []
    concorde_failed = False

    coords_t = torch.from_numpy(coords_np).to(device)
    n_total = coords_np.shape[0]
    for start in range(0, n_total, batch_size):
        batch = coords_t[start : start + batch_size]
        with torch.no_grad():
            tour_g, _ = model(batch, decode_type="greedy", random_start=False)
            lengths_g = compute_tour_length(batch, tour_g, metric=metric).cpu().numpy()
            model_greedy.append(lengths_g)

            best_lengths = None
            for _ in range(best_k):
                tour_s, _ = model(batch, decode_type="sample", random_start=False)
                lengths_s = compute_tour_length(batch, tour_s, metric=metric).cpu().numpy()
                if best_lengths is None:
                    best_lengths = lengths_s
                else:
                    best_lengths = np.minimum(best_lengths, lengths_s)
            model_best.append(best_lengths)

    for coords in coords_np:
        tour_h, _ = greedy_algorithm(coords, distance_fn=distance_fn)
        length_h = compute_tour_length(
            torch.from_numpy(coords).unsqueeze(0),
            torch.from_numpy(tour_h).unsqueeze(0),
            metric=metric,
        ).item()
        heuristic_greedy.append(length_h)

        if do_concorde and not concorde_failed:
            try:
                tour_c, _ = solve_tsp_concorde(
                    coords,
                    norm=concorde_norm,
                    scale=concorde_scale,
                    verbose=False,
                )
                length_c = compute_tour_length(
                    torch.from_numpy(coords).unsqueeze(0),
                    torch.from_numpy(tour_c.astype(np.int64)).unsqueeze(0),
                    metric=metric,
                ).item()
                concorde.append(length_c)
            except ConcordeUnavailable:
                concorde_failed = True
            except Exception:
                concorde_failed = True

    model_greedy = np.concatenate(model_greedy)
    model_best = np.concatenate(model_best)
    heuristic_greedy = np.array(heuristic_greedy)
    concorde = None if concorde_failed or not concorde else np.array(concorde)

    return model_greedy, model_best, heuristic_greedy, concorde


def save_eval_data(out_path, payload):
    np.savez_compressed(out_path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", default="test_data")
    parser.add_argument(
        "--checkpoints_dirs",
        default="checkpoints,checkpoints_manhattan,checkpoints_two_islands_50,checkpoints_three_columns_50",
    )
    parser.add_argument("--out_dir", default="hist_data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--best_k", type=int, default=50)
    parser.add_argument("--max_instances", type=int, default=500)
    parser.add_argument("--concorde_scale", type=float, default=10000.0)
    parser.add_argument("--strict_dataset_match", action="store_true", help="Only evaluate datasets matching checkpoint dir tag")
    parser.add_argument("--core_datasets", default="tsp_20_10000,tsp_50_10000", help="Comma-separated dataset basenames for core checkpoints")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt_dirs = [d.strip() for d in args.checkpoints_dirs.split(",") if d.strip()]
    core_names = [n.strip() for n in args.core_datasets.split(",") if n.strip()]
    datasets = list(iter_datasets(args.test_data_dir))
    if not datasets:
        raise FileNotFoundError(f"No .h5 files found in {args.test_data_dir}")

    for ckpt_dir in ckpt_dirs:
        if not os.path.isdir(ckpt_dir):
            print(f"[skip] checkpoint dir not found: {ckpt_dir}")
            continue

        metric = "manhattan" if "manhattan" in os.path.basename(ckpt_dir) else "euclidean"
        ckpt_paths = get_all_checkpoint_paths(ckpt_dir)
        if not ckpt_paths:
            print(f"[skip] no checkpoints found in {ckpt_dir}")
            continue

        tag = dataset_tag_from_checkpoint_dir(ckpt_dir)
        base = os.path.basename(ckpt_dir)
        if base in ("checkpoints", "checkpoints_manhattan"):
            datasets_to_use = filter_datasets_by_names(datasets, core_names)
        elif args.strict_dataset_match and tag is not None:
            datasets_to_use = [d for d in datasets if tag in os.path.basename(d)]
        else:
            datasets_to_use = datasets

        ckpt_scores = []
        for ckpt_path in ckpt_paths:
            model = load_model(ckpt_path, device=device)
            greedy_means = []
            best_means = []
            for dataset_path in datasets_to_use:
                with h5py.File(dataset_path, "r") as f:
                    data = f["D"][:].astype(np.float32)

                rng = np.random.default_rng(0)
                if args.max_instances is not None and args.max_instances < len(data):
                    idx = rng.choice(len(data), size=args.max_instances, replace=False)
                    data = data[idx]

                model_greedy, model_best, _, _ = evaluate_methods(
                    data,
                    model,
                    device,
                    metric=metric,
                    best_k=args.best_k,
                    batch_size=args.batch_size,
                    concorde_scale=args.concorde_scale,
                    do_concorde=False,
                )
                greedy_means.append(model_greedy.mean())
                best_means.append(model_best.mean())

            ckpt_scores.append(
                {
                    "path": ckpt_path,
                    "greedy_mean": float(np.mean(greedy_means)),
                    "best_mean": float(np.mean(best_means)),
                }
            )

        best_greedy = min(ckpt_scores, key=lambda x: x["greedy_mean"])
        best_best = min(ckpt_scores, key=lambda x: x["best_mean"])

        for mode_label, selected in [("greedy", best_greedy), ("bestk", best_best)]:
            ckpt_path = selected["path"]
            model = load_model(ckpt_path, device=device)
            for dataset_path in datasets_to_use:
                with h5py.File(dataset_path, "r") as f:
                    data = f["D"][:].astype(np.float32)

                rng = np.random.default_rng(0)
                if args.max_instances is not None and args.max_instances < len(data):
                    idx = rng.choice(len(data), size=args.max_instances, replace=False)
                    data = data[idx]

                model_greedy, model_best, heuristic_greedy, concorde = evaluate_methods(
                    data,
                    model,
                    device,
                    metric=metric,
                    best_k=args.best_k,
                    batch_size=args.batch_size,
                    concorde_scale=args.concorde_scale,
                    do_concorde=True,
                )

                payload = {
                    "model_greedy": model_greedy,
                    "model_best_k": model_best,
                    "heuristic_greedy": heuristic_greedy,
                    "concorde": np.array([]) if concorde is None else concorde,
                    "metric": np.array(metric),
                    "best_k": np.array(args.best_k),
                    "checkpoint": np.array(ckpt_path),
                    "selection_mode": np.array(mode_label),
                    "dataset": np.array(dataset_path),
                    "max_instances": np.array(-1 if args.max_instances is None else args.max_instances),
                }

                ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
                data_name = os.path.splitext(os.path.basename(dataset_path))[0]
                out_name = f"{ckpt_dir}_{mode_label}_{ckpt_name}_{data_name}_{metric}.npz".replace(os.sep, "_")
                out_path = os.path.join(args.out_dir, out_name)
                save_eval_data(out_path, payload)
                print(f"Saved data: {out_path}")


if __name__ == "__main__":
    main()
