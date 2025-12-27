import os
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import AttentionModel, compute_tour_length


def evaluate(model_path, data_dir='data', num_points=20, num_instances=1000, batch_size=128, device='cpu', n_samples=1, distance_metric="euclidean"):
    device = torch.device(device)
    # load dataset
    fname = os.path.join(data_dir, f"tsp_{num_points}_{num_instances}.h5")
    with h5py.File(fname, 'r') as f:
        data = f['D'][:].astype(np.float32)

    dataset = torch.from_numpy(data)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = AttentionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    greedy_lengths = []
    sample_lengths = []

    with torch.no_grad():
        for batch in loader:
            coords = batch.to(device)
            # greedy
            tour_g, _ = model(coords, decode_type='greedy', random_start=False)
            gre_len = compute_tour_length(coords, tour_g, metric=distance_metric)
            greedy_lengths.append(gre_len.cpu().numpy())

            # samples (one sample per instance by default; can be increased)
            if n_samples > 0:
                # average sampled length across n_samples
                acc = 0.0
                for _ in range(n_samples):
                    tour_s, _ = model(coords, decode_type='sample', random_start=False)
                    acc += compute_tour_length(coords, tour_s, metric=distance_metric)
                sample_lengths.append((acc / n_samples).cpu().numpy())

    greedy_lengths = np.concatenate(greedy_lengths)
    sample_lengths = np.concatenate(sample_lengths) if n_samples > 0 else None

    print(f"Greedy: mean={greedy_lengths.mean():.4f} std={greedy_lengths.std():.4f} n={len(greedy_lengths)}")
    if sample_lengths is not None:
        print(f"Sample (avg {n_samples}): mean={sample_lengths.mean():.4f} std={sample_lengths.std():.4f} n={len(sample_lengths)}")

    return greedy_lengths, sample_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--num_points', type=int, default=20)
    parser.add_argument('--num_instances', type=int, default=1000, help='Number of instances in the HDF5 filename')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_samples', type=int, default=1, help='Number of sampled tours to average per instance')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--distance_metric', default='euclidean', choices=['euclidean', 'manhattan'])
    args = parser.parse_args()

    evaluate(
        args.model,
        data_dir=args.data_dir,
        num_points=args.num_points,
        num_instances=args.num_instances,
        batch_size=args.batch_size,
        device=args.device,
        n_samples=args.n_samples,
        distance_metric=args.distance_metric,
    )
