import os
import math
import argparse
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from model import AttentionModel, compute_tour_length, assert_permutation


class H5TSPDataset(Dataset):
    def __init__(self, data_dir="data", num_points=20, num_instances=1000, data_file=None):
        if data_file is None:
            fname = os.path.join(data_dir, f"tsp_{num_points}_{num_instances}.h5")
        else:
            fname = data_file
        if os.path.exists(fname):
            with h5py.File(fname, "r") as f:
                self.data = f["D"][:].astype(np.float32)
        else:
            raise FileNotFoundError(
                f"HDF5 dataset not found: {fname}. "
                "Generate the dataset before instantiating H5TSPDataset."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(
    data_dir="data",
    num_points=20,
    num_instances=1000,
    batch_size=128,
    d_model=128,
    n_layers=2,
    n_heads=8,
    lr=1e-4,
    epochs=10,
    distance_metric="euclidean",
    device="cpu",
    save_dir=None,
    resume=None,
    data_file=None,
):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    if save_dir is None:
        save_dir = f"checkpoints_{distance_metric}"
    os.makedirs(save_dir, exist_ok=True)

    dataset = H5TSPDataset(
        data_dir=data_dir,
        num_points=num_points,
        num_instances=num_instances,
        data_file=data_file,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = AttentionModel(d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    best_eval = float("inf")

    # optionally resume
    if resume is not None:
        if os.path.exists(resume):
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if "opt_state" in ckpt and ckpt["opt_state"] is not None:
                opt.load_state_dict(ckpt["opt_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_eval = ckpt.get("best_eval", best_eval)
            print(f"Resumed from checkpoint {resume}, starting at epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"Checkpoint to resume not found: {resume}")

    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()
        losses = []
        lengths_mean = []
        for i, batch in enumerate(loader):
            coords = batch.to(device)
            B = coords.size(0)

            model.train()
            tour, logp_sum = model(coords, decode_type="sample")
            assert_permutation(tour)
            lengths = compute_tour_length(coords, tour, metric=distance_metric)  # (B,)

            # Rollout baseline (greedy policy evaluated with no grad)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                tour_greedy, _ = model(coords, decode_type="greedy")
                assert_permutation(tour_greedy)
                baseline_lengths = compute_tour_length(coords, tour_greedy, metric=distance_metric)
            if was_training:
                model.train()

            adv = (lengths - baseline_lengths).detach()
            loss = (adv * logp_sum).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            lengths_mean.append(lengths.mean().item())

            if (i + 1) % 10 == 0:
                print(
                    f"Ep {ep} it {i+1}/{len(loader)} | loss {np.mean(losses):.6f} | len {np.mean(lengths_mean):.4f}"
                )

        dt = time.time() - t0
        # eval greedy on 100 random instances
        model.eval()
        with torch.no_grad():
            eval_idx = np.random.choice(len(dataset), size=100, replace=False)
            eval_batch = torch.from_numpy(dataset.data[eval_idx]).to(device)
            tour_greedy, _ = model(eval_batch, decode_type="greedy")
            assert_permutation(tour_greedy)
            gre_len = compute_tour_length(eval_batch, tour_greedy, metric=distance_metric).mean().item()
        print("---")
        print(f"Epoch {ep} finished in {dt:.1f}s | train len {np.mean(lengths_mean):.4f} | greedy eval len {gre_len:.4f}")
        print("---")

        # checkpoint
        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "best_eval": best_eval,
        }
        ckpt_path = os.path.join(save_dir, f"ckpt_ep{ep}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # save final model (also save as latest)
    final_path = os.path.join(save_dir, "tsp_attention_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--num_points", type=int, default=20)
    parser.add_argument("--num_instances", type=int, default=1000, help="Number of instances in the HDF5 filename")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--distance_metric", default="euclidean", choices=["euclidean", "manhattan"], help="Tour length metric for training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=None, help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_file", default=None, help="Explicit HDF5 dataset path")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        num_points=args.num_points,
        num_instances=args.num_instances,
        batch_size=args.batch_size,
        epochs=args.epochs,
        distance_metric=args.distance_metric,
        device=args.device,
        save_dir=args.save_dir,
        resume=args.resume,
        data_file=args.data_file,
    )
