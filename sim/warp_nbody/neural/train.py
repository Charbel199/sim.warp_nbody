import argparse
import pathlib
import time

import numpy as np
import h5py
import torch
import torch_cluster
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data

from .model import NBodyGNN


class NBodyDataset(Dataset):

    def __init__(self, h5_path: str, radius: float = 5.0):
        self._h5_path = h5_path
        self._radius = radius
        with h5py.File(h5_path, "r") as f:
            self.positions = f["positions"][:]
            self.velocities = f["velocities"][:]
            self.masses = f["masses"][:]
            self.accelerations = f["accelerations"][:]

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, idx: int) -> Data:
        pos = torch.tensor(self.positions[idx], dtype=torch.float32)
        vel = torch.tensor(self.velocities[idx], dtype=torch.float32)
        mass = torch.tensor(self.masses[idx], dtype=torch.float32)
        acc = torch.tensor(self.accelerations[idx], dtype=torch.float32)

        return Data(
            pos=pos,
            vel=vel,
            mass=mass,
            y=acc,
            num_nodes=pos.shape[0],
        )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    print(f"[train] Loading dataset from {args.data} ...")
    t0 = time.time()
    dataset = NBodyDataset(args.data, radius=args.radius)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"[train] Dataset loaded in {time.time() - t0:.1f}s — {n_total} frames ({n_train} train / {n_val} val)")

    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[train] Batches per epoch: {len(train_loader)} train, {len(val_loader)} val (batch_size={args.batch_size})")

    model = NBodyGNN(cutoff=args.radius).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: {n_params:,} parameters, cutoff={args.radius}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    train_start = time.time()

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Epoch Time':>10} | {'Note':>20}")
    print("-" * 82)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_acc = model(batch.pos, batch.vel, batch.mass, batch=batch.batch)
            loss = criterion(pred_acc, batch.y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch.num_graphs
            train_count += batch.num_graphs

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_acc = model(batch.pos, batch.vel, batch.mass, batch=batch.batch)
                loss = criterion(pred_acc, batch.y)
                val_loss_sum += loss.item() * batch.num_graphs
                val_count += batch.num_graphs

        scheduler.step()

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        note = ""
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss < float("inf") else 0
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(model.state_dict(), output_dir / "model_best.pt")
            note = f"* BEST (-{improvement:.1f}%)" if improvement > 0 else "* BEST (first)"
        else:
            epochs_since_best += 1
            if epochs_since_best >= 10:
                note = f"no improve x{epochs_since_best}"

        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {lr:>10.6f} | {epoch_time:>8.1f}s | {note:>20}")

    total_time = time.time() - train_start
    print(f"\n[train] Done in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"[train] Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"[train] Checkpoint: {output_dir / 'model_best.pt'}")

    _inference_benchmark(model, device)


def _inference_benchmark(model: NBodyGNN, device: torch.device) -> None:
    print("\n--- Inference Timing Benchmark ---")
    print(f"{'N':>8} | {'Time (ms)':>12}")
    print("-" * 26)

    model.eval()
    for n in [1_000, 10_000, 100_000]:
        pos = torch.randn(n, 3, device=device)
        vel = torch.randn(n, 3, device=device)
        mass = torch.ones(n, 1, device=device)

        with torch.no_grad():
            for _ in range(10):
                model(pos, vel, mass)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start.record()
            for _ in range(10):
                model(pos, vel, mass)
            end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 10.0
        print(f"{n:>8} | {elapsed_ms:>12.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
