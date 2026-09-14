"""
Training Pipeline for PhICNet and Comparative Baselines.

Supports:
- PhICNet (Physics + ConvRNN Source Model)
- CNNBaseline (Pure spatial ConvNet)
- ConvLSTMBaseline (Pure spatiotemporal recurrent network)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any

from phicnet.models.phicnet import PhICNet, CNNBaseline, ConvLSTMBaseline
from phicnet.training.losses import PhICNetLoss, compute_relative_l2_error


class HeatSequenceDataset(Dataset):
    """
    Subsequence dataset for spatiotemporal sequence training.
    """
    def __init__(self, npz_path: str, subseq_len: int = 20, stride: int = 5):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found at {npz_path}. Run generate_data.py first!")
        
        data = np.load(npz_path)
        self.u = data["u"]  # (N, T, H, W)
        self.s = data["s"]  # (N, T, H, W)
        self.alpha = float(data.get("alpha", 0.01))
        self.dt = float(data.get("dt", 0.005))
        self.dx = float(data.get("dx", 1.0 / 63.0))

        # Create windowed subsequence indices
        self.subseq_len = subseq_len
        self.samples = []
        n_seq, t_steps, h, w = self.u.shape

        for seq_idx in range(n_seq):
            for start_t in range(0, t_steps - subseq_len + 1, stride):
                self.samples.append((seq_idx, start_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq_idx, start_t = self.samples[idx]
        u_clip = self.u[seq_idx, start_t : start_t + self.subseq_len]  # (T, H, W)
        s_clip = self.s[seq_idx, start_t : start_t + self.subseq_len]  # (T, H, W)
        return torch.from_numpy(u_clip).float(), torch.from_numpy(s_clip).float()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    model_type: str = "phicnet",
    unroll_steps: int = 1
) -> Dict[str, float]:
    model.train()
    running_losses = {}

    for u_seq, _ in dataloader:
        u_seq = u_seq.to(device)  # (B, T, H, W)
        optimizer.zero_grad()

        if model_type == "phicnet":
            u_pred, s_pred, _ = model(u_seq, unroll_steps=unroll_steps)
            # Targets: next time step observations
            u_true = u_seq[:, 1:].unsqueeze(2)  # (B, T-1, 1, H, W)
            u_curr = u_seq[:, :-1].unsqueeze(2)  # (B, T-1, 1, H, W)

            loss, metrics = loss_fn(u_pred, u_true, s_pred, u_curr)
        else:
            # Baseline models (pure neural)
            u_pred, s_pred = model(u_seq)
            u_true = u_seq[:, 1:].unsqueeze(2)
            loss = nn.functional.mse_loss(u_pred, u_true)
            rel_l2 = compute_relative_l2_error(u_pred, u_true).item()
            metrics = {"loss_total": loss.item(), "loss_pred": loss.item(), "rel_l2_u": rel_l2}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in metrics.items():
            running_losses[k] = running_losses.get(k, 0.0) + v

    num_batches = max(1, len(dataloader))
    return {k: v / num_batches for k, v in running_losses.items()}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    model_type: str = "phicnet",
    unroll_steps: int = 1
) -> Dict[str, float]:
    model.eval()
    running_losses = {}

    for u_seq, _ in dataloader:
        u_seq = u_seq.to(device)

        if model_type == "phicnet":
            u_pred, s_pred, _ = model(u_seq, unroll_steps=unroll_steps)
            u_true = u_seq[:, 1:].unsqueeze(2)
            u_curr = u_seq[:, :-1].unsqueeze(2)
            _, metrics = loss_fn(u_pred, u_true, s_pred, u_curr)
        else:
            u_pred, s_pred = model(u_seq)
            u_true = u_seq[:, 1:].unsqueeze(2)
            loss = nn.functional.mse_loss(u_pred, u_true)
            rel_l2 = compute_relative_l2_error(u_pred, u_true).item()
            metrics = {"loss_total": loss.item(), "loss_pred": loss.item(), "rel_l2_u": rel_l2}

        for k, v in metrics.items():
            running_losses[k] = running_losses.get(k, 0.0) + v

    num_batches = max(1, len(dataloader))
    return {k: v / num_batches for k, v in running_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Train PhICNet or Baselines")
    parser.add_argument("--model", type=str, default="phicnet", choices=["phicnet", "cnn", "conv_lstm"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subseq_len", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--beta_source", type=float, default=0.1)
    parser.add_argument("--lambda_sparse", type=float, default=0.001)
    parser.add_argument("--unroll_steps", type=int, default=4, help="Multi-step unrolled training horizon to eliminate exposure bias")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load datasets
    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")

    train_dataset = HeatSequenceDataset(train_npz, subseq_len=args.subseq_len, stride=5)
    val_dataset = HeatSequenceDataset(val_npz, subseq_len=args.subseq_len, stride=10)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # Build model
    alpha = train_dataset.alpha
    dt = train_dataset.dt
    dx = train_dataset.dx

    if args.model == "phicnet":
        model = PhICNet(alpha=alpha, dt=dt, dx=dx, hidden_dim=args.hidden_dim).to(device)
        loss_fn = PhICNetLoss(
            beta_source=args.beta_source,
            lambda_sparse=args.lambda_sparse,
            alpha=alpha,
            dt=dt,
            dx=dx
        ).to(device)
    elif args.model == "cnn":
        model = CNNBaseline(hidden_dim=args.hidden_dim).to(device)
        loss_fn = nn.MSELoss()
    elif args.model == "conv_lstm":
        model = ConvLSTMBaseline(hidden_dim=args.hidden_dim).to(device)
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\nTraining model: {args.model.upper()} for {args.epochs} epochs...")
    best_val_loss = float("inf")
    save_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, args.model, unroll_steps=args.unroll_steps)
        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device, args.model, unroll_steps=args.unroll_steps)
        scheduler.step()

        val_total = val_metrics["loss_total"]
        val_rel_l2 = val_metrics["rel_l2_u"]
        val_accuracy = max(0.0, (1.0 - val_rel_l2) * 100.0)

        log_str = (
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_metrics['loss_total']:.5f} | "
            f"Val Loss: {val_total:.5f} | "
            f"Val Acc: {val_accuracy:.2f}% (Error: {val_rel_l2 * 100:.2f}%)"
        )
        if "loss_source" in train_metrics:
            log_str += f" | Src Loss: {train_metrics['loss_source']:.5f}"

        print(log_str, flush=True)

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_total,
                "model_type": args.model,
                "alpha": alpha,
                "dt": dt,
                "dx": dx,
                "hidden_dim": args.hidden_dim
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {best_val_loss:.5f})")

    print(f"\nTraining finished for {args.model}! Best validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()
