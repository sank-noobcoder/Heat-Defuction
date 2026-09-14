"""
Generate synthetic 2D heat diffusion dataset with unknown external sources.

Saves train, val, and test splits as compressed NPZ or NPY archives containing:
- 'u': (N_sequences, seq_len, H, W)
- 's': (N_sequences, seq_len, H, W)
- 'params': simulation constants (alpha, dt, dx)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from phicnet.physics.heat_equation import generate_heat_trajectory


def generate_dataset_split(
    num_sequences: int,
    grid_size: int = 64,
    seq_len: int = 100,
    alpha: float = 0.01,
    dt: float = 0.005,
    seed: int = 42,
    split_name: str = "train"
):
    rng = np.random.default_rng(seed)
    u_all = np.zeros((num_sequences, seq_len, grid_size, grid_size), dtype=np.float32)
    s_all = np.zeros((num_sequences, seq_len, grid_size, grid_size), dtype=np.float32)

    # Distribute across source types: pulsing (50%), static (30%), moving (20%)
    source_choices = ["pulsing", "static", "moving"]
    source_weights = [0.5, 0.3, 0.2]

    print(f"Generating {split_name} split ({num_sequences} sequences of length {seq_len})...")
    for i in tqdm(range(num_sequences), desc=f"Generating {split_name}"):
        stype = rng.choice(source_choices, p=source_weights)
        u_traj, s_traj, _ = generate_heat_trajectory(
            grid_size=grid_size,
            num_steps=seq_len,
            alpha=alpha,
            dt=dt,
            source_type=stype,
            rng=rng
        )
        u_all[i] = u_traj
        s_all[i] = s_traj

    return u_all, s_all


def main():
    parser = argparse.ArgumentParser(description="Generate 2D Heat Diffusion Data for PhICNet")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save generated datasets")
    parser.add_argument("--train_seqs", type=int, default=60, help="Number of training sequences")
    parser.add_argument("--val_seqs", type=int, default=15, help="Number of validation sequences")
    parser.add_argument("--test_seqs", type=int, default=15, help="Number of test sequences")
    parser.add_argument("--seq_len", type=int, default=100, help="Frames per sequence")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size H=W")
    parser.add_argument("--alpha", type=float, default=0.01, help="Diffusivity constant")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step dt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    dx = 1.0 / (args.grid_size - 1)

    splits = [
        ("train", args.train_seqs, args.seed),
        ("val", args.val_seqs, args.seed + 100),
        ("test", args.test_seqs, args.seed + 200),
    ]

    for name, n_seq, seed in splits:
        u_data, s_data = generate_dataset_split(
            num_sequences=n_seq,
            grid_size=args.grid_size,
            seq_len=args.seq_len,
            alpha=args.alpha,
            dt=args.dt,
            seed=seed,
            split_name=name
        )
        filepath = os.path.join(args.data_dir, f"{name}.npz")
        np.savez_compressed(
            filepath,
            u=u_data,
            s=s_data,
            alpha=args.alpha,
            dt=args.dt,
            dx=dx,
            grid_size=args.grid_size
        )
        print(f"Saved {name} data to {filepath} (u: {u_data.shape}, s: {s_data.shape})")

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
