"""
Quantitative Benchmark Evaluation for PhICNet and Comparative Baselines.

Compares:
1. Physics-Only Baseline
2. Pure CNN Baseline
3. Pure ConvLSTM Baseline
4. PhICNet (Physics-Incorporated ConvRNN)

Metrics:
- 1-step Relative L2 Error (RMSE / Norm)
- 1-step MSE & MAE
- Multi-step rollout forecasting error (T = 5, 10, 20 steps)
- Source recovery error & spatial correlation
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from phicnet.models.phicnet import PhICNet, PhysicsOnlyBaseline, CNNBaseline, ConvLSTMBaseline
from phicnet.training.losses import compute_relative_l2_error


def load_model_checkpoint(model_type: str, checkpoint_path: str, alpha: float, dt: float, dx: float, device: torch.device):
    if model_type == "physics_only":
        return PhysicsOnlyBaseline(alpha=alpha, dt=dt, dx=dx).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 32)

    if model_type == "phicnet":
        model = PhICNet(alpha=alpha, dt=dt, dx=dx, hidden_dim=hidden_dim).to(device)
    elif model_type == "cnn":
        model = CNNBaseline(hidden_dim=hidden_dim).to(device)
    elif model_type == "conv_lstm":
        model = ConvLSTMBaseline(hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_model_on_data(
    model: torch.nn.Module,
    model_type: str,
    u_data: np.ndarray,
    s_data: np.ndarray,
    device: torch.device,
    rollout_horizons: List[int] = [5, 10, 20],
    warmup_steps: int = 5
) -> Dict[str, Any]:
    model.eval()
    n_seq, t_steps, h, w = u_data.shape

    one_step_rel_l2_list = []
    one_step_mse_list = []
    source_mae_list = []
    rollout_errors = {h_step: [] for h_step in rollout_horizons}

    for i in range(n_seq):
        u_seq = torch.from_numpy(u_data[i:i+1]).float().to(device)  # (1, T, H, W)
        s_seq = torch.from_numpy(s_data[i:i+1]).float().to(device)  # (1, T, H, W)

        # 1-step evaluation
        if model_type == "phicnet":
            u_pred, s_pred, _ = model(u_seq)
            # Source evaluation
            s_true = s_seq[:, :t_steps-1].unsqueeze(2)
            src_mae = F.l1_loss(s_pred, s_true).item()
            source_mae_list.append(src_mae)
        else:
            u_pred, s_pred = model(u_seq)

        u_true = u_seq[:, 1:].unsqueeze(2)
        rel_l2 = compute_relative_l2_error(u_pred, u_true).item()
        mse = F.mse_loss(u_pred, u_true).item()
        one_step_rel_l2_list.append(rel_l2)
        one_step_mse_list.append(mse)

        # Multi-step autoregressive rollout evaluation
        max_horizon = max(rollout_horizons)
        if t_steps > warmup_steps + max_horizon:
            u_warmup = u_seq[:, :warmup_steps]
            if model_type == "phicnet":
                rollout_u, _ = model.rollout_autoregressive(u_warmup, rollout_steps=max_horizon)
            elif model_type in ["physics_only", "cnn", "conv_lstm"]:
                rollout_u = model.rollout_autoregressive(u_warmup, rollout_steps=max_horizon)

            target_future = u_seq[:, warmup_steps : warmup_steps + max_horizon].unsqueeze(2)
            for h_step in rollout_horizons:
                h_pred = rollout_u[:, :h_step]
                h_true = target_future[:, :h_step]
                h_rel_l2 = compute_relative_l2_error(h_pred, h_true).item()
                rollout_errors[h_step].append(h_rel_l2)

    results = {
        "one_step_rel_l2": np.mean(one_step_rel_l2_list),
        "one_step_mse": np.mean(one_step_mse_list),
        "source_mae": np.mean(source_mae_list) if source_mae_list else None,
        "rollout_errors": {h: np.mean(rollout_errors[h]) for h in rollout_horizons}
    }
    return results


def evaluate_models_on_test_set(
    test_npz_path: str = "data/test.npz",
    checkpoint_dir: str = "checkpoints"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_data = np.load(test_npz_path)
    u_test = test_data["u"]
    s_test = test_data["s"]
    alpha = float(test_data.get("alpha", 0.01))
    dt = float(test_data.get("dt", 0.005))
    dx = float(test_data.get("dx", 1.0 / 63.0))

    models_to_test = [
        ("physics_only", "Physics Only (S=0)"),
        ("cnn", "Pure CNN Baseline"),
        ("conv_lstm", "Pure ConvLSTM Baseline"),
        ("phicnet", "PhICNet (Ours)"),
    ]

    summary_table = []
    horizons = [5, 10, 20]

    for m_type, m_name in models_to_test:
        ckpt_path = os.path.join(checkpoint_dir, f"{m_type}_best.pt")
        if m_type != "physics_only" and not os.path.exists(ckpt_path):
            print(f"Skipping {m_name} (no checkpoint found at {ckpt_path})")
            continue

        print(f"\nEvaluating: {m_name}...")
        model = load_model_checkpoint(m_type, ckpt_path, alpha, dt, dx, device)
        res = evaluate_model_on_data(model, m_type, u_test, s_test, device, rollout_horizons=horizons)

        summary_table.append({
            "model": m_name,
            "rel_l2": res["one_step_rel_l2"],
            "mse": res["one_step_mse"],
            "source_mae": res["source_mae"],
            "rollout_5": res["rollout_errors"].get(5, 0.0),
            "rollout_10": res["rollout_errors"].get(10, 0.0),
            "rollout_20": res["rollout_errors"].get(20, 0.0),
        })

    # Print markdown table
    print("\n" + "=" * 95)
    print("                              BENCHMARK EVALUATION RESULTS")
    print("=" * 95)
    header = f"| {'Model':<25} | {'Accuracy (%)':<12} | {'Rel L2 Error':<13} | {'1-Step MSE':<11} | {'Rollout 5-step':<14} | {'Rollout 10-step':<15} | {'Source MAE':<10} |"
    sep = f"|{'-'*27}|{'-'*14}|{'-'*15}|{'-'*13}|{'-'*16}|{'-'*17}|{'-'*12}|"
    print(header)
    print(sep)
    for row in summary_table:
        src_str = f"{row['source_mae']:.4f}" if row['source_mae'] is not None else "N/A"
        acc = max(0.0, (1.0 - row['rel_l2']) * 100.0)
        line = (
            f"| {row['model']:<25} "
            f"| {acc:>10.2f}% "
            f"| {row['rel_l2'] * 100:>11.2f}% "
            f"| {row['mse']:>11.6f} "
            f"| {row['rollout_5'] * 100:>12.2f}% "
            f"| {row['rollout_10'] * 100:>13.2f}% "
            f"| {src_str:>10} |"
        )
        print(line)
    print("=" * 95)

    return summary_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhICNet and Baselines")
    parser.add_argument("--test_data", type=str, default="data/test.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    evaluate_models_on_test_set(args.test_data, args.checkpoint_dir)


if __name__ == "__main__":
    main()
