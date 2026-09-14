"""
Visualization and Animation Generator for PhICNet.

Implements the 4-quadrant and multi-panel comparison requested in Section 35 of PhICNet notes:
[ Real Temperature U(t) | PhICNet Prediction | Physics Prediction | Estimated Source | Absolute Error ]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from phicnet.models.phicnet import PhICNet, PhysicsOnlyBaseline


def plot_comparison_snapshot(
    u_real: np.ndarray,
    u_pred: np.ndarray,
    u_phy: np.ndarray,
    s_est: np.ndarray,
    s_real: np.ndarray,
    step_idx: int,
    save_path: str = "visualization_snapshot.png"
):
    """
    Generate high-resolution 5-panel comparative snapshot at a specific time step.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), dpi=200)

    # Set common temperature colorbar bounds
    vmax_u = max(u_real.max(), u_pred.max(), u_phy.max(), 0.1)
    vmin_u = 0.0

    # 1. Ground Truth Temperature
    im0 = axes[0].imshow(u_real, cmap="inferno", vmin=vmin_u, vmax=vmax_u, origin="lower")
    axes[0].set_title(f"True State $U(t)$ [t={step_idx}]", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Physics-Only Prediction (Diffusive, missing source)
    im1 = axes[1].imshow(u_phy, cmap="inferno", vmin=vmin_u, vmax=vmax_u, origin="lower")
    axes[1].set_title("Physics Only Prediction", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. PhICNet Prediction (Physics + Recovered Source)
    im2 = axes[2].imshow(u_pred, cmap="inferno", vmin=vmin_u, vmax=vmax_u, origin="lower")
    axes[2].set_title("PhICNet Prediction", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Recovered Source vs Real Source
    vmax_s = max(s_real.max(), s_est.max(), 1.0)
    im3 = axes[3].imshow(s_est, cmap="plasma", vmin=0.0, vmax=vmax_s, origin="lower")
    axes[3].set_title(r"Estimated Source $\hat{S}(t)$", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. PhICNet Absolute Error
    err = np.abs(u_real - u_pred)
    im4 = axes[4].imshow(err, cmap="magma", vmin=0.0, vmax=max(err.max(), 0.01), origin="lower")
    axes[4].set_title("Absolute Error $|U - \hat{U}|$", fontsize=11, fontweight="bold")
    axes[4].axis("off")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshot to {save_path}")


def create_comparison_animation(
    u_real_seq: np.ndarray,
    u_pred_seq: np.ndarray,
    u_phy_seq: np.ndarray,
    s_est_seq: np.ndarray,
    s_real_seq: np.ndarray,
    save_path: str = "phicnet_evolution.gif",
    fps: int = 10
):
    """
    Generate an animated GIF comparing True State, Physics Prediction, PhICNet, and Estimated Source over time.
    """
    num_frames = min(len(u_real_seq), len(u_pred_seq))
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=150)

    vmax_u = max(u_real_seq.max(), u_pred_seq.max())
    vmax_s = max(s_real_seq.max(), s_est_seq.max(), 1.0)

    im0 = axes[0, 0].imshow(u_real_seq[0], cmap="inferno", vmin=0, vmax=vmax_u, origin="lower")
    axes[0, 0].set_title("Ground Truth Temperature $U(t)$", fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(u_pred_seq[0], cmap="inferno", vmin=0, vmax=vmax_u, origin="lower")
    axes[0, 1].set_title("PhICNet Prediction", fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(u_phy_seq[0], cmap="inferno", vmin=0, vmax=vmax_u, origin="lower")
    axes[1, 0].set_title("Physics Only Prediction (No Source)", fontweight="bold")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(s_est_seq[0], cmap="plasma", vmin=0, vmax=vmax_s, origin="lower")
    axes[1, 1].set_title(r"Estimated Unknown Source $\hat{S}(t)$", fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle("PhICNet Dynamics & Unknown Source Recovery", fontsize=14, fontweight="bold")
    plt.tight_layout()

    def update(frame):
        im0.set_data(u_real_seq[frame])
        im1.set_data(u_pred_seq[frame])
        im2.set_data(u_phy_seq[frame])
        im3.set_data(s_est_seq[frame])
        fig.suptitle(f"PhICNet Dynamics & Source Recovery (Step {frame}/{num_frames})", fontsize=14, fontweight="bold")
        return [im0, im1, im2, im3]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved animation to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for PhICNet")
    parser.add_argument("--test_data", type=str, default="data/test.npz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phicnet_best.pt")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seq_idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.test_data)
    u_test = data["u"]
    s_test = data["s"]
    alpha = float(data.get("alpha", 0.01))
    dt = float(data.get("dt", 0.005))
    dx = float(data.get("dx", 1.0 / 63.0))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PhICNet(alpha=alpha, dt=dt, dx=dx, hidden_dim=ckpt.get("hidden_dim", 32)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    phy_model = PhysicsOnlyBaseline(alpha=alpha, dt=dt, dx=dx).to(device)

    # Pick sequence
    u_seq_np = u_test[args.seq_idx]
    s_seq_np = s_test[args.seq_idx]
    u_seq_tensor = torch.from_numpy(u_seq_np).unsqueeze(0).float().to(device)

    with torch.no_grad():
        u_pred, s_pred, u_phy = model(u_seq_tensor)
        u_pred = u_pred.squeeze().cpu().numpy()
        s_pred = s_pred.squeeze().cpu().numpy()
        u_phy = u_phy.squeeze().cpu().numpy()

    # Align sequence lengths (predictions start at t=1)
    u_real = u_seq_np[1:]
    s_real = s_seq_np[:-1]

    # Save snapshot at step 20
    snapshot_step = min(20, len(u_real) - 1)
    snapshot_path = os.path.join(args.out_dir, f"phicnet_snapshot_step{snapshot_step}.png")
    plot_comparison_snapshot(
        u_real=u_real[snapshot_step],
        u_pred=u_pred[snapshot_step],
        u_phy=u_phy[snapshot_step],
        s_est=s_pred[snapshot_step],
        s_real=s_real[snapshot_step],
        step_idx=snapshot_step,
        save_path=snapshot_path
    )

    # Create GIF animation of the first 40 frames
    gif_path = os.path.join(args.out_dir, "phicnet_diffusion_evolution.gif")
    clip_len = min(40, len(u_real))
    create_comparison_animation(
        u_real_seq=u_real[:clip_len],
        u_pred_seq=u_pred[:clip_len],
        u_phy_seq=u_phy[:clip_len],
        s_est_seq=s_pred[:clip_len],
        s_real_seq=s_real[:clip_len],
        save_path=gif_path,
        fps=10
    )


if __name__ == "__main__":
    main()
