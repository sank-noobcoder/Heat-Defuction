"""
Interactive Streamlit Web Dashboard for PhICNet.

Allows users to explore PhICNet in real-time:
- Visualize 2D heat diffusion under unknown perturbations.
- Compare Physics-Only vs PhICNet predictions.
- Inspect the recovered unknown source map S_hat(x, y, t).
- Test multi-step autonomous future forecasting.
"""

import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from phicnet.models.phicnet import PhICNet, PhysicsOnlyBaseline
from phicnet.physics.heat_equation import generate_heat_trajectory


st.set_page_config(
    page_title="PhICNet: Physics-Incorporated Neural Network",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 PhICNet: Physics-Incorporated ConvRNN for 2D Heat Diffusion")
st.markdown("""
**PhICNet** merges a known finite-difference physical operator with a learned convolutional recurrent source model.
It forecasts spatiotemporal physical fields while uncovering **unknown external sources** *without direct source supervision*.
""")


@st.cache_resource
def load_phicnet_model(checkpoint_path: str = "checkpoints/phicnet_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        return None, device

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    alpha = ckpt.get("alpha", 0.01)
    dt = ckpt.get("dt", 0.005)
    dx = ckpt.get("dx", 1.0 / 63.0)
    hidden_dim = ckpt.get("hidden_dim", 32)

    model = PhICNet(alpha=alpha, dt=dt, dx=dx, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


# Sidebar Controls
st.sidebar.header("⚙️ Simulation & Model Settings")

data_source = st.sidebar.radio(
    "Data Source",
    ["Pre-generated Test Dataset", "Interactive On-The-Fly Simulation"]
)

model, device = load_phicnet_model()
if model is None:
    st.sidebar.warning("⚠️ No trained PhICNet checkpoint found at `checkpoints/phicnet_best.pt`. Train the model first to see live neural inference!")

if data_source == "Pre-generated Test Dataset":
    test_file = "data/test.npz"
    if os.path.exists(test_file):
        test_data = np.load(test_file)
        u_all = test_data["u"]
        s_all = test_data["s"]
        seq_idx = st.sidebar.slider("Select Test Sequence", 0, len(u_all) - 1, 0)
        u_seq = u_all[seq_idx]
        s_seq = s_all[seq_idx]
        alpha = float(test_data.get("alpha", 0.01))
        dt = float(test_data.get("dt", 0.005))
        dx = float(test_data.get("dx", 1.0 / 63.0))
    else:
        st.warning("`data/test.npz` not found. Switching to on-the-fly simulation.")
        data_source = "Interactive On-The-Fly Simulation"

if data_source == "Interactive On-The-Fly Simulation":
    st.sidebar.subheader("Physics Parameters")
    source_type = st.sidebar.selectbox("Unknown Source Type", ["pulsing", "static", "moving"])
    seq_len = st.sidebar.slider("Sequence Length (Frames)", 30, 100, 50)
    alpha = st.sidebar.slider("Diffusivity (alpha)", 0.005, 0.025, 0.01, step=0.001)
    dt = 0.005
    dx = 1.0 / 63.0

    if st.sidebar.button("🎲 Generate New Simulation") or "custom_u" not in st.session_state:
        u_traj, s_traj, _ = generate_heat_trajectory(
            grid_size=64, num_steps=seq_len, alpha=alpha, dt=dt, source_type=source_type
        )
        st.session_state["custom_u"] = u_traj
        st.session_state["custom_s"] = s_traj

    u_seq = st.session_state["custom_u"]
    s_seq = st.session_state["custom_s"]


# Model Inference
total_frames = len(u_seq)
phy_baseline = PhysicsOnlyBaseline(alpha=alpha, dt=dt, dx=dx).to(device)

u_tensor = torch.from_numpy(u_seq).unsqueeze(0).float().to(device)

with torch.no_grad():
    if model is not None:
        u_pred_t, s_pred_t, u_phy_t = model(u_tensor)
        u_pred = u_pred_t.squeeze().cpu().numpy()
        s_pred = s_pred_t.squeeze().cpu().numpy()
        u_phy = u_phy_t.squeeze().cpu().numpy()
    else:
        # Fallback if no weights trained yet
        u_phy_t, _ = phy_baseline(u_tensor)
        u_phy = u_phy_t.squeeze().cpu().numpy()
        u_pred = u_phy
        s_pred = np.zeros_like(s_seq[:-1])

# UI Step Slider
t_idx = st.slider("⏱️ Time Step Index (t)", 1, total_frames - 1, min(15, total_frames - 1))
idx_pred = t_idx - 1

# Metrics banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
u_true_frame = u_seq[t_idx]
u_pred_frame = u_pred[idx_pred]
u_phy_frame = u_phy[idx_pred]
s_est_frame = s_pred[idx_pred]
s_true_frame = s_seq[idx_pred]

err_phic = np.linalg.norm(u_true_frame - u_pred_frame) / (np.linalg.norm(u_true_frame) + 1e-7)
err_phy = np.linalg.norm(u_true_frame - u_phy_frame) / (np.linalg.norm(u_true_frame) + 1e-7)

col_m1.metric("PhICNet Rel L2 Error", f"{err_phic * 100:.2f}%")
col_m2.metric("Physics-Only Rel L2 Error", f"{err_phy * 100:.2f}%")
col_m3.metric("Max Temperature", f"{u_true_frame.max():.2f}")
col_m4.metric("Estimated Source Peak", f"{s_est_frame.max():.2f}")

# Main Visual Plots
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), dpi=150)
vmax_u = max(u_true_frame.max(), u_pred_frame.max(), u_phy_frame.max(), 0.1)

# 1. Real Temperature
im0 = axes[0].imshow(u_true_frame, cmap="inferno", vmin=0, vmax=vmax_u, origin="lower")
axes[0].set_title(f"True Temperature $U(t)$", fontweight="bold")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# 2. Physics-Only (misses the heater)
im1 = axes[1].imshow(u_phy_frame, cmap="inferno", vmin=0, vmax=vmax_u, origin="lower")
axes[1].set_title("Physics Only Prediction", fontweight="bold")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# 3. PhICNet Prediction
im2 = axes[2].imshow(u_pred_frame, cmap="inferno", vmin=0, vmax=vmax_u, origin="lower")
axes[2].set_title("PhICNet Prediction", fontweight="bold")
axes[2].axis("off")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# 4. Recovered Source
vmax_s = max(s_true_frame.max(), s_est_frame.max(), 1.0)
im3 = axes[3].imshow(s_est_frame, cmap="plasma", vmin=0, vmax=vmax_s, origin="lower")
axes[3].set_title(r"Estimated Source $\hat{S}(t)$", fontweight="bold")
axes[3].axis("off")
fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

# 5. PhICNet Error
err_map = np.abs(u_true_frame - u_pred_frame)
im4 = axes[4].imshow(err_map, cmap="magma", vmin=0, vmax=max(err_map.max(), 0.01), origin="lower")
axes[4].set_title("PhICNet Error $|U - \hat{U}|$", fontweight="bold")
axes[4].axis("off")
fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""
---
### 🧠 How PhICNet Works:
1. **Physics Operator (Convolutional)**: Uses finite difference stencils as fixed convolution kernels to compute $\Delta t \cdot \alpha \nabla^2 U^t$.
2. **Discrepancy Extraction**: Computes the difference between observed states and pure physics to isolate unknown physical influences.
3. **Temporal Source Memory (ConvLSTM & RED-Net)**: Recovers the unknown heater $S(x, y, t)$ over time without ground truth source labels.
4. **Autonomous Forecasting**: Incorporates both physics and learned source dynamics for multi-step forward rollouts.
""")
