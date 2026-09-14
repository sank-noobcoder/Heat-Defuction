# PhICNet: Physics-Incorporated Convolutional Recurrent Neural Network

A complete PyTorch implementation of **PhICNet** for physical dynamical systems with unknown external sources/perturbations, demonstrated on the **2D Heat Diffusion Equation**:

$$\frac{\partial U}{\partial t} = \alpha \left( \frac{\partial^2 U}{\partial x^2} + \frac{\partial^2 U}{\partial y^2} \right) + S(x, y, t)$$

Where:
- $U(x, y, t)$ is the 2D temperature field over time.
- $\alpha$ is the thermal diffusivity.
- $S(x, y, t)$ is an **unknown, unobserved heat source** (e.g. localized heater turning ON/OFF or moving across the domain).

---

## 🌟 Key Architecture & Concepts

PhICNet merges physical principles with deep learning using a dual-brain paradigm:

```text
                        Current Observation U(t)
                                   |
                +------------------+------------------+
                |                                     |
                v                                     v
       PHYSICS OPERATOR                       SOURCE ESTIMATION
     (Finite-Difference Conv2D)              (ConvLSTM + RED-Net)
  "What should happen normally?"         "What hidden source is present?"
                |                                     |
           U_phy(t+1)                              S_hat(t)
                |                                     |
                +------------------+------------------+
                                   |
                                   v
                       FINAL PREDICTION U(t+1)
                     = U_phy(t+1) + dt * S_hat(t)
```

1. **Differential Operator as Convolution**: The spatial Laplacian $\nabla^2 U$ is computed using a differentiable $3 \times 3$ finite-difference stencil via `nn.Conv2d`.
2. **Discrepancy Extraction**: The discrepancy $(U^t - U_{phy}^t)/\Delta t$ provides the neural network with physical residual cues.
3. **No Direct Source Supervision Required**: The model learns the hidden source indirectly through physical prediction consistency and an L1 sparsity penalty ($\mathcal{L}_{pred} + \beta \mathcal{L}_{source} + \lambda \mathcal{L}_{sparse}$).
4. **Autonomous Forecasting**: Autoregressive rollouts allow multi-step future state forecasting without future observations.

---

## 📁 Repository Structure

```text
internship/
├── phicnet/
│   ├── physics/
│   │   ├── __init__.py
│   │   └── heat_equation.py      # Numerical 2D finite-difference heat equation solver
│   ├── models/
│   │   ├── __init__.py
│   │   ├── physics_model.py      # Conv2d finite-difference Laplacian operator
│   │   ├── source_net.py         # RED-Net (Residual Encoder-Decoder Network)
│   │   ├── recurrent_source.py   # Spatiotemporal ConvLSTM source memory
│   │   └── phicnet.py            # Complete PhICNet and comparative baselines
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py             # Composite loss: L_pred + L_source + L_sparse
│   │   └── train.py              # End-to-end multi-model trainer with CUDA support
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py           # Benchmark suite comparing Baselines vs PhICNet
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py          # High-resolution 5-panel plots and animated GIF export
│   └── app.py                    # Interactive Streamlit dashboard
├── generate_data.py              # Synthetic data generation CLI
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Generate train, validation, and test trajectories ($64 \times 64$ grid, sequences of 100 frames with varying source behaviors):
```bash
python generate_data.py --train_seqs 60 --val_seqs 15 --test_seqs 15 --seq_len 100
```

### 3. Train Models
Train the full **PhICNet** model using GPU acceleration:
```bash
python -m phicnet.training.train --model phicnet --epochs 25 --batch_size 16
```

Optionally train the neural baselines for comparison:
```bash
python -m phicnet.training.train --model cnn --epochs 25
python -m phicnet.training.train --model conv_lstm --epochs 25
```

### 4. Benchmark & Evaluate
Run the quantitative benchmark comparing Physics-Only, Pure CNN, Pure ConvLSTM, and PhICNet across 1-step errors and multi-step rollouts:
```bash
python -m phicnet.evaluation.evaluate
```

### 5. Generate Visualizations & Animated GIFs
Export high-resolution comparison snapshots and an animated GIF:
```bash
python -m phicnet.visualization.visualize --out_dir outputs
```

### 6. Launch Interactive Streamlit Dashboard
```bash
streamlit run phicnet/app.py
```
