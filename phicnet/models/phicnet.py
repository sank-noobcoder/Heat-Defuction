"""
PhICNet Architecture and Experimental Baselines.

Combines explicit finite-difference physical operator with a recurrent neural
source model to perform physical state forecasting and unknown source recovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

from .physics_model import PhysicsHeat2D
from .recurrent_source import RecurrentSourceNetwork
from .source_net import REDNet


class PhICNet(nn.Module):
    """
    PhICNet: Physics-Incorporated Convolutional Recurrent Neural Network.
    
    Architecture:
    - Physics Branch: Discrete 2D heat equation operator (Conv2D with finite-difference Laplacian).
    - Source Branch: Spatiotemporal ConvLSTM with RED-Net decoder that estimates unknown source S^t.
    - Integration: U^{t+1} = U_{phy}^{t+1} + dt * S^t
    """
    def __init__(
        self,
        alpha: float = 0.01,
        dt: float = 0.005,
        dx: float = 1.0 / 63.0,
        hidden_dim: int = 32,
        non_negative_source: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.dt = dt
        self.dx = dx

        # Physics branch
        self.physics = PhysicsHeat2D(alpha=alpha, dt=dt, dx=dx, learnable_alpha=False)

        # Source estimation branch (takes current state U^t and physics discrepancy)
        # Input channels: 2 (U^t and discrepancy)
        self.source_net = RecurrentSourceNetwork(
            in_channels=2,
            hidden_dim=hidden_dim,
            out_channels=1,
            non_negative_source=non_negative_source
        )

    def compute_discrepancy(self, u_curr: torch.Tensor, u_prev: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute discrepancy between actual observation and physics-only prediction.
        Discrepancy ~ (U^t - U_{phy}^t) / dt
        """
        if u_prev is None:
            # At t=0, discrepancy starts at zero
            return torch.zeros_like(u_curr)
        
        u_phy_t = self.physics(u_prev)
        discrepancy = (u_curr - u_phy_t) / self.dt
        # Clamp discrepancy to avoid runaway feedback amplification during multi-step rollouts
        discrepancy = torch.clamp(discrepancy, -25.0, 25.0)
        return discrepancy

    def step(
        self,
        u_curr: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Advance one step from u_curr:
        1. Physics: u_phy = physics(u_curr)
        2. Discrepancy: d = (u_curr - physics(u_prev)) / dt
        3. Source: s_pred, next_state = source_net([u_curr, d], state)
        4. Final: u_next = u_phy + dt * s_pred
        """
        # 1. Physics prediction
        u_phy_next = self.physics(u_curr)

        # 2. Compute discrepancy
        disc = self.compute_discrepancy(u_curr, u_prev)

        # 3. Source estimation
        source_in = torch.cat([u_curr, disc], dim=1)
        s_pred, next_state = self.source_net(source_in, state)

        # 4. Integrated prediction
        u_next = u_phy_next + self.dt * s_pred

        # Enforce boundary condition
        u_next[:, :, 0, :] = 0.0
        u_next[:, :, -1, :] = 0.0
        u_next[:, :, :, 0] = 0.0
        u_next[:, :, :, -1] = 0.0

        return u_next, s_pred, next_state

    def forward(
        self,
        u_seq: torch.Tensor,
        unroll_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass over sequence of observed states for training.
        Supports multi-step unrolled training to eliminate exposure bias.
        
        Args:
            u_seq: Tensor of shape (B, T, 1, H, W) or (B, T, H, W)
            unroll_steps: Number of consecutive autoregressive steps before re-anchoring to ground truth
        Returns:
            u_pred_seq: Predicted states for t=1..T of shape (B, T-1, 1, H, W)
            s_pred_seq: Predicted sources for t=0..T-2 of shape (B, T-1, 1, H, W)
            u_phy_seq:  Physics-only predictions for t=1..T of shape (B, T-1, 1, H, W)
        """
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)  # (B, T, 1, H, W)

        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []
        u_phys = []

        state = None
        u_prev = None
        curr_u = u_seq[:, 0]

        for t in range(t_steps - 1):
            if unroll_steps <= 1 or (t % unroll_steps == 0):
                u_curr = u_seq[:, t]
                u_prev = u_seq[:, t - 1] if t > 0 else None
            else:
                u_curr = curr_u
                # u_prev is already the previous step state

            u_phy_next = self.physics(u_curr)
            u_phys.append(u_phy_next)

            disc = self.compute_discrepancy(u_curr, u_prev)
            source_in = torch.cat([u_curr, disc], dim=1)
            s_pred, state = self.source_net(source_in, state)
            s_preds.append(s_pred)

            u_next = u_phy_next + self.dt * s_pred
            u_next[:, :, 0, :] = 0.0
            u_next[:, :, -1, :] = 0.0
            u_next[:, :, :, 0] = 0.0
            u_next[:, :, :, -1] = 0.0
            u_preds.append(u_next)

            u_prev = u_curr
            curr_u = u_next

        u_pred_seq = torch.stack(u_preds, dim=1)
        s_pred_seq = torch.stack(s_preds, dim=1)
        u_phy_seq = torch.stack(u_phys, dim=1)

        return u_pred_seq, s_pred_seq, u_phy_seq

    @torch.no_grad()
    def rollout_autoregressive(
        self,
        u_warmup: torch.Tensor,
        rollout_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform autonomous multi-step forecasting without ground truth observations.
        
        Args:
            u_warmup: Warmup observation sequence (B, T_warmup, 1, H, W) to prime recurrent state.
            rollout_steps: Number of future time steps to forecast.
        Returns:
            u_forecast: (B, rollout_steps, 1, H, W)
            s_forecast: (B, rollout_steps, 1, H, W)
        """
        if u_warmup.dim() == 4:
            u_warmup = u_warmup.unsqueeze(2)

        b, t_warm, c, h, w = u_warmup.shape
        state = None
        u_prev = None

        # Prime the recurrent memory with warmup observations
        for t in range(t_warm):
            u_curr = u_warmup[:, t]
            disc = self.compute_discrepancy(u_curr, u_prev)
            source_in = torch.cat([u_curr, disc], dim=1)
            s_pred, state = self.source_net(source_in, state)
            u_prev = u_curr

        # Autoregressive rollout: feed model's own predictions as next input
        u_preds = []
        s_preds = []
        # Crucial fix: u_prev must be the frame immediately preceding curr_u (i.e. t_warm - 2)
        # to ensure true 1-step physical lag throughout rollout!
        u_prev = u_warmup[:, -2] if t_warm > 1 else None
        curr_u = u_warmup[:, -1]

        for step in range(rollout_steps):
            u_phy_next = self.physics(curr_u)
            disc = self.compute_discrepancy(curr_u, u_prev)
            source_in = torch.cat([curr_u, disc], dim=1)
            s_pred, state = self.source_net(source_in, state)
            s_preds.append(s_pred)

            next_u = u_phy_next + self.dt * s_pred
            next_u[:, :, 0, :] = 0.0
            next_u[:, :, -1, :] = 0.0
            next_u[:, :, :, 0] = 0.0
            next_u[:, :, :, -1] = 0.0
            u_preds.append(next_u)

            # Correctly advance temporal states: u_prev becomes current step, curr_u becomes next step!
            u_prev = curr_u
            curr_u = next_u

        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)


class PhysicsOnlyBaseline(nn.Module):
    """
    Baseline 1: Physics Only (assumes zero external source S=0).
    """
    def __init__(self, alpha: float = 0.01, dt: float = 0.005, dx: float = 1.0 / 63.0):
        super().__init__()
        self.physics = PhysicsHeat2D(alpha=alpha, dt=dt, dx=dx)

    def forward(self, u_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)
        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []
        for t in range(t_steps - 1):
            u_curr = u_seq[:, t]
            u_next = self.physics(u_curr)
            u_preds.append(u_next)
            s_preds.append(torch.zeros_like(u_curr))
        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)

    @torch.no_grad()
    def rollout_autoregressive(self, u_init: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        if u_init.dim() == 5:
            curr_u = u_init[:, -1]
        elif u_init.dim() == 4:
            curr_u = u_init[:, -1].unsqueeze(1)
        elif u_init.dim() == 3:
            curr_u = u_init.unsqueeze(1)
        else:
            curr_u = u_init

        preds = []
        for _ in range(rollout_steps):
            curr_u = self.physics(curr_u)
            preds.append(curr_u)
        return torch.stack(preds, dim=1)


class CNNBaseline(nn.Module):
    """
    Baseline 2: Pure CNN (no physics differential operator).
    Directly predicts U^{t+1} from U^t using a deep convolutional network.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, u_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)
        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []
        for t in range(t_steps - 1):
            u_curr = u_seq[:, t]
            u_next = self.net(u_curr)
            u_next[:, :, 0, :] = 0.0
            u_next[:, :, -1, :] = 0.0
            u_next[:, :, :, 0] = 0.0
            u_next[:, :, :, -1] = 0.0
            u_preds.append(u_next)
            s_preds.append(torch.zeros_like(u_curr))
        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)

    @torch.no_grad()
    def rollout_autoregressive(self, u_init: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        if u_init.dim() == 5:
            curr_u = u_init[:, -1]
        elif u_init.dim() == 4:
            curr_u = u_init[:, -1].unsqueeze(1)
        elif u_init.dim() == 3:
            curr_u = u_init.unsqueeze(1)
        else:
            curr_u = u_init

        preds = []
        for _ in range(rollout_steps):
            curr_u = self.net(curr_u)
            curr_u[:, :, 0, :] = 0.0
            curr_u[:, :, -1, :] = 0.0
            curr_u[:, :, :, 0] = 0.0
            curr_u[:, :, :, -1] = 0.0
            preds.append(curr_u)
        return torch.stack(preds, dim=1)


class ConvLSTMBaseline(nn.Module):
    """
    Baseline 3: Pure ConvLSTM (data-driven recurrent spatiotemporal network without physics).
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv_lstm = nn.ModuleList([
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=3, padding=1)
        ])
        self.decoder = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, u_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_seq.dim() == 4:
            u_seq = u_seq.unsqueeze(2)
        b, t_steps, c, h, w = u_seq.shape
        u_preds = []
        s_preds = []

        h_state = torch.zeros(b, self.hidden_dim, h, w, device=u_seq.device)
        c_state = torch.zeros(b, self.hidden_dim, h, w, device=u_seq.device)

        for t in range(t_steps - 1):
            u_curr = u_seq[:, t]
            feat = F.relu(self.encoder(u_curr))
            
            combined = torch.cat([feat, h_state], dim=1)
            gates = self.conv_lstm[0](combined)
            i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
            c_state = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
            h_state = torch.sigmoid(o) * torch.tanh(c_state)

            u_next = self.decoder(h_state)
            u_next[:, :, 0, :] = 0.0
            u_next[:, :, -1, :] = 0.0
            u_next[:, :, :, 0] = 0.0
            u_next[:, :, :, -1] = 0.0
            u_preds.append(u_next)
            s_preds.append(torch.zeros_like(u_curr))

        return torch.stack(u_preds, dim=1), torch.stack(s_preds, dim=1)

    @torch.no_grad()
    def rollout_autoregressive(self, u_warmup: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        if u_warmup.dim() == 4:
            u_warmup = u_warmup.unsqueeze(2)
        b, t_warm, c, h, w = u_warmup.shape

        h_state = torch.zeros(b, self.hidden_dim, h, w, device=u_warmup.device)
        c_state = torch.zeros(b, self.hidden_dim, h, w, device=u_warmup.device)

        for t in range(t_warm):
            u_curr = u_warmup[:, t]
            feat = F.relu(self.encoder(u_curr))
            combined = torch.cat([feat, h_state], dim=1)
            gates = self.conv_lstm[0](combined)
            i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
            c_state = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
            h_state = torch.sigmoid(o) * torch.tanh(c_state)

        preds = []
        curr_u = u_warmup[:, -1]
        for _ in range(rollout_steps):
            feat = F.relu(self.encoder(curr_u))
            combined = torch.cat([feat, h_state], dim=1)
            gates = self.conv_lstm[0](combined)
            i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
            c_state = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
            h_state = torch.sigmoid(o) * torch.tanh(c_state)
            curr_u = self.decoder(h_state)
            curr_u[:, :, 0, :] = 0.0
            curr_u[:, :, -1, :] = 0.0
            curr_u[:, :, :, 0] = 0.0
            curr_u[:, :, :, -1] = 0.0
            preds.append(curr_u)

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    print("Testing PhICNet models...")
    dummy_seq = torch.randn(2, 10, 1, 64, 64)
    model = PhICNet(alpha=0.01, dt=0.005, dx=1.0/63.0)
    u_pred, s_pred, u_phy = model(dummy_seq)
    print(f"PhICNet sequence forward: u_pred {u_pred.shape}, s_pred {s_pred.shape}, u_phy {u_phy.shape}")
    assert u_pred.shape == (2, 9, 1, 64, 64)

    rollout_u, rollout_s = model.rollout_autoregressive(dummy_seq[:, :3], rollout_steps=5)
    print(f"PhICNet rollout: u {rollout_u.shape}, s {rollout_s.shape}")
    assert rollout_u.shape == (2, 5, 1, 64, 64)
    print("All PhICNet model checks passed!")
