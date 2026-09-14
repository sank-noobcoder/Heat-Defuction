"""
Recurrent Convolutional Network (ConvLSTM) for spatiotemporal source dynamics.

Maintains spatial memory across consecutive time frames (Section 14 & 15 in PhICNet notes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM Cell.
    Maintains spatial dimensions H x W while propagating temporal hidden and cell states.
    """
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # 4 gates: input, forget, candidate cell, output
        self.conv = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            state: Tuple of (h_prev, c_prev), each of shape (B, hidden_dim, H, W)
        Returns:
            (h_next, c_next)
        """
        b, _, h, w = x.shape
        if state is None:
            h_prev = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class RecurrentSourceNetwork(nn.Module):
    """
    Spatiotemporal source estimation network incorporating ConvLSTM temporal memory
    and RED-Net spatial reconstruction.
    """
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 32,
        out_channels: int = 1,
        non_negative_source: bool = True
    ):
        super().__init__()
        self.non_negative_source = non_negative_source
        self.hidden_dim = hidden_dim

        # Input feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Spatiotemporal recurrent cell
        self.recurrent_cell = ConvLSTMCell(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=3
        )

        # Output spatial decoder with residual connection
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        One time-step forward pass.
        Args:
            x: Input discrepancy / state tensor (B, in_channels, H, W)
            state: Previous recurrent state (h, c)
        Returns:
            source_pred: (B, out_channels, H, W)
            next_state: (h_next, c_next)
        """
        feat = self.encoder(x)
        h_next, c_next = self.recurrent_cell(feat, state)
        source = self.decoder(h_next)

        if self.non_negative_source:
            source = F.relu(source)

        return source, (h_next, c_next)


if __name__ == "__main__":
    rnn_net = RecurrentSourceNetwork(in_channels=2, hidden_dim=32, out_channels=1)
    inp = torch.randn(4, 2, 64, 64)
    out, state = rnn_net(inp)
    print(f"RecurrentSourceNetwork test: input {inp.shape} -> output {out.shape}, state h: {state[0].shape}")
    assert out.shape == (4, 1, 64, 64), "Shape error in RecurrentSourceNetwork!"
    print("RecurrentSourceNetwork test passed successfully.")
