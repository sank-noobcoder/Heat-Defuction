"""
Residual Encoder-Decoder Network (RED-Net) for spatial source mapping.

Implements symmetrical convolutional encoder-decoder with residual skip connections
to capture localized spatial source patterns (Section 16 in PhICNet notes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class REDNetBlock(nn.Module):
    """
    Conv + BatchNorm (optional) + ReLU block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class REDNet(nn.Module):
    """
    Residual Encoder-Decoder Network for 2D spatial fields.
    Extracts multiscale spatial representations and reconstructs source maps with skip connections.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 3,
        non_negative_source: bool = True
    ):
        super().__init__()
        self.non_negative_source = non_negative_source

        # Encoder layers
        self.enc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)

        # Decoder layers (with residual skip connections)
        self.dec3 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual skip connections.
        Args:
            x: Tensor of shape (B, in_channels, H, W)
        Returns:
            source_map: Tensor of shape (B, out_channels, H, W)
        """
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))

        # Decoder with symmetrical residual addition
        d3 = self.relu(self.dec3(e3) + e2)
        d2 = self.relu(self.dec2(d3) + e1)
        out = self.dec1(d2)

        if self.non_negative_source:
            out = F.relu(out)

        return out


if __name__ == "__main__":
    net = REDNet(in_channels=2, out_channels=1, hidden_dim=32)
    inp = torch.randn(4, 2, 64, 64)
    out = net(inp)
    print(f"REDNet test: input {inp.shape} -> output {out.shape}")
    assert out.shape == (4, 1, 64, 64), "Output shape mismatch!"
    print("REDNet test passed successfully.")
