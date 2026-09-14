"""
Physics-Incorporated Operator Module for 2D Heat Diffusion.

Implements the differential Laplacian operator using 2D convolutions with
finite-difference stencil kernels as described in PhICNet Section 8 & 9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsHeat2D(nn.Module):
    """
    Differentiable 2D Heat Equation operator using fixed 3x3 finite-difference convolution kernels.
    
    Diffusion update:
        U_{phy}^{t+1} = U^t + dt * alpha * (d^2 U / dx^2 + d^2 U / dy^2)
    """
    def __init__(
        self,
        alpha: float = 0.01,
        dt: float = 0.005,
        dx: float = 1.0 / 63.0,
        learnable_alpha: bool = False
    ):
        super().__init__()
        self.dt = dt
        self.dx = dx

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        # Directional finite-difference second-derivative kernels (Section 9 of notes)
        # K_x: d^2/dx^2 = [[0, 0, 0], [1, -2, 1], [0, 0, 0]] / dx^2
        # K_y: d^2/dy^2 = [[0, 1, 0], [0, -2, 0], [0, 1, 0]] / dy^2
        # Combined 5-point discrete Laplacian kernel:
        laplacian_weight = torch.tensor([
            [[[0.0,  1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0,  1.0, 0.0]]]
        ], dtype=torch.float32) / (dx ** 2)

        self.register_buffer("laplacian_kernel", laplacian_weight)

    def compute_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete Laplacian for batch u of shape (B, 1, H, W).
        Uses Dirichlet zero-boundary padding.
        """
        is_3d = False
        if u.dim() == 2:
            u = u.unsqueeze(0).unsqueeze(0)
        elif u.dim() == 3:
            u = u.unsqueeze(1)
            is_3d = True

        u_padded = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)
        laplacian = F.conv2d(u_padded, self.laplacian_kernel, padding=0)
        
        # Enforce Dirichlet zero boundary on Laplacian output
        laplacian[:, :, 0, :] = 0.0
        laplacian[:, :, -1, :] = 0.0
        laplacian[:, :, :, 0] = 0.0
        laplacian[:, :, :, -1] = 0.0

        if is_3d:
            laplacian = laplacian.squeeze(1)

        return laplacian

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-only next-state prediction U_{phy}^{t+1}:
            U_{phy}^{t+1} = U^t + dt * alpha * Laplacian(U^t)
        """
        orig_dim = u.dim()
        if orig_dim == 3:
            u = u.unsqueeze(1)

        laplacian = self.compute_laplacian(u)
        u_next = u + self.dt * (self.alpha * laplacian)
        
        # Enforce boundary condition
        u_next[:, :, 0, :] = 0.0
        u_next[:, :, -1, :] = 0.0
        u_next[:, :, :, 0] = 0.0
        u_next[:, :, :, -1] = 0.0

        if orig_dim == 3:
            u_next = u_next.squeeze(1)

        return u_next


if __name__ == "__main__":
    phy = PhysicsHeat2D(alpha=0.01, dt=0.005, dx=1.0/63.0)
    x = torch.randn(2, 1, 64, 64)
    out = phy(x)
    print(f"Physics operator test: input shape {x.shape} -> output shape {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("Physics operator test passed successfully.")
