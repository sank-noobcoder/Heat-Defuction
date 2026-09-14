"""
2D Finite Difference Heat Equation Solver with Unknown External Sources.

Equation:
    dU/dt = alpha * (d^2 U / dx^2 + d^2 U / dy^2) + S(x, y, t)

Numerical discretization:
    Forward-Time Central-Space (FTCS) finite-difference scheme.
    Stability requirement: r = alpha * dt / (dx^2) <= 0.25 in 2D.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any


class HeatEquation2D:
    """
    2D Heat Equation numerical simulator using finite differences.
    """
    def __init__(
        self,
        grid_size: int = 64,
        domain_length: float = 1.0,
        alpha: float = 0.01,
        dt: float = 0.005,
        boundary_condition: str = "dirichlet"
    ):
        """
        Args:
            grid_size: Number of grid points along each spatial dimension (Nx = Ny).
            domain_length: Physical length of square domain [0, L] x [0, L].
            alpha: Thermal diffusivity constant.
            dt: Time step size.
            boundary_condition: 'dirichlet' (zero temperature at boundaries) or 'neumann' (insulated).
        """
        self.nx = grid_size
        self.ny = grid_size
        self.l = domain_length
        self.alpha = alpha
        self.dt = dt
        self.boundary_condition = boundary_condition.lower()

        self.dx = domain_length / (grid_size - 1)
        self.dy = self.dx

        # Stability parameter check
        self.r = self.alpha * self.dt / (self.dx ** 2)
        if self.r > 0.25:
            raise ValueError(
                f"FTCS stability criterion violated: r = {self.r:.4f} > 0.25. "
                f"Reduce dt (current: {dt}) or increase dx."
            )

        # Coordinate grid
        x = np.linspace(0, domain_length, self.nx)
        y = np.linspace(0, domain_length, self.ny)
        self.xx, self.yy = np.meshgrid(x, y)

    def compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete 2D Laplacian using 5-point stencil:
        laplacian = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx^2
        """
        laplacian = np.zeros_like(u)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
            - 4.0 * u[1:-1, 1:-1]
        ) / (self.dx ** 2)

        # Boundary conditions
        if self.boundary_condition == "neumann":
            # Insulated boundaries: zero normal derivative (du/dn = 0)
            # Top boundary (row 0)
            laplacian[0, 1:-1] = (
                2.0 * u[1, 1:-1] + u[0, 2:] + u[0, :-2] - 4.0 * u[0, 1:-1]
            ) / (self.dx ** 2)
            # Bottom boundary (row -1)
            laplacian[-1, 1:-1] = (
                2.0 * u[-2, 1:-1] + u[-1, 2:] + u[-1, :-2] - 4.0 * u[-1, 1:-1]
            ) / (self.dx ** 2)
            # Left boundary (col 0)
            laplacian[1:-1, 0] = (
                u[2:, 0] + u[:-2, 0] + 2.0 * u[1:-1, 1] - 4.0 * u[1:-1, 0]
            ) / (self.dx ** 2)
            # Right boundary (col -1)
            laplacian[1:-1, -1] = (
                u[2:, -1] + u[:-2, -1] + 2.0 * u[1:-1, -2] - 4.0 * u[1:-1, -1]
            ) / (self.dx ** 2)
        else:
            # Dirichlet: U = 0 at boundaries, Laplacian stays 0 at boundary
            pass

        return laplacian

    def step(self, u: np.ndarray, source: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance state by one time step: U^{t+1} = U^t + dt * (alpha * laplacian(U^t) + source^t)
        """
        laplacian = self.compute_laplacian(u)
        u_next = u + self.dt * (self.alpha * laplacian)
        if source is not None:
            u_next = u_next + self.dt * source

        if self.boundary_condition == "dirichlet":
            u_next[0, :] = 0.0
            u_next[-1, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0

        return u_next

    def simulate(
        self,
        u_init: np.ndarray,
        num_steps: int,
        source_func: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate heat equation over multiple time steps.

        Args:
            u_init: Initial temperature field of shape (nx, ny).
            num_steps: Number of simulation time steps.
            source_func: Function (t_idx, xx, yy) -> source_map of shape (nx, ny).

        Returns:
            u_trajectory: Array of shape (num_steps, nx, ny).
            s_trajectory: Array of shape (num_steps, nx, ny).
        """
        u_traj = np.zeros((num_steps, self.nx, self.ny), dtype=np.float32)
        s_traj = np.zeros((num_steps, self.nx, self.ny), dtype=np.float32)

        u_curr = u_init.copy().astype(np.float32)
        if self.boundary_condition == "dirichlet":
            u_curr[0, :] = 0.0
            u_curr[-1, :] = 0.0
            u_curr[:, 0] = 0.0
            u_curr[:, -1] = 0.0

        for t in range(num_steps):
            u_traj[t] = u_curr
            if source_func is not None:
                s_curr = source_func(t, self.xx, self.yy).astype(np.float32)
            else:
                s_curr = np.zeros_like(u_curr, dtype=np.float32)
            s_traj[t] = s_curr
            u_curr = self.step(u_curr, s_curr)

        return u_traj, s_traj


def make_gaussian_source(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float = 0.05,
    amplitude: float = 10.0
) -> np.ndarray:
    """
    Generate a localized 2D Gaussian heat source.
    """
    r_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return amplitude * np.exp(-r_sq / (2.0 * sigma ** 2))


def generate_heat_trajectory(
    grid_size: int = 64,
    num_steps: int = 100,
    alpha: float = 0.01,
    dt: float = 0.005,
    source_type: str = "pulsing",
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a single physically consistent heat diffusion sequence with unknown source.

    Source types:
        - 'static': Localized Gaussian heater remains at fixed position.
        - 'pulsing': Localized heater periodically switches ON and OFF.
        - 'moving': Localized heater moves across the plate.
        - 'none': Free diffusion with no source.

    Returns:
        u_traj: (num_steps, grid_size, grid_size)
        s_traj: (num_steps, grid_size, grid_size)
        metadata: dict of simulation parameters
    """
    if rng is None:
        rng = np.random.default_rng()

    solver = HeatEquation2D(
        grid_size=grid_size,
        domain_length=1.0,
        alpha=alpha,
        dt=dt,
        boundary_condition="dirichlet"
    )

    # Initial condition: random smooth temperature field or cold plate
    # Generate 1 to 3 initial smooth warm spots
    u_init = np.zeros((grid_size, grid_size), dtype=np.float32)
    num_initial_spots = rng.integers(1, 4)
    for _ in range(num_initial_spots):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        sigma = rng.uniform(0.06, 0.12)
        amp = rng.uniform(1.0, 4.0)
        u_init += make_gaussian_source(solver.xx, solver.yy, cx, cy, sigma, amp).astype(np.float32)

    # Source parameters
    source_params = {}
    if source_type == "static":
        cx = rng.uniform(0.25, 0.75)
        cy = rng.uniform(0.25, 0.75)
        sigma = rng.uniform(0.04, 0.07)
        amp = rng.uniform(8.0, 15.0)
        source_params = {"cx": cx, "cy": cy, "sigma": sigma, "amp": amp}

        def source_fn(t_idx, xx, yy):
            return make_gaussian_source(xx, yy, cx, cy, sigma, amp)

    elif source_type == "pulsing":
        cx = rng.uniform(0.25, 0.75)
        cy = rng.uniform(0.25, 0.75)
        sigma = rng.uniform(0.04, 0.07)
        amp = rng.uniform(10.0, 18.0)
        period = rng.integers(20, 40)
        source_params = {"cx": cx, "cy": cy, "sigma": sigma, "amp": amp, "period": period}

        def source_fn(t_idx, xx, yy):
            # ON for first half of period, OFF for second half
            is_on = ((t_idx % period) < (period // 2))
            if is_on:
                return make_gaussian_source(xx, yy, cx, cy, sigma, amp)
            return np.zeros_like(xx)

    elif source_type == "moving":
        start_x, end_x = rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)
        start_y, end_y = rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)
        sigma = rng.uniform(0.04, 0.06)
        amp = rng.uniform(10.0, 16.0)
        source_params = {
            "start": (start_x, start_y), "end": (end_x, end_y),
            "sigma": sigma, "amp": amp
        }

        def source_fn(t_idx, xx, yy):
            progress = t_idx / max(1, num_steps - 1)
            cx = start_x + progress * (end_x - start_x)
            cy = start_y + progress * (end_y - start_y)
            return make_gaussian_source(xx, yy, cx, cy, sigma, amp)

    elif source_type == "none":
        source_fn = None
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    u_traj, s_traj = solver.simulate(u_init, num_steps, source_fn)
    metadata = {
        "alpha": alpha,
        "dt": dt,
        "grid_size": grid_size,
        "num_steps": num_steps,
        "source_type": source_type,
        "source_params": source_params
    }
    return u_traj, s_traj, metadata


if __name__ == "__main__":
    print("Testing 2D Heat Equation Solver...")
    u, s, meta = generate_heat_trajectory(
        grid_size=64, num_steps=50, alpha=0.01, dt=0.005, source_type="pulsing"
    )
    print(f"Generated trajectory shape: U = {u.shape}, S = {s.shape}")
    print(f"U min: {u.min():.4f}, max: {u.max():.4f}, mean: {u.mean():.4f}")
    print(f"S min: {s.min():.4f}, max: {s.max():.4f}, mean: {s.mean():.4f}")
    assert not np.isnan(u).any(), "NaN found in U trajectory!"
    assert not np.isnan(s).any(), "NaN found in S trajectory!"
    print("Test passed successfully!")
