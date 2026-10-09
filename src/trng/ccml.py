"""Coupled chaotic map lattice (CCML) used as a mixing stage."""

from __future__ import annotations

import numpy as np

ALPHA = 1.99999
"""Tent-map control parameter; chaotic (positive Lyapunov exponent) as alpha -> 2."""


def tent(x: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Apply the tent map ``f(x) = alpha * min(x, 1 - x)`` element-wise.

    Args:
        x: Values in ``[0, 1]``.
        alpha: Control parameter in ``(0, 2]``.

    Returns:
        The mapped values, same shape as ``x``.
    """
    return alpha * np.minimum(x, 1.0 - x)


class CCML:
    """Ring of ``size`` tent maps with nearest-neighbour coupling.

    ``x_{t+1}[i] = (1 - eps) f(x_t[i]) + eps/2 (f(x_t[i+1]) + f(x_t[i-1]))``

    Note that the coupling acts on ``f(x)``, as in the standard CCML definition.

    Args:
        size: Number of lattice sites ``L``.
        eps: Coupling constant.
        alpha: Tent-map control parameter.
        seed: Initial state in ``(0, 1)``; a fixed irrational-looking default is used
            when omitted. Real entropy always comes from :meth:`inject`.
    """

    def __init__(
        self,
        size: int = 8,
        eps: float = 0.05,
        alpha: float = ALPHA,
        seed: np.ndarray | None = None,
    ) -> None:
        if size < 3:
            raise ValueError("size must be >= 3")
        self.size = size
        self.eps = eps
        self.alpha = alpha
        if seed is None:
            seed = (np.arange(1, size + 1) * 0.6180339887498949) % 1.0
        self.x = np.asarray(seed, dtype=np.float64).copy()
        if self.x.shape != (size,):
            raise ValueError("seed must have shape (size,)")

    def step(self) -> np.ndarray:
        """Advance the lattice one time step and return the new state."""
        fx = tent(self.x, self.alpha)
        self.x = (1.0 - self.eps) * fx + 0.5 * self.eps * (np.roll(fx, -1) + np.roll(fx, 1))
        return self.x

    def inject(self, symbols: np.ndarray) -> None:
        """Perturb the lattice with up to ``size`` noise symbols in ``[0, 255]``.

        Args:
            symbols: Integer noise symbols; element ``j`` perturbs site ``j``.
        """
        s = np.zeros(self.size)
        s[: len(symbols)] = (np.asarray(symbols, dtype=np.float64) + 0.5) / 256.0
        self.x = (self.x + s) % 1.0
        # A float tent map can fall onto 0 / a fixed point; keep strictly inside (0, 1).
        self.x = np.clip(self.x, 1e-9, 1.0 - 1e-9)

    def mantissa_bytes(self) -> bytes:
        """Return the low 32 mantissa bits of every site (the fastest-decorrelating bits)."""
        low = self.x.view(np.uint64) & np.uint64(0xFFFFFFFF)
        return low.astype("<u4").tobytes()
