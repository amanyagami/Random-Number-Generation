"""Continuous health tests for a noise source (NIST SP 800-90B, section 4.4)."""

from __future__ import annotations

import math

import numpy as np


class HealthTestError(RuntimeError):
    """Raised when the noise source fails a continuous health test."""


def rct_cutoff(h: float, alpha_exp: int = 40) -> int:
    """Repetition Count Test cutoff ``C = 1 + ceil(alpha_exp / H)``."""
    return 1 + math.ceil(alpha_exp / h)


def apt_cutoff(h: float, window: int = 512, alpha_exp: int = 40) -> int:
    """Adaptive Proportion Test cutoff.

    Returns the smallest ``c`` with ``P(Bin(window, 2^-h) >= c) <= 2^-alpha_exp``.
    """
    p = 2.0**-h
    alpha = 2.0**-alpha_exp
    tail = 0.0
    for c in range(window, 0, -1):
        log_pmf = (
            math.lgamma(window + 1)
            - math.lgamma(c + 1)
            - math.lgamma(window - c + 1)
            + c * math.log(p)
            + (window - c) * math.log1p(-p)
        )
        tail += math.exp(log_pmf)
        if tail > alpha:
            return c + 1
    return 1


class HealthTests:
    """Repetition Count and Adaptive Proportion tests over a symbol stream.

    SP 800-90B suggests a false-positive rate of 2^-20 per sample, which would make a
    healthy source fail about once per eight million samples. A generator that raises
    on failure needs a lower rate, so 2^-40 is used; a stuck source is still caught
    after ``1 + 40 / H`` samples.

    Args:
        assumed_h: Min-entropy per symbol claimed for the source, in bits. Lower is more
            tolerant. Defaults to 2 bits for the low bits of microphone samples.
        window: Adaptive Proportion window (512 for non-binary sources).
    """

    def __init__(self, assumed_h: float = 2.0, window: int = 512) -> None:
        self.rct_c = rct_cutoff(assumed_h)
        self.apt_c = apt_cutoff(assumed_h, window)
        self.window = window

    def check(self, symbols: np.ndarray) -> None:
        """Run both tests over ``symbols``.

        Args:
            symbols: 1-D integer array.

        Raises:
            HealthTestError: If a run or a symbol proportion exceeds its cutoff.
        """
        s = np.asarray(symbols)
        if len(s) == 0:
            return
        change = np.flatnonzero(np.diff(s) != 0)
        edges = np.concatenate(([-1], change, [len(s) - 1]))
        longest = int(np.diff(edges).max())
        if longest >= self.rct_c:
            raise HealthTestError(f"repetition count test: run of {longest} >= {self.rct_c}")
        for start in range(0, len(s) - self.window + 1, self.window):
            w = s[start : start + self.window]
            count = int((w == w[0]).sum())
            if count >= self.apt_c:
                raise HealthTestError(
                    f"adaptive proportion test: {count}/{self.window} equal to first sample "
                    f"(cutoff {self.apt_c})"
                )
