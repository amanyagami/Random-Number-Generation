"""Conservative entropy estimates for raw noise symbols."""

from __future__ import annotations

import math

import numpy as np


def min_entropy_mcv(symbols: np.ndarray, z: float = 2.576) -> float:
    """Most-common-value min-entropy estimate (NIST SP 800-90B, section 6.3.1).

    This is one of the ten SP 800-90B estimators and is only an upper bound on what a
    full assessment would credit. It is conservative for biased sources and blind to
    serial dependence; use :func:`serial_correlation` alongside it.

    Args:
        symbols: 1-D array of non-negative integer symbols.
        z: Normal quantile for the 99% upper confidence bound.

    Returns:
        Estimated min-entropy in bits per symbol.
    """
    n = len(symbols)
    if n < 2:
        raise ValueError("need at least 2 symbols")
    counts = np.bincount(np.asarray(symbols, dtype=np.int64))
    p_hat = counts.max() / n
    p_upper = min(1.0, p_hat + z * math.sqrt(p_hat * (1.0 - p_hat) / (n - 1)))
    return -math.log2(p_upper)


def serial_correlation(symbols: np.ndarray) -> float:
    """Lag-1 autocorrelation of ``symbols`` (0 for independent samples)."""
    x = np.asarray(symbols, dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0.0:
        return 1.0
    return float((x[:-1] * x[1:]).sum() / denom)


def bit_bias(data: bytes) -> tuple[float, int]:
    """Return the overall 1-bit frequency and the count of never-set bit positions per byte."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr).reshape(-1, 8)
    freq = bits.mean(axis=0)
    return float(bits.mean()), int((freq == 0).sum())


def min_entropy_markov(symbols: np.ndarray, k: int | None = None, d: int = 128) -> float:
    """First-order Markov min-entropy estimate (NIST SP 800-90B, section 6.3.3).

    Catches sources whose next symbol is predictable from the previous one, which the
    most-common-value estimate cannot see (e.g. a slowly varying deterministic signal).

    Args:
        symbols: 1-D array of integer symbols in ``[0, k)``.
        k: Alphabet size; inferred from the data when omitted.
        d: Sequence length for the most-probable-path search (128 in the standard).

    Returns:
        Estimated min-entropy in bits per symbol, at most ``log2(k)``.
    """
    s = np.asarray(symbols, dtype=np.int64)
    k = int(k or s.max() + 1)
    n = len(s)
    if n < 2:
        raise ValueError("need at least 2 symbols")
    alpha = min(0.99 ** (k * k), 0.99**d)
    init = np.bincount(s, minlength=k) / n
    eps0 = math.sqrt(math.log(1.0 / (1.0 - alpha)) / (2.0 * n))
    p0 = np.minimum(1.0, init + eps0)
    trans = np.zeros((k, k))
    np.add.at(trans, (s[:-1], s[1:]), 1.0)
    row = trans.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(row[:, None] > 0, trans / row[:, None], 0.0)
        eps = np.where(row > 0, np.sqrt(math.log(1.0 / (1.0 - alpha)) / (2.0 * row)), 0.0)
    t = np.minimum(1.0, t + eps[:, None])
    log_t = np.log2(np.maximum(t, 1e-300))
    best = np.log2(np.maximum(p0, 1e-300))
    for _ in range(d - 1):
        best = np.max(best[:, None] + log_t, axis=0)
    h = -float(best.max()) / d
    return min(h, math.log2(k))
