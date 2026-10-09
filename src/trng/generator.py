"""Entropy-accounted generator: noise -> health tests -> CCML mixing -> SHA3-256."""

from __future__ import annotations

import hashlib

import numpy as np

from trng.ccml import CCML
from trng.entropy import min_entropy_markov, min_entropy_mcv, serial_correlation
from trng.health import HealthTests
from trng.source import Source

NOISE_BITS_PER_SAMPLE = 2
"""Only the low bits of a microphone sample are treated as noise."""


class InsufficientEntropyError(RuntimeError):
    """Raised when the measured source entropy cannot back a full output block."""


class Generator:
    """True random generator that refuses to output more than it can justify.

    Each output block is the SHA3-256 digest of the raw noise bytes and the CCML-mixed
    state. The CCML provides diffusion; the security argument rests on the measured
    min-entropy of the raw noise and the hash conditioner, as in NIST SP 800-90B.

    Args:
        source: Raw sample source.
        block_samples: Raw samples consumed per 32-byte output block.
        safety: Fraction of the estimated min-entropy that is credited (0 < safety <= 1).
        health: Health tests; a default instance is created when omitted.
        ccml: Mixing lattice; a default instance is created when omitted.
    """

    def __init__(
        self,
        source: Source,
        block_samples: int = 4096,
        safety: float = 0.5,
        health: HealthTests | None = None,
        ccml: CCML | None = None,
    ) -> None:
        if not 0 < safety <= 1:
            raise ValueError("safety must be in (0, 1]")
        self.source = source
        self.block_samples = block_samples
        self.safety = safety
        self.health = health or HealthTests()
        self.ccml = ccml or CCML()
        self.last_min_entropy: float = 0.0

    def block(self) -> bytes:
        """Return one 32-byte block of conditioned random bytes.

        Raises:
            HealthTestError: If the source fails a continuous health test.
            InsufficientEntropyError: If credited entropy is below 256 bits per block.
        """
        samples = np.asarray(self.source.read(self.block_samples), dtype=np.int16)
        noise = (samples & ((1 << NOISE_BITS_PER_SAMPLE) - 1)).astype(np.int64)
        self.health.check(noise)
        h = min(min_entropy_mcv(noise), min_entropy_markov(noise, k=1 << NOISE_BITS_PER_SAMPLE))
        self.last_min_entropy = h
        if abs(serial_correlation(noise)) > 0.1:
            h *= 0.5  # strongly dependent samples carry less than the estimators say
        credited = h * len(noise) * self.safety
        if credited < 256:
            raise InsufficientEntropyError(
                f"{credited:.0f} credited bits < 256 per block (min-entropy {h:.2f} "
                f"bits/sample); use more samples per block or a better source"
            )
        low8 = (samples & 0xFF).astype(np.int64)
        for start in range(0, len(low8), self.ccml.size):
            self.ccml.inject(low8[start : start + self.ccml.size])
            self.ccml.step()
        mixed = bytearray()
        for _ in range(4):
            self.ccml.step()
            mixed += self.ccml.mantissa_bytes()
        return hashlib.sha3_256(samples.tobytes() + bytes(mixed)).digest()

    def read(self, n: int) -> bytes:
        """Return ``n`` random bytes."""
        out = bytearray()
        while len(out) < n:
            out += self.block()
        return bytes(out[:n])
