"""Entropy-accounted generator: noise -> health tests -> CCML mixing -> SHA3-256."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from trng.ccml import CCML
from trng.entropy import min_entropy_markov, min_entropy_mcv, serial_correlation
from trng.health import HealthTests
from trng.source import Source

NOISE_BITS_PER_SAMPLE = 2
"""Only the low bits of a microphone sample are treated as noise."""


BLOCK_BYTES = 32
"""Bytes produced per conditioned block (one SHA3-256 digest)."""

SAMPLE_RATE = 44100
"""Default microphone sample rate in Hz, used for recording-time estimates."""

PRESET_COUNTS = (1, 10, 100, 1000, 10000, 100000)
"""Counts offered in the CLI help; any positive integer is accepted."""


@dataclass(frozen=True)
class Plan:
    """What it takes to produce a requested amount of random numbers.

    Attributes:
        count: Numbers requested.
        bits_per_number: Bits needed to represent one number.
        blocks: 32-byte conditioned blocks to generate (rejection sampling excluded).
        samples: Raw audio samples to record.
        seconds: Recording time at ``SAMPLE_RATE`` for those samples.
    """

    count: int
    bits_per_number: int
    blocks: int
    samples: int
    seconds: float


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
        self.on_block: Callable[[int], None] | None = None
        self._pool = bytearray()
        self._blocks_done = 0

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
        digest = hashlib.sha3_256(samples.tobytes() + bytes(mixed)).digest()
        self._blocks_done += 1
        if self.on_block is not None:
            self.on_block(self._blocks_done)
        return digest

    def read(self, n: int) -> bytes:
        """Return ``n`` random bytes (leftover bytes of a block are kept for the next call)."""
        while len(self._pool) < n:
            self._pool += self.block()
        out = bytes(self._pool[:n])
        del self._pool[:n]
        return out

    def plan(self, count: int, bits: int = 32, rate: int = SAMPLE_RATE) -> Plan:
        """Estimate the recording needed for ``count`` numbers of ``bits`` bits each.

        Args:
            count: How many numbers the user wants (for example 1, 10, 100, ... 100000).
            bits: Bits per number.
            rate: Microphone sample rate in Hz.

        Returns:
            A :class:`Plan` with the number of blocks, samples and seconds of audio.
        """
        if count < 1 or bits < 1:
            raise ValueError("count and bits must be positive")
        needed = count * math.ceil(bits / 8)
        blocks = max(0, math.ceil((needed - len(self._pool)) / BLOCK_BYTES))
        samples = blocks * self.block_samples
        return Plan(count, bits, blocks, samples, samples / rate)

    def integers(self, count: int, low: int, high: int) -> list[int]:
        """Return ``count`` unbiased integers uniformly drawn from ``[low, high]``.

        Uses rejection sampling, so there is no modulo bias.

        Args:
            count: How many integers to return.
            low: Smallest value (inclusive).
            high: Largest value (inclusive).
        """
        if high < low or count < 0:
            raise ValueError("need count >= 0 and high >= low")
        span = high - low + 1
        nbits = (span - 1).bit_length()
        nbytes = max(1, math.ceil(nbits / 8))
        mask = (1 << nbits) - 1
        out: list[int] = []
        while len(out) < count:
            v = int.from_bytes(self.read(nbytes), "big") & mask
            if v < span:
                out.append(low + v)
        return out
