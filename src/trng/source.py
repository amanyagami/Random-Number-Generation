"""Noise sources producing raw 16-bit PCM samples."""

from __future__ import annotations

import wave
from collections.abc import Callable
from typing import Protocol

import numpy as np


class Source(Protocol):
    """Anything that can return ``n`` raw int16 samples."""

    def read(self, n: int) -> np.ndarray:
        """Return ``n`` int16 samples."""
        ...


class WavSource:
    """Read samples from a mono or stereo 16-bit WAV recording (first channel used).

    Args:
        path: Path to a 16-bit PCM WAV file.
    """

    def __init__(self, path: str) -> None:
        """Load the whole recording into memory."""
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("only 16-bit PCM WAV files are supported")
            channels = w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        self._data = raw[::channels]
        self._pos = 0

    def read(self, n: int) -> np.ndarray:
        """Return the next ``n`` samples; raises ``EOFError`` when the file is exhausted."""
        if self._pos + n > len(self._data):
            raise EOFError("recording exhausted")
        out = self._data[self._pos : self._pos + n]
        self._pos += n
        return out


class MicSource:
    """Live microphone capture via ``sounddevice`` (install the ``audio`` extra).

    Args:
        rate: Sample rate in Hz.
    """

    def __init__(self, rate: int = 44100) -> None:
        """Open the default input device lazily on first read."""
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("install the 'audio' extra: uv sync --extra audio") from exc
        self._sd = sd
        self._rate = rate

    def read(self, n: int) -> np.ndarray:  # pragma: no cover - needs hardware
        """Record ``n`` mono int16 samples."""
        frames = self._sd.rec(n, samplerate=self._rate, channels=1, dtype="int16", blocking=True)
        return frames[:, 0]


class CallableSource:
    """Wrap a ``fn(n) -> int16 array`` (used for tests and simulation)."""

    def __init__(self, fn: Callable[[int], np.ndarray]) -> None:
        """Store the sample-producing callable."""
        self._fn = fn

    def read(self, n: int) -> np.ndarray:
        """Return ``n`` samples from the wrapped callable."""
        return self._fn(n)
