"""Command line: choose how many random numbers you want; the tool records as long as needed."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from trng.generator import PRESET_COUNTS, SAMPLE_RATE, Generator
from trng.source import CallableSource, MicSource, Source, WavSource


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} min {int(seconds % 60)} s"


def main(argv: list[str] | None = None) -> int:
    """Entry point; prints one random number per line (or raw bytes with ``--raw``)."""
    presets = ", ".join(str(c) for c in PRESET_COUNTS)
    ap = argparse.ArgumentParser(prog="trng", description=__doc__)
    ap.add_argument("--count", "-n", type=int, default=1, help=f"how many numbers ({presets}, ...)")
    ap.add_argument("--bits", type=int, default=32, help="bits per number (default 32)")
    ap.add_argument("--min", type=int, dest="low", help="smallest value, with --max (e.g. 1)")
    ap.add_argument("--max", type=int, dest="high", help="largest value, with --min (e.g. 6)")
    ap.add_argument("--hex", action="store_true", help="print numbers in hexadecimal")
    ap.add_argument("--raw", action="store_true", help="write count*ceil(bits/8) raw bytes")
    ap.add_argument("--source", default="mic", help="'mic' or path to a 16-bit WAV file")
    ap.add_argument("--rate", type=int, default=SAMPLE_RATE, help="microphone sample rate")
    ap.add_argument("--dry-run", action="store_true", help="only print the recording time")
    args = ap.parse_args(argv)
    if (args.low is None) != (args.high is None):
        ap.error("--min and --max must be given together")
    if args.count < 1:
        ap.error("--count must be at least 1")

    bits = (args.high - args.low).bit_length() if args.low is not None else args.bits
    src: Source
    if args.dry_run:
        src = CallableSource(lambda n: np.zeros(n, dtype=np.int16))  # never read
    else:
        src = MicSource(args.rate) if args.source == "mic" else WavSource(args.source)
    gen = Generator(src)
    plan = gen.plan(args.count, max(1, bits), args.rate)
    print(
        f"{args.count} number(s) x {plan.bits_per_number} bits -> {plan.blocks} blocks, "
        f"recording {_fmt_duration(plan.seconds)} of audio",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0

    def progress(done: int) -> None:
        if plan.blocks and (done == plan.blocks or done % max(1, plan.blocks // 50) == 0):
            print(f"\r  {100 * done // plan.blocks:3d}%", end="", file=sys.stderr, flush=True)

    gen.on_block = progress
    try:
        if args.raw:
            sys.stdout.buffer.write(gen.read(args.count * ((bits + 7) // 8)))
        else:
            if args.low is not None:
                numbers = gen.integers(args.count, args.low, args.high)
            else:
                numbers = gen.integers(args.count, 0, (1 << args.bits) - 1)
            for n in numbers:
                print(f"{n:x}" if args.hex else n)
    except EOFError:
        print(
            f"\nrecording too short: need about {_fmt_duration(plan.seconds)} of audio",
            file=sys.stderr,
        )
        return 2
    print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
