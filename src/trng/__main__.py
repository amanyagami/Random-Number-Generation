"""Command line: ``trng --source mic|<file.wav> --bytes N``."""

from __future__ import annotations

import argparse
import sys

from trng.generator import Generator
from trng.source import MicSource, Source, WavSource


def main(argv: list[str] | None = None) -> int:
    """Entry point; writes random bytes (hex unless ``--raw``) to stdout."""
    ap = argparse.ArgumentParser(prog="trng", description=__doc__)
    ap.add_argument("--source", default="mic", help="'mic' or path to a 16-bit WAV file")
    ap.add_argument("--bytes", type=int, default=32, dest="n")
    ap.add_argument("--raw", action="store_true", help="write raw bytes instead of hex")
    args = ap.parse_args(argv)
    src: Source = MicSource() if args.source == "mic" else WavSource(args.source)
    data = Generator(src).read(args.n)
    if args.raw:
        sys.stdout.buffer.write(data)
    else:
        print(data.hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
