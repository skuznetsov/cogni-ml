#!/usr/bin/env python3
"""Compare a native trace bundle with pinned-reference traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spec.support.drama3.reference_compare import compare_bundles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--prefix", action="append", default=[], help="compare only trace names starting with this prefix; repeatable")
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--rtol", type=float, default=3e-4)
    args = parser.parse_args()
    try:
        result = compare_bundles(
            args.reference,
            args.actual,
            atol=args.atol,
            rtol=args.rtol,
            prefixes=tuple(args.prefix),
        )
    except (AssertionError, ValueError) as exc:
        print(json.dumps({"status": "mismatch", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
