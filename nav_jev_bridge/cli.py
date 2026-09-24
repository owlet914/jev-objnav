"""Evaluate one JSON snapshot from a file or standard input."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .bridge import STATE_PROFILES, DemoProvider, OpenRouterJevProvider, decide


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a high-level navigation goal from a structured snapshot")
    parser.add_argument("snapshot", help="JSON snapshot path, or '-' for stdin")
    parser.add_argument("--provider", choices=("demo", "openrouter"), default="demo")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--state-profile", choices=STATE_PROFILES, default="full")
    args = parser.parse_args()

    if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("set OPENROUTER_API_KEY before using --provider openrouter")

    try:
        raw = sys.stdin.read() if args.snapshot == "-" else Path(args.snapshot).read_text(encoding="utf-8")
        snapshot = json.loads(raw)
        result = decide(
            snapshot,
            OpenRouterJevProvider() if args.provider == "openrouter" else DemoProvider(),
            min_confidence=args.min_confidence,
            min_probability=args.min_probability,
            state_profile=args.state_profile,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid snapshot: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "GOAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
