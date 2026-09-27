"""Fail closed unless port 8766 is the real region-graph Jev bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav_jev_bridge.server import implementation_hash
from urllib.error import URLError
from urllib.request import urlopen


EXPECTED_MODEL = "typesafe/jev-1.13"


def validate_health(health):
    if not isinstance(health,dict): raise ValueError("health must be an object")
    provider = health.get("provider") if isinstance(health, dict) else None
    errors: list[str] = []
    if health.get("status") != "ok":
        errors.append("status is not ok")
    if not isinstance(provider, dict) or provider.get("name") != "openrouter":
        errors.append("provider is not openrouter")
    if not isinstance(provider, dict) or provider.get("model") != EXPECTED_MODEL:
        errors.append(f"requested model is not {EXPECTED_MODEL}")
    if health.get("state_profile") != "region_graph":
        errors.append("state_profile is not region_graph")
    if health.get("schema_version") != "jev-region-graph/1.1":
        errors.append("schema_version is not jev-region-graph/1.1")
    if health.get("trace_enabled") is not True:
        errors.append("trace is not enabled")
    if not isinstance(provider,dict) or provider.get("diagnostics_version") != "region-audit-v2":
        errors.append("bridge code is stale; restart required")
    if isinstance(provider,dict) and provider.get("implementation_sha256") != implementation_hash():
        errors.append("running bridge source differs from working tree; restart required")
    if not health.get("trace_dir"):
        errors.append("trace directory is missing")
    if errors:
        raise ValueError("Jev preflight refused: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/health")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    try:
        with urlopen(args.url, timeout=args.timeout) as response:
            health = json.load(response)
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Jev preflight refused: health endpoint unavailable or invalid ({type(exc).__name__})")
    validate_health(health)
    print(json.dumps(health, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
