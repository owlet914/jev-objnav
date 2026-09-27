"""Rebuild and send one saved region-graph model request to real Jev.

This is a capacity/decision diagnostic. It never prints or stores the API key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nav_jev_bridge.bridge import OpenRouterJevProvider, build_model_request, encode_request
from nav_jev_bridge.region_graph import _compact_model_state, _criterion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--key-file", type=Path,
        default=Path.home() / ".config" / "jev-obj" / "openrouter.key",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    old_request = trace["model_request"]
    state = _compact_model_state(old_request["state"])
    entity_by_id = {}
    for container in [*state["regions"], state["unassigned_entities"]]:
        for group in ("objects", "frontiers"):
            for entity in container.get(group, []):
                entity_by_id[entity["id"]] = entity
    for entity in state.get("observation_actions", []):
        entity_by_id[entity["id"]] = entity
    criteria = {
        option["id"]: _criterion(option, entity_by_id.get(option["target_id"]))
        for option in state["options"]
    }
    request = build_model_request(state, criteria)
    key = args.key_file.read_text(encoding="utf-8").strip()
    provider = OpenRouterJevProvider(timeout_s=args.timeout, api_key=key)
    selected, confidence, probabilities, metadata = provider.choose_request_with_metadata(request)
    print(json.dumps({
        "status": "ok",
        "source_request_bytes": trace.get("wire", {}).get("request_bytes"),
        "compacted_request_bytes": len(encode_request(request)),
        "projection_summary": state.get("projection_summary"),
        "safe_option_count": len(criteria),
        "selected_option_id": selected,
        "confidence": confidence,
        "selected_probability": probabilities.get(selected),
        "actual_model": metadata.get("actual_model"),
        "generation_id": metadata.get("generation_id"),
        "usage": metadata.get("usage"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
