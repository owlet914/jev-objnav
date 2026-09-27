"""Send a C++-generated Jev snapshot through the production HTTP handler.

The provider records the exact model request and deterministically chooses the first criterion.
This is an interface fixture, not a Jev-quality or navigation-success test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nav_jev_bridge.server import make_handler


class RecordingProvider:
    def __init__(self) -> None:
        self.model_request: Optional[dict[str, Any]] = None

    def choose_request(self, payload: dict[str, Any]):
        self.model_request = payload
        criteria = payload["questions"]["next_goal"]["criteria"]
        choice = next(iter(criteria))
        return choice, 1.0, {choice: 1.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    snapshot_bytes = args.snapshot.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    provider = RecordingProvider()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(provider, 0.0, 0.0, state_profile="full")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/decide",
            data=json.dumps(snapshot, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            decision = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    if provider.model_request is None:
        raise RuntimeError("provider did not receive a model request")
    evidence = {
        "evidence_type": "compiled C++ fixture -> production HTTP handler -> recording provider",
        "limitations": [
            "recording provider is deterministic and is not Jev",
            "fixture validates serialization and transport, not ROS timing or closed-loop navigation",
        ],
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "input_counts": snapshot["coverage"]["entity_counts"],
        "decision": decision,
        "model_request": provider.model_request,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
