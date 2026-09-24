"""Local HTTP endpoint for Jev Obj's C++ planner (Python 3.10+ process)."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .bridge import STATE_PROFILES, DemoProvider, OpenRouterJevProvider, decide


MAX_BODY_BYTES = 1024 * 1024


def make_handler(
    provider: Any, min_confidence: float, min_probability: float,
    trace_dir: Path | None = None,
    state_profile: str = "full",
):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/decide":
                self._send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("request body is empty or too large")
                snapshot = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                result = decide(
                    snapshot,
                    provider,
                    min_confidence=min_confidence,
                    min_probability=min_probability,
                    state_profile=state_profile,
                )
                if trace_dir is not None:
                    trace = {
                        "state_profile": state_profile,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                        "snapshot": snapshot,
                        "decision": result,
                    }
                    try:
                        (trace_dir / f"{uuid.uuid4().hex}.json").write_text(
                            json.dumps(trace, ensure_ascii=False, allow_nan=False),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass  # Logging must not change navigation behavior.
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"status": "HOLD", "reason": "invalid_snapshot", "detail": str(exc)})
                return
            self._send_json(200, result)

        def log_message(self, format: str, *args: object) -> None:
            # Do not log state payloads or API keys.
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Jev decision endpoint for Jev Obj")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--provider", choices=("demo", "openrouter"), default="openrouter")
    parser.add_argument("--jev-timeout", type=float, default=4.0)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--state-profile", choices=STATE_PROFILES, default="full")
    parser.add_argument("--trace-dir", type=Path, help="save decision snapshots for a pilot run")
    args = parser.parse_args()

    if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for --provider openrouter")
    if args.jev_timeout <= 0:
        parser.error("--jev-timeout must be positive")

    provider = OpenRouterJevProvider(timeout_s=args.jev_timeout) if args.provider == "openrouter" else DemoProvider()
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(provider, args.min_confidence, args.min_probability, args.trace_dir, args.state_profile),
    )
    print(f"Jev Obj bridge listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
