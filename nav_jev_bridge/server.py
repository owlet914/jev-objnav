"""Local HTTP endpoint for Jev Obj's C++ planner (Python 3.10+ process)."""

from __future__ import annotations

import argparse
import json
import os
import logging
import time
import hashlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from .bridge import (
    STATE_PROFILES,
    DemoProvider,
    OpenRouterJevProvider,
    ProviderError,
    SnapshotError,
    decide,
)
from .api_diagnostics import request_facts, encode_request


DIAGNOSTICS_VERSION = "region-audit-v2"


MAX_BODY_BYTES = int(os.environ.get("JEV_MAX_BODY_BYTES", 8 * 1024 * 1024))


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for name in ('server.py', 'bridge.py', 'region_graph.py', 'api_diagnostics.py'):
        digest.update(name.encode('ascii'))
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def make_handler(
    provider: Any, min_confidence: float, min_probability: float,
    trace_dir: Optional[Path] = None,
    state_profile: str = "full",
):
    coordination_lock = threading.Lock()
    inflight: dict[str, tuple[str, threading.Event]] = {}
    completed: dict[str, tuple[str, float, int, dict[str, Any]]] = {}
    attempts: dict[str,int] = {}
    provider_info = {
        "name": "openrouter" if isinstance(provider, OpenRouterJevProvider) else
                "demo" if isinstance(provider, DemoProvider) else "custom",
        "model": provider.model if isinstance(provider, OpenRouterJevProvider) else None,
        "pid": os.getpid(),
        "instance_id": uuid.uuid4().hex,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "implementation_sha256": implementation_hash(),
    }

    class Handler(BaseHTTPRequestHandler):
        def _write_trace(self, request_snapshot: dict[str, Any], started: float, **extra: Any) -> bool:
            if trace_dir is None:
                return True
            region_state = request_snapshot.get("region_graph", {})
            source_map = request_snapshot.get("map", {})
            if not isinstance(source_map,dict): source_map = {}
            identity = region_state.get("identity", {}) if isinstance(region_state, dict) else {}
            coverage = region_state.get("coverage", {}) if isinstance(region_state, dict) else {}
            trace = {
                "provider": provider_info,
                "recorded_at_unix_s": time.time(),
                "attempt_index": getattr(self, "attempt_index", 1),
                "schema_version": region_state.get("schema_version", request_snapshot.get("schema_version")),
                "state_profile": state_profile,
                "request_id": identity.get("request_id", request_snapshot.get("request_id")),
                "episode_id": identity.get("episode_id", request_snapshot.get("episode_id")),
                "observation_id": identity.get("observation_id", request_snapshot.get("observation_id")),
                "map_revision": identity.get("map_revision", source_map.get("revision")),
                "region_graph_revision": identity.get("region_graph_revision"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "input_counts": coverage if coverage else request_snapshot.get("coverage", {}).get("entity_counts"),
                "coverage": coverage if coverage else request_snapshot.get("coverage"),
                **extra,
            }
            model_request = extra.get("model_request")
            if isinstance(model_request, dict):
                trace["wire"] = request_facts(model_request)
            try:
                (trace_dir / f"{uuid.uuid4().hex}.json").write_text(
                    json.dumps(trace, ensure_ascii=False, allow_nan=False), encoding="utf-8"
                )
                return True
            except OSError as exc:
                logging.error("Decision trace write failed: %s", type(exc).__name__)
                return False

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "provider": provider_info,
                                      "state_profile": state_profile,
                                      "schema_version": "jev-region-graph/1.1"
                                      if state_profile == "region_graph" else None,
                                      "trace_enabled": trace_dir is not None,
                                      "trace_dir": str(trace_dir.resolve()) if trace_dir else None})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/decide":
                self._send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
                if length <= 0:
                    raise SnapshotError("SCHEMA_ERROR", "request body is empty")
                if length > MAX_BODY_BYTES:
                    self._send_json(413, {
                        "status": "ERROR",
                        "error_code": "INPUT_CAPACITY_EXCEEDED",
                        "retryable": False,
                        "request_bytes": length,
                        "max_body_bytes": MAX_BODY_BYTES,
                    })
                    return
                raw = self.rfile.read(length)
                snapshot = json.loads(raw)
                if not isinstance(snapshot, dict):
                    raise SnapshotError("SCHEMA_ERROR", "snapshot must be an object")
                if "region_graph" in snapshot:
                    graph = snapshot["region_graph"]
                    if not isinstance(graph,dict) or not isinstance(graph.get("identity"),dict):
                        raise SnapshotError("SCHEMA_ERROR", "region_graph.identity must be an object")
            except (SnapshotError, TypeError, ValueError, UnicodeError) as exc:
                self._send_json(400, {
                    "status": "ERROR",
                    "error_code": "SCHEMA_ERROR",
                    "retryable": False,
                    "detail": str(exc),
                })
                return

            region_identity = snapshot.get("region_graph", {}).get("identity", {})
            logical_request_id = region_identity.get("request_id", snapshot.get("request_id"))
            if logical_request_id is not None and (not isinstance(logical_request_id,str) or not logical_request_id):
                self._send_json(400, {"status":"ERROR","error_code":"SCHEMA_ERROR","retryable":False})
                return
            fingerprint_snapshot = dict(snapshot)
            # Wall-clock serialization time is not part of a full logical observation.
            if state_profile != "region_graph": fingerprint_snapshot.pop("timestamp_ms", None)
            try:
                fingerprint = hashlib.sha256(encode_request(fingerprint_snapshot)).hexdigest()
            except (ValueError, TypeError):
                self._send_json(400, {"status":"ERROR","error_code":"SCHEMA_ERROR","retryable":False})
                return
            owner = True
            wait_event = None
            if isinstance(logical_request_id, str) and logical_request_id:
                with coordination_lock:
                    now = time.monotonic()
                    cached = completed.get(logical_request_id)
                    if cached and cached[1] >= now:
                        if cached[0] != fingerprint:
                            self._send_json(409, {"status": "ERROR",
                                "error_code": "REQUEST_ID_CONFLICT", "retryable": False})
                        else:
                            self._send_json(cached[2], {**cached[3], "single_flight_cache_hit": True})
                        return
                    if cached:
                        completed.pop(logical_request_id, None)
                    current = inflight.get(logical_request_id)
                    if current:
                        if current[0] != fingerprint:
                            self._send_json(409, {"status": "ERROR",
                                "error_code": "REQUEST_ID_CONFLICT", "retryable": False})
                            return
                        owner = False
                        wait_event = current[1]
                    else:
                        wait_event = threading.Event()
                        inflight[logical_request_id] = (fingerprint, wait_event)
            if not owner and wait_event is not None:
                if not wait_event.wait(timeout=125.0):
                    self._send_json(503, {"status": "ERROR", "error_code": "SINGLE_FLIGHT_TIMEOUT",
                        "retryable": True})
                    return
                with coordination_lock:
                    cached = completed.get(logical_request_id)
                if cached and cached[0] == fingerprint:
                    self._send_json(cached[2], {**cached[3], "single_flight_cache_hit": True})
                else:
                    self._send_json(503, {"status": "ERROR", "error_code": "SINGLE_FLIGHT_RESULT_MISSING",
                        "retryable": True})
                return

            started = time.perf_counter()
            status = 500
            payload: dict[str, Any] = {"status":"ERROR", "error_code":"INTERNAL_CONTRACT_ERROR", "retryable":False}
            with coordination_lock:
                attempts[logical_request_id] = attempts.get(logical_request_id, 0) + 1
                self.attempt_index = attempts[logical_request_id]
            retryable = False
            try:
                result, decision_trace = decide(
                    snapshot, provider, min_confidence=min_confidence,
                    min_probability=min_probability, state_profile=state_profile, return_trace=True,
                )
                if not self._write_trace(
                    snapshot, started, model_request=decision_trace["model_request"],
                    eligible_candidate_ids=decision_trace["eligible_candidate_ids"],
                    upstream=decision_trace.get("provider_metadata", {}), decision=result,
                    local_candidate_registry=snapshot.get("candidates", []),
                ):
                    status = 500
                    payload = {"status": "ERROR", "error_code": "TRACE_WRITE_FAILED",
                               "retryable": False}
                else:
                    status, payload = 200, result
            except SnapshotError as exc:
                payload = {"status": "ERROR", "error_code": exc.code,
                           "retryable": False, "detail": str(exc)}
                status = 400
                if not self._write_trace(snapshot, started, snapshot=snapshot,
                                         eligible_candidate_ids=[], decision=payload):
                    status = 500
                    payload = {"status": "ERROR", "error_code": "TRACE_WRITE_FAILED",
                               "retryable": False}
            except ProviderError as exc:
                retryable = exc.retryable
                payload = {"status": "ERROR", "error_code": exc.code,
                           "retryable": exc.retryable, "detail": str(exc),
                           "upstream": exc.diagnostics}
                status = 503 if exc.retryable else 422
                if not self._write_trace(
                    snapshot, started, model_request=getattr(exc, "model_request", None),
                    eligible_candidate_ids=getattr(exc, "eligible_candidate_ids", []),
                    decision=payload,
                ):
                    status = 500
                    payload = {"status": "ERROR", "error_code": "TRACE_WRITE_FAILED",
                               "retryable": False}
                    retryable = False
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                status = 400
                payload = {"status":"ERROR", "error_code":"SCHEMA_ERROR", "retryable":False, "detail":str(exc)}
                if not self._write_trace(snapshot, started, decision=payload):
                    status, payload = 500, {"status":"ERROR", "error_code":"TRACE_WRITE_FAILED", "retryable":False}
            finally:
                if isinstance(logical_request_id, str) and logical_request_id and owner:
                    ttl = 0.0 if retryable else 120.0
                    with coordination_lock:
                        completed[logical_request_id] = (
                            fingerprint, time.monotonic() + ttl, status, payload,
                        )
                        current = inflight.pop(logical_request_id, None)
                        if current:
                            current[1].set()
            self._send_json(status, payload)

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
    parser.add_argument(
        "--min-confidence", type=float, default=0.4,
        help="deprecated compatibility value; validated but not used to gate a legal choice",
    )
    parser.add_argument(
        "--min-probability", type=float, default=0.5,
        help="deprecated compatibility value; validated but not used to gate a legal choice",
    )
    parser.add_argument("--state-profile", choices=STATE_PROFILES, default="full")
    parser.add_argument("--trace-dir", type=Path, help="save decision snapshots for a pilot run")
    args = parser.parse_args()

    if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for --provider openrouter")
    if args.provider == "demo" and args.state_profile == "region_graph":
        parser.error(
            "--state-profile region_graph requires --provider openrouter; "
            "the demo provider is not a Jev quality evaluation"
        )
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
