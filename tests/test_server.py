import json
import http.client
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from nav_jev_bridge.bridge import DemoProvider
from nav_jev_bridge.server import MAX_BODY_BYTES, make_handler
from test_region_graph_contract import region_snapshot


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "snapshot.json"
PARITY_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "jev_parity" / "cpp_snapshot.json"


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DemoProvider(), 0.4, 0.5))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_decision_http_contract(self):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/decide",
            data=EXAMPLE.read_bytes(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=2) as response:
            result = json.load(response)
        self.assertEqual(result["status"], "GOAL")
        self.assertEqual(result["map_revision"], 1)

    def test_invalid_snapshot_returns_400(self):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/decide",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=2)
        self.assertEqual(context.exception.code, 400)

    def test_full_context_validation_error_writes_auditable_trace(self):
        snapshot = json.loads(PARITY_EXAMPLE.read_text(encoding="utf-8"))
        snapshot["coverage"]["high_level_evidence_complete"] = False
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(DemoProvider(), 0.4, 0.5, trace_dir, "full")
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/decide",
                    data=json.dumps(snapshot).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=2)
                self.assertEqual(context.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            trace_files = list(trace_dir.glob("*.json"))
            self.assertEqual(len(trace_files), 1)
            trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
            self.assertEqual(trace["decision"]["error_code"], "INCOMPLETE_FULL_CONTEXT")
            self.assertEqual(trace["snapshot"]["request_id"], snapshot["request_id"])
            self.assertIn("complete high-level evidence", trace["decision"]["detail"])

    def test_oversized_request_is_nonretryable_capacity_error(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        connection.putrequest("POST", "/decide")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        payload = json.load(response)
        connection.close()
        self.assertEqual(response.status, 413)
        self.assertEqual(payload["error_code"], "INPUT_CAPACITY_EXCEEDED")
        self.assertFalse(payload["retryable"])

    def test_invalid_provider_response_trace_uses_exact_model_request(self):
        class InvalidProvider:
            def choose(self, state, options):
                return "hold", 1.0, {"hold": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(InvalidProvider(), 0.4, 0.5, trace_dir)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/decide",
                    data=EXAMPLE.read_bytes(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=2)
                self.assertEqual(context.exception.code, 503)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            trace_files = list(trace_dir.glob("*.json"))
            self.assertEqual(len(trace_files), 1)
            trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
            self.assertEqual(trace["decision"]["error_code"], "INVALID_PROVIDER_RESPONSE")
            self.assertEqual(
                set(trace["model_request"]["questions"]["next_goal"]["criteria"]),
                set(trace["eligible_candidate_ids"]),
            )
            serialized = json.dumps(trace)
            self.assertNotIn("Authorization", serialized)
            self.assertNotIn("OPENROUTER_API_KEY", serialized)

    def test_region_graph_single_flight_calls_provider_once(self):
        class SlowProvider:
            def __init__(self):
                self.calls = 0
                self.lock = threading.Lock()

            def choose_request(self, payload):
                with self.lock:
                    self.calls += 1
                time.sleep(0.15)
                return "approach:object-7-goal", 0.8, {
                    "approach:object-7-goal": 0.8,
                    "explore:frontier-2-goal": 0.2,
                }

        provider = SlowProvider()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(provider, 0.0, 0.0, None, "region_graph")
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = json.dumps(region_snapshot()).encode("utf-8")
        results = []

        def call():
            request = Request(
                f"http://127.0.0.1:{server.server_port}/decide", data=body,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=3) as response:
                results.append(json.load(response))

        callers = [threading.Thread(target=call) for _ in range(2)]
        try:
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(timeout=3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual({item["candidate_id"] for item in results}, {"object-7-goal"})
        self.assertEqual(sum(item.get("single_flight_cache_hit", False) for item in results), 1)

    def test_trace_failure_stops_audited_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            not_a_directory = Path(directory) / "trace-file"
            not_a_directory.write_text("occupied", encoding="utf-8")
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(DemoProvider(), 0.4, 0.5, not_a_directory)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/decide", data=EXAMPLE.read_bytes(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=2)
                self.assertEqual(context.exception.code, 500)
                payload = json.load(context.exception)
                self.assertEqual(payload["error_code"], "TRACE_WRITE_FAILED")
                self.assertFalse(payload["retryable"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_malformed_envelopes_return_nonretryable_schema_error(self):
        for body in (b'[]', b'{"region_graph":null}', b'{"region_graph":{"identity":[]}}'):
            request=Request(f'http://127.0.0.1:{self.server.server_port}/decide',data=body,headers={'Content-Type':'application/json'})
            with self.subTest(body=body), self.assertRaises(HTTPError) as ctx:
                urlopen(request,timeout=2)
            self.assertEqual(ctx.exception.code,400)
            self.assertFalse(json.load(ctx.exception)['retryable'])


if __name__ == "__main__":
    unittest.main()
