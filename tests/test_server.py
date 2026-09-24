import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from nav_jev_bridge.bridge import DemoProvider
from nav_jev_bridge.server import make_handler


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "snapshot.json"


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


if __name__ == "__main__":
    unittest.main()
