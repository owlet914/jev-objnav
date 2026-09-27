import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jev_obj"))

from vlm import server_wrapper


class ServerWrapperTests(unittest.TestCase):
    def test_retry_does_not_terminate_process(self):
        with (
            patch(
                "vlm.server_wrapper._send_request",
                side_effect=[TimeoutError("slow"), {"ok": True}],
            ) as request,
            patch("vlm.server_wrapper.time.sleep"),
        ):
            response = server_wrapper.send_request(
                "http://localhost/model",
                max_attempts=2,
                retry_delay_s=0,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.kwargs["request_timeout_s"], 30.0)

    def test_exhausted_retries_raise_runtime_error(self):
        with (
            patch(
                "vlm.server_wrapper._send_request",
                side_effect=TimeoutError("slow"),
            ) as request,
            patch("vlm.server_wrapper.time.sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
                server_wrapper.send_request(
                    "http://localhost/model",
                    max_attempts=2,
                    retry_delay_s=0,
                )

        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
