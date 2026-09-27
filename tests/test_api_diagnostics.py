import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from nav_jev_bridge.api_diagnostics import MAX_ERROR_BODY_BYTES, http_error_evidence, encode_request
from nav_jev_bridge.bridge import OpenRouterJevProvider, ProviderError
from nav_jev_bridge.server import make_handler
from tools.diagnose_jev_api import build_cases, conclusion


class ApiDiagnosticTests(unittest.TestCase):
    def test_explicit_context_error_preserved_with_request_size_and_redaction(self):
        key = 'sk-or-unit-test-secret'
        error = HTTPError('https://openrouter.ai/api/alpha/decisions', 400, 'bad',
                          {'x-request-id': 'request-123', 'Authorization': key},
                          io.BytesIO(json.dumps({'error': {
                              'message': 'Provider rejected request',
                              'metadata': {'provider_name': 'TypeSafe',
                                           'raw': json.dumps({'error': {'code': 'context_length_exceeded',
                                               'message': 'Maximum context length is 32000 tokens. Bearer ' + key}}),
                                           'headers': {'Authorization': key}}
                          }}).encode()))
        payload = {'model': 'typesafe/jev-1.13', 'state': {'unchanged': [1.2345678901234567]}}
        with patch('nav_jev_bridge.bridge.urllib.request.urlopen', side_effect=error) as call:
            with self.assertRaises(ProviderError) as ctx:
                OpenRouterJevProvider(api_key=key).request_json(payload)
        self.assertEqual(ctx.exception.code, 'INPUT_CAPACITY_EXCEEDED')
        evidence = ctx.exception.diagnostics
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(evidence['http_status'], 400)
        self.assertEqual(evidence['classification'], 'context_length_reported_by_upstream')
        self.assertEqual(evidence['request_bytes'], len(encode_request(payload)))
        self.assertEqual(call.call_args.args[0].data, encode_request(payload))
        serialized = json.dumps(evidence)
        self.assertNotIn(key, serialized)
        self.assertNotIn('Authorization', serialized)
        self.assertIn('32000', serialized)
        self.assertEqual(evidence['request_ids'], {'x-request-id': 'request-123'})

    def test_generic_400_does_not_become_context_error(self):
        error = HTTPError('url', 400, 'bad', {}, io.BytesIO(b'{"error":{"message":"Invalid question"}}'))
        evidence = http_error_evidence(error, {}, '')
        self.assertEqual(evidence['classification'], 'bad_request_cause_unconfirmed')
        self.assertEqual(evidence['error_details']['error']['message'], 'Invalid question')

    def test_large_nonjson_error_is_bounded_and_secret_is_redacted(self):
        error = HTTPError('url', 413, 'large', {},
                          io.BytesIO(b'api_key=secret123 ' + b'x' * (MAX_ERROR_BODY_BYTES + 100)))
        evidence = http_error_evidence(error, {}, 'secret123')
        self.assertTrue(evidence['body_truncated'])
        self.assertLess(len(json.dumps(evidence)), 6000)
        self.assertNotIn('secret123', json.dumps(evidence))
        self.assertEqual(evidence['classification'], 'request_body_too_large_not_necessarily_token_limit')

    def test_failure_reading_error_body_does_not_mask_status(self):
        error = HTTPError('url', 502, 'bad gateway', {}, io.BytesIO())
        with patch.object(error, 'read', side_effect=TimeoutError):
            evidence = http_error_evidence(error, {}, '')
        self.assertEqual(evidence['http_status'], 502)
        self.assertEqual(evidence['body_read_error'], 'TimeoutError')

    def test_trial_builders_leave_originals_and_full_criteria_unchanged(self):
        failed = {'model': 'typesafe/jev-1.13', 'state': {'target_object': 'chair'},
                  'questions': {'next_goal': {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}}}}
        original = encode_request(failed)
        cases = dict(build_cases(failed, failed))
        self.assertEqual(encode_request(cases['original_failed_exact']), original)
        self.assertEqual(encode_request(failed), original)
        self.assertEqual(cases['minimal_state_original_criteria']['questions'], failed['questions'])
        self.assertEqual(cases['synthetic_minimal_two_choices']['questions'], cases['synthetic_same_two_choices_large_text']['questions'])

    def test_synthetic_context_error_does_not_prove_original_cause(self):
        rows = [{'case': 'original_failed_exact', 'outcome': 'error',
                 'upstream': {'classification': 'bad_request_cause_unconfirmed'}},
                {'case': 'synthetic_same_two_choices_large_text', 'outcome': 'error',
                 'upstream': {'classification': 'context_length_reported_by_upstream'}}]
        self.assertIn('requires inspection', conclusion(rows))

    def test_upstream_error_survives_handler_response_and_trace(self):
        evidence = {'http_status': 400, 'classification': 'bad_request_cause_unconfirmed',
                    'error_details': {'error': {'message': 'Invalid question field'}}}
        class FailingProvider:
            def choose_request(self, payload):
                raise ProviderError('PROVIDER_HTTP_ERROR', 'HTTP 400', retryable=False, diagnostics=evidence)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(FailingProvider(), 0, 0, Path(directory)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urlopen(f'http://127.0.0.1:{server.server_port}/health') as response:
                    health = json.load(response)
                self.assertEqual(health['provider']['name'], 'custom')
                self.assertIsNone(health['provider']['model'])
                request = Request(f'http://127.0.0.1:{server.server_port}/decide',
                                  data=(root / 'examples/snapshot.json').read_bytes(),
                                  headers={'Content-Type': 'application/json'})
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(request)
                self.assertEqual(json.load(ctx.exception)['upstream'], evidence)
                trace = json.loads(next(Path(directory).glob('*.json')).read_text(encoding='utf-8'))
                self.assertEqual(trace['decision']['upstream'], evidence)
                self.assertEqual(trace['provider'], health['provider'])
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)

if __name__ == '__main__':
    unittest.main()
