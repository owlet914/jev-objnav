"""Independent, offline fault probes. Not a real Jev evaluation."""
from __future__ import annotations
import copy
import hashlib
import subprocess
import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_region_graph_contract import region_snapshot, ChoiceProvider
from nav_jev_bridge.bridge import prepare_decision, decide, ProviderError
from nav_jev_bridge.server import make_handler


def check(label, mutate):
    state = region_snapshot()
    mutate(state)
    try:
        result = decide(state, ChoiceProvider(), state_profile='region_graph')
        return {'case': label, 'rejected': False, 'status': result['status'],
                'returned_kind': result['kind'], 'returned_target': result['target_id']}
    except Exception as exc:
        return {'case': label, 'rejected': True, 'exception': type(exc).__name__, 'detail': str(exc)}


def main():
    results = []
    results.append(check('local_path_collision_false', lambda s: s['candidates'][1]['path'].update(collision_free=False)))
    results.append(check('local_mode_target_mismatch', lambda s: s['candidates'][1].update(kind='explore_frontier', target_id='frontier-2')))
    results.append(check('source_objects_total_mismatch', lambda s: s['region_graph']['coverage'].update(objects_total=999)))
    results.append(check('stale_verified_edge', lambda s: s['region_graph']['connections'][0].update(verified_map_revision=0)))
    results.append(check('invalid_semantic_null_contract', lambda s: s['region_graph']['regions'][1]['semantic'].update(valid=False, mean=0.9)))
    def leak(s):
        s['region_graph']['ground_truth'] = {'goal_position': [7, 8]}
        s['region_graph']['api_key'] = 'SYNTHETIC_NOT_A_CREDENTIAL'
    s = region_snapshot(); leak(s)
    results.append(check('unexpected_gt_and_credential_fields', leak))
    def add_option(s):
        candidate = copy.deepcopy(s['candidates'][1]); candidate['id'] = 'object-7-other-goal'; s['candidates'].append(candidate)
        option = copy.deepcopy(s['region_graph']['options'][1]); option.update(id='approach:object-7-other-goal', candidate_id=candidate['id'], region_id='R0001', route_region_ids=['R0001'])
        s['region_graph']['options'].append(option)
        s['region_graph']['coverage']['executable_candidates_exported'] = 3
        s['region_graph']['coverage']['executable_candidates_total'] = 3
    results.append(check('valid_alternative_approach_region', add_option))
    class InspectServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            self.errors.append(traceback.format_exc())
    with tempfile.TemporaryDirectory() as temp:
        server = InspectServer(('127.0.0.1', 0), make_handler(ChoiceProvider(), 0, 0, Path(temp), 'region_graph'))
        server.errors = []
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            for body in (b'[]', b'{"region_graph":null}'):
                try:
                    with urlopen(Request(f'http://127.0.0.1:{server.server_port}/decide', data=body, headers={'Content-Type': 'application/json'}), timeout=3) as response:
                        outcome = {'status': response.status}
                except Exception as exc:
                    outcome = {'exception': type(exc).__name__}
                results.append({'case': 'malformed_http_' + body.decode(), **outcome})
            results.append({'case': 'http_handler_errors', 'errors': server.errors})
        finally:
            server.shutdown(); server.server_close(); worker.join()
    class TemporaryProvider:
        def choose_request(self, payload):
            raise ProviderError('PROVIDER_TEMPORARY_FAILURE', 'synthetic', retryable=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(TemporaryProvider(), 0, 0))
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    full = json.loads((ROOT / 'examples/snapshot.json').read_text(encoding='utf-8'))
    full['request_id'] = 'same-episode:same-observation:same-map'
    codes = []
    try:
        for increment in (0, 1):
            full['timestamp_ms'] += increment
            try:
                urlopen(Request(f'http://127.0.0.1:{server.server_port}/decide', data=json.dumps(full).encode(), headers={'Content-Type': 'application/json'}), timeout=3)
            except HTTPError as exc:
                codes.append({'status': exc.code, 'error': json.load(exc)['error_code']})
    finally:
        server.shutdown(); server.server_close(); worker.join()
    results.append({'case': 'full_retry_timestamp_changed', 'responses': codes})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(['git', 'init', '-q', directory], check=True, capture_output=True)
        from region_run_fingerprint import source_fingerprint
        def fingerprint(): return source_fingerprint(root)['sha256']
        (root/'nav_jev_bridge').mkdir()
        f = root / 'nav_jev_bridge/region_graph.py'; f.write_text('first version')
        before = fingerprint(); f.write_text('different implementation'); after = fingerprint()
        results.append({'case': 'manifest_untracked_source_change', 'fingerprint_unchanged': before == after})
    report = {'scope': 'independent offline probes, synthetic inputs, zero real API calls', 'results': results}
    path = ROOT / 'docs/JEV_REGION_GRAPH_REVIEW_EVIDENCE_AFTER_FIX.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    expected_rejections={'local_path_collision_false','local_mode_target_mismatch','source_objects_total_mismatch','stale_verified_edge','invalid_semantic_null_contract','unexpected_gt_and_credential_fields'}
    assert all(r.get('rejected') is True for r in results if r['case'] in expected_rejections)
    assert next(r for r in results if r['case']=='valid_alternative_approach_region')['rejected'] is False
    assert not next(r for r in results if r['case']=='http_handler_errors')['errors']
    assert all(r['exception']=='HTTPError' for r in results if r['case'].startswith('malformed_http'))
    assert next(r for r in results if r['case']=='manifest_untracked_source_change')['fingerprint_unchanged'] is False
    assert [r['status'] for r in next(r for r in results if r['case']=='full_retry_timestamp_changed')['responses']]==[503,503]

if __name__ == '__main__':
    main()
