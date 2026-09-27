"""Bounded real Jev API diagnostic calls. Does not run or control navigation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav_jev_bridge.api_diagnostics import request_facts, redact_text
from nav_jev_bridge.bridge import OpenRouterJevProvider, ProviderError

TRACE_ROOT = ROOT / '.runtime/jev_real_api_20260925'
FAILED = TRACE_ROOT / 'f19e44baa2a44909ab92500119918dc5.json'
SUCCEEDED = TRACE_ROOT / 'fd0a18c1dc374f978a649684932702f3.json'


def build_cases(failed, succeeded):
    minimal = {
        'model': OpenRouterJevProvider.MODEL,
        'state': {'task': 'diagnostic_only', 'signal': 'green'},
        'questions': {'next_goal': {
            'type': 'choice',
            'instructions': 'This is an API diagnostic, not robot control. Choose proceed for green, otherwise choose wait.',
            'criteria': {'proceed': 'The signal is green.', 'wait': 'The signal is not green.'},
        }},
    }
    sixteen = copy.deepcopy(failed)
    sixteen['state'] = {
        'diagnostic_only': True,
        'target_object': failed['state']['target_object'],
        'candidate_ids': list(failed['questions']['next_goal']['criteria']),
        'note': 'Diagnostic state intentionally omits navigation evidence. No output will be executed or counted in navigation metrics.',
    }
    padded = copy.deepcopy(minimal)
    # This is a controlled repetition count, NOT an asserted Jev token count.
    padded['state']['diagnostic_padding_ignore_for_choice'] = ' x' * 40000
    return [
        ('original_failed_exact', copy.deepcopy(failed)),
        ('historical_success_exact', copy.deepcopy(succeeded)),
        ('synthetic_minimal_two_choices', minimal),
        ('minimal_state_original_criteria', sixteen),
        ('synthetic_same_two_choices_large_text', padded),
    ]


def safe_response_summary(body, payload):
    answer = body.get('answers', {}).get('next_goal', {})
    if not isinstance(answer, dict):
        return {'valid_choice_response': False}
    choice = answer.get('choice')
    criteria = payload['questions']['next_goal']['criteria']
    confidence = answer.get('confidence')
    probabilities = answer.get('probabilities', {})
    probability = probabilities.get(choice) if isinstance(probabilities, dict) and isinstance(choice, str) else None
    def valid_number(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
    valid = isinstance(choice, str) and choice in criteria and valid_number(confidence) and valid_number(probability)
    result = {'valid_choice_response': valid}
    if valid:
        result.update(choice=choice, confidence=confidence, probability=probability)
    usage = body.get('usage')
    if isinstance(usage, dict):
        result['usage'] = {k: v for k, v in usage.items()
                           if k in ('input_tokens', 'output_tokens', 'total_tokens', 'cost', 'inputTokens', 'outputTokens')
                           and isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)}
    for name in ('id', 'model', 'request_id'):
        if isinstance(body.get(name), str):
            result[name] = redact_text(body[name], os.getenv('OPENROUTER_API_KEY', ''), 200)
    return result


def conclusion(results):
    original = next((r for r in results if r['case'] == 'original_failed_exact'), None)
    if not original:
        return 'Original failure has not been replayed.'
    if original.get('outcome') == 'accepted':
        return 'Original request is accepted now; historical HTTP 400 cause is not established.'
    diagnostic = original.get('upstream', {})
    category = diagnostic.get('classification')
    if category == 'context_length_reported_by_upstream':
        return 'Original request rejected with explicit upstream context-length evidence; inspect exact error_details for limits/counts.'
    if category in ('authentication_or_access', 'billing_or_credit', 'rate_limit'):
        return 'Current blocker is ' + category + '; historical HTTP 400 cause is not established.'
    return 'Original failure requires inspection of exact upstream error_details. Do not infer a token limit from byte counts or from a different synthetic request.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-key', action='store_true', help='Read a fresh key without echo; never store it.')
    parser.add_argument('--dry-run', action='store_true', help='List exact payload sizes/hashes without a key or network calls.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    failed = json.loads(FAILED.read_text(encoding='utf-8'))['model_request']
    succeeded = json.loads(SUCCEEDED.read_text(encoding='utf-8'))['model_request']
    cases = build_cases(failed, succeeded)
    if any(p.get('model') != OpenRouterJevProvider.MODEL for _, p in cases):
        raise SystemExit('Refusing a different model; diagnostic suite is pinned to typesafe/jev-1.13.')
    if args.dry_run:
        print(json.dumps({'network_calls': 0, 'cases': [{'case': n, **request_facts(p)} for n, p in cases]}, indent=2))
        return
    if args.prompt_key:
        if not sys.stdin.isatty():
            raise SystemExit('Hidden key entry needs an interactive terminal. No API call made.')
        secret = getpass.getpass('OpenRouter API key (hidden): ').strip()
        if not secret:
            raise SystemExit('No key entered. No API call made.')
        os.environ['OPENROUTER_API_KEY'] = secret
        del secret
    if not os.getenv('OPENROUTER_API_KEY'):
        raise SystemExit('OPENROUTER_API_KEY is missing. Run with --prompt-key in your own terminal. No demo fallback.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / '.runtime/jev_api_diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    output = args.output or directory / ('diagnosis-' + stamp + '.json')
    if output.exists():
        raise SystemExit('Refusing to overwrite existing evidence: ' + str(output))
    report = {'started_at_utc': stamp, 'provider': 'OpenRouterJevProvider',
              'model': OpenRouterJevProvider.MODEL, 'endpoint': OpenRouterJevProvider.ENDPOINT,
              'scope': 'API diagnosis only; no ROS/Habitat actions; max 5 calls, no retries',
              'results': []}
    def save():
        report['conclusion'] = conclusion(report['results'])
        content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(output)
    provider = OpenRouterJevProvider(timeout_s=30)
    print('Real Jev diagnosis: max 5 calls, no retries, no navigation.', flush=True)
    for name, payload in cases:
        request_file = output.with_name(output.stem + '-' + name + '-request.json')
        request_file.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        row = {'case': name, 'request_artifact': request_file.name, **request_facts(payload)}
        started = time.perf_counter()
        print('Calling ' + name + ' (' + str(row['request_bytes']) + ' bytes)...', flush=True)
        try:
            body = provider.request_json(payload)
            row['response'] = safe_response_summary(body, payload)
            row['outcome'] = 'accepted' if row['response']['valid_choice_response'] else 'invalid_response'
        except ProviderError as exc:
            row.update(outcome='error', error_code=exc.code, retryable=exc.retryable,
                       detail=str(exc), upstream=exc.diagnostics)
        row['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
        report['results'].append(row)
        save()
        print(json.dumps(row, ensure_ascii=True, allow_nan=False), flush=True)
        status = row.get('upstream', {}).get('http_status')
        if status in (401, 402, 403, 429) or (status is not None and status >= 500) or row.get('error_code') == 'PROVIDER_TEMPORARY_FAILURE':
            report['stopped_early'] = 'Stop on auth/billing/rate-limit/temporary failure; no repeated calls.'
            save()
            break
    print('REPORT=' + str(output), flush=True)
    print(report['conclusion'], flush=True)

if __name__ == '__main__':
    main()
