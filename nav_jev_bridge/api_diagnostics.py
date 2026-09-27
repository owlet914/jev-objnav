"""Bounded, credential-redacted upstream error evidence (not a token estimator)."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MAX_ERROR_BODY_BYTES = 65536
MAX_ERROR_TEXT_CHARS = 4096


def encode_request(payload: dict[str, Any]) -> bytes:
    # Production serializer v2: compact UTF-8; request_facts uses these same bytes.
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def request_facts(payload: dict[str, Any]) -> dict[str, Any]:
    wire = encode_request(payload)
    return {
        'request_bytes': len(wire),
        'request_sha256': hashlib.sha256(wire).hexdigest(),
        'criteria_count': len(payload.get('questions', {}).get('next_goal', {}).get('criteria', {})),
    }


def redact_text(value: Any, secret: str = '', limit: int = MAX_ERROR_TEXT_CHARS) -> str:
    text = str(value)
    if secret:
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)Bearer\s+[^\s\"\'<>;,}]+', 'Bearer [REDACTED]', text)
    text = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_ -]?key|authorization)[\"\']?\s*[:=]\s*[\"\']?)[^\s\"\'<>;,}]+', r'\1[REDACTED]', text)
    return text[:limit]


def _error_fields(value: Any, secret: str, depth: int = 0) -> dict[str, Any]:
    """Keep error details, never arbitrary headers/request/response metadata."""
    if depth > 4:
        return {'nested_error_omitted': True}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {'message': redact_text(value, secret)}
        if isinstance(parsed, (dict, list)):
            return _error_fields(parsed, secret, depth + 1)
        return {'message': redact_text(value, secret)}
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ('code', 'message', 'type', 'param', 'request_id'):
        field = value.get(key)
        if isinstance(field, (str, int, float, bool)):
            result[key] = redact_text(field, secret) if isinstance(field, str) else field
    if isinstance(value.get('detail'), (str, dict)):
        result['detail'] = _error_fields(value['detail'], secret, depth + 1)
    if 'error' in value:
        result['error'] = _error_fields(value['error'], secret, depth + 1)
    metadata = value.get('metadata')
    if isinstance(metadata, dict):
        if isinstance(metadata.get('provider_name'), str):
            result['provider_name'] = redact_text(metadata['provider_name'], secret, 160)
        if isinstance(metadata.get('raw'), (str, dict)):
            result['provider_error'] = _error_fields(metadata['raw'], secret, depth + 1)
    return result


def classify_error(status: int, details: dict[str, Any]) -> str:
    text = json.dumps(details, ensure_ascii=False).lower()
    context_patterns = (
        'context_length_exceeded', 'maximum context length', 'context window exceeded',
        'exceeds the context', 'exceeded the context', 'too many tokens',
        'input token limit exceeded', 'max_tokens_exceeded',
    )
    if status in (400, 413, 422) and any(p in text for p in context_patterns):
        return 'context_length_reported_by_upstream'
    if status == 413:
        return 'request_body_too_large_not_necessarily_token_limit'
    if status in (401, 403):
        return 'authentication_or_access'
    if status == 402:
        return 'billing_or_credit'
    if status == 429:
        return 'rate_limit'
    if status >= 500:
        return 'upstream_temporary_failure'
    if status in (400, 422):
        return 'bad_request_cause_unconfirmed'
    return 'upstream_http_error'


def http_error_evidence(exc: Any, payload: dict[str, Any], secret: str) -> dict[str, Any]:
    evidence = {'http_status': exc.code, **request_facts(payload)}
    try:
        raw = exc.read(MAX_ERROR_BODY_BYTES + 1)
        evidence['body_truncated'] = len(raw) > MAX_ERROR_BODY_BYTES
        raw = raw[:MAX_ERROR_BODY_BYTES]
        try:
            parsed = json.loads(raw)
            evidence['body_format'] = 'json'
            details = _error_fields(parsed, secret)
        except (ValueError, UnicodeError):
            evidence['body_format'] = 'text'
            details = {'message': redact_text(raw.decode('utf-8', errors='replace'), secret)}
    except Exception as read_error:
        evidence['body_read_error'] = type(read_error).__name__
        details = {}
    evidence['error_details'] = details
    for header in ('x-request-id', 'x-openrouter-request-id', 'cf-ray'):
        value = exc.headers.get(header) if exc.headers else None
        if value:
            evidence.setdefault('request_ids', {})[header] = redact_text(value, secret, 200)
    evidence['classification'] = classify_error(exc.code, details)
    return evidence
