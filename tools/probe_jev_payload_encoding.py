"""Offline-only lossless table/reference encoding experiment; never calls Jev.

This is not connected to the production bridge. Byte savings do not establish
model token savings, API acceptance, or decision quality.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def pack_state(state, deduplicate=True):
    counts = Counter()
    def count(value):
        if isinstance(value, dict):
            if any(k in value for k in ('__jev_ref', '__jev_table')):
                raise ValueError('reserved encoding key in input')
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)
        if isinstance(value, (dict, list)) and len(text(value)) >= 180:
            counts[text(value)] += 1
    count(state)
    refs, definitions = {}, {}
    def encode(value):
        serial = text(value) if isinstance(value, (dict, list)) else None
        if deduplicate and serial is not None and counts[serial] > 1:
            if serial not in refs:
                identity = 'e' + str(len(refs))
                refs[serial] = identity
                definitions[identity] = raw(value)
            return {'__jev_ref': refs[serial]}
        return raw(value)
    def raw(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) >= 3 and all(isinstance(row, dict) for row in value):
                columns = sorted(value[0])
                if columns and all(sorted(row) == columns for row in value):
                    return {'__jev_table': {'columns': columns,
                        'rows': [[encode(row[k]) for k in columns] for row in value]}}
            return [encode(row) for row in value]
        return value
    root = encode(state)
    return {'encoding': 'offline_lossless_tables_refs_v1',
        'reading_guide': 'Tables pair columns with every row in order. __jev_ref points to definitions in this same state. All values and rows are preserved exactly; no rounding, truncation or external files.',
        'definitions': definitions, 'state': root}


def unpack_state(packed):
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'__jev_ref'}:
                return decode(packed['definitions'][value['__jev_ref']])
            if set(value) == {'__jev_table'}:
                table = value['__jev_table']
                return [{k: decode(v) for k, v in zip(table['columns'], row)}
                        for row in table['rows']]
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value
    return decode(packed['state'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.traces.glob('*.json')):
        trace = json.loads(path.read_text(encoding='utf-8'))
        request = trace.get('model_request')
        if not isinstance(request, dict) or not isinstance(request.get('state'), dict):
            continue
        state = request['state']
        row = {'trace': path.name, 'decision': trace.get('decision'),
               'original_wire_bytes': len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')),
               'minified_request_bytes': len(text(request).encode('utf-8')),
               'criteria_count': len(request['questions']['next_goal']['criteria']),
               'field_bytes': {k: len(text(v).encode('utf-8')) for k, v in state.items()}}
        for name, dedup in [('tables_only', False), ('tables_and_refs', True)]:
            packed = pack_state(state, dedup)
            restored = unpack_state(packed)
            if text(restored) != text(state):
                raise AssertionError('lossless round trip failed: ' + path.name)
            candidate_request = dict(request, state=packed)
            row[name] = {'request_bytes': len(text(candidate_request).encode('utf-8')),
                         'round_trip_exact': True,
                         'definitions': len(packed['definitions']),
                         'criteria_unchanged': candidate_request['questions'] == request['questions']}
        rows.append(row)
    result = {'scope': 'offline representation experiment, not a deployed format',
              'token_counts': 'not measured with Jev tokenizer',
              'api_acceptance': 'not tested', 'requests': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
