"""Resume an interrupted E3 prefix without changing the frozen executor.

Completed parent observations take precedence. Whole tail cases are rerun with
the original shuffle sequence; duplicate observations remain diagnostic only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['case_id'], row['strategy'], row['attempt']


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def resumed_rng(seed, shuffle_steps, strategies):
    rng = random.Random(seed)
    for _ in range(shuffle_steps):
        order = list(strategies)
        rng.shuffle(order)
    return rng


def build_selection(protocol, records):
    repeats = protocol['repeats']
    attempts = repeats + protocol['operation_probe_repeats']
    planned = [{'case_id': c['id'], 'strategy': s, 'attempt': a,
                'operation_probe': a > repeats}
               for c in protocol['cases'] for a in range(1, attempts + 1)
               for s in protocol['strategies']]
    keys = {key(r) for r in records}
    if len(keys) != len(records):
        raise ValueError('duplicate parent records')
    if keys - {key(r) for r in planned}:
        raise ValueError('unplanned parent record')
    if any(r['operation_probe'] != (r['attempt'] > repeats) for r in records):
        raise ValueError('incorrect probe flag')
    gap = False
    for case in protocol['cases']:
        for attempt in range(1, attempts + 1):
            present = [((case['id'], s, attempt) in keys) for s in protocol['strategies']]
            if any(present) and not all(present):
                raise ValueError('partial pair requires a separate recovery decision')
            if all(present) and gap:
                raise ValueError('parent is not a complete pair prefix')
            gap |= not any(present)
    missing = [r for r in planned if key(r) not in keys]
    if not missing:
        raise ValueError('parent already covers the planned matrix')
    first = next(i for i, c in enumerate(protocol['cases']) if c['id'] == missing[0]['case_id'])
    tail = [c['id'] for c in protocol['cases'][first:]]
    return {
        'rule': 'parent records take precedence; fill missing keys only; retain duplicates as diagnostics',
        'resume_case_ids': tail, 'shuffle_steps': first * attempts,
        'selected_from_parent': [r for r in planned if key(r) in keys],
        'selected_from_continuation': missing,
        'diagnostic_only_continuation': [r for r in planned if r['case_id'] in tail and key(r) in keys],
    }


def prepare_and_run(parent, output, selection_path, database_url):
    from experiments import e3_product_runner as runner

    protocol = read(parent / 'protocol.json')
    parent_summary = read(parent / 'summary.json')
    if parent_summary.get('terminal_error', {}).get('type') != 'TimeoutError':
        raise ValueError('this recovery path requires a documented timeout')
    if read(parent / 'source_manifest.json')['source_hashes'] != runner.input_hashes():
        raise ValueError('frozen execution inputs changed')
    selection = build_selection(protocol, load_rows(parent / 'executions.jsonl'))
    selection.update({
        'parent_batch': str(parent), 'continuation_batch': str(output),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_records_sha256': sha(parent / 'executions.jsonl'),
        'parent_protocol_sha256': sha(parent / 'protocol.json'),
        'orchestrator_sha256': sha(Path(__file__)), 'argv': sys.argv,
        'interruption': parent_summary['terminal_error'],
    })
    if selection_path.exists():
        raise FileExistsError(selection_path)
    runner.write_json(selection_path, selection)
    real_random = runner.random
    def factory(seed):
        if seed != protocol['seed']:
            raise ValueError('unexpected random seed')
        return resumed_rng(seed, selection['shuffle_steps'], protocol['strategies'])
    try:
        # Override this module's RNG factory only, never random.Random globally.
        with patch.object(runner, 'random', SimpleNamespace(Random=factory)):
            summary = runner.run(output, database_url, repeats=protocol['repeats'],
                                 case_ids=selection['resume_case_ids'],
                                 probes=bool(protocol['operation_probe_repeats']))
    finally:
        assert runner.random is real_random
    runner.write_json(output / 'continuation_provenance.json', selection)
    return summary


def merge_batches(parent, continuation, selection_path, output):
    from experiments import e3_product_runner as runner

    selection = read(selection_path)
    if sha(parent / 'executions.jsonl') != selection['parent_records_sha256']:
        raise ValueError('parent records changed')
    if sha(parent / 'protocol.json') != selection['parent_protocol_sha256']:
        raise ValueError('parent protocol changed')
    if sha(Path(__file__)) != selection['orchestrator_sha256']:
        raise ValueError('recovery orchestration changed')
    protocol, tail_protocol = read(parent / 'protocol.json'), read(continuation / 'protocol.json')
    tail_cases = [c for c in protocol['cases'] if c['id'] in selection['resume_case_ids']]
    if tail_protocol['cases'] != tail_cases:
        raise ValueError('continuation case definitions differ')
    for field in ['strategies', 'repeats', 'operation_probe_repeats', 'seed', 'timing', 'weather', 'binding_scope']:
        if protocol[field] != tail_protocol[field]:
            raise ValueError('protocol field changed: ' + field)
    parent_source, tail_source = read(parent / 'source_manifest.json'), read(continuation / 'source_manifest.json')
    for field in ['source_hashes', 'python', 'packages', 'platform', 'git_head']:
        if parent_source[field] != tail_source[field]:
            raise ValueError('execution environment changed: ' + field)
    if parent_source['source_hashes'] != runner.input_hashes():
        raise ValueError('current frozen inputs changed')
    if not read(continuation / 'summary.json')['acceptance_passed']:
        raise ValueError('continuation did not pass')
    identities = [read(p / 'isolation.json') for p in [parent, continuation]]
    if identities[0]['server_version'] != identities[1]['server_version']:
        raise ValueError('database version changed')
    selected = []
    selected_lines = []
    lineage = []
    pair_source = {}
    for source, field in [(parent, 'selected_from_parent'), (continuation, 'selected_from_continuation')]:
        wanted = {key(r) for r in selection[field]}
        found = set()
        for line_number, raw in enumerate((source / 'executions.jsonl').read_bytes().splitlines(keepends=True), 1):
            row = json.loads(raw)
            if key(row) not in wanted:
                continue
            if key(row) in found:
                raise ValueError('duplicate selected key')
            found.add(key(row))
            selected.append(row)
            selected_lines.append(raw)
            pair_source[(row['case_id'], row['attempt'])] = source
            lineage.append({'case_id': row['case_id'], 'strategy': row['strategy'],
                            'attempt': row['attempt'], 'operation_probe': row['operation_probe'],
                            'batch': source.name, 'line': line_number,
                            'line_sha256': hashlib.sha256(raw).hexdigest()})
        if found != wanted:
            raise ValueError('selected keys are missing')
    summary = runner.summarize_records(protocol, selected, input_hashes_unchanged=True)
    if not summary['acceptance_passed']:
        raise ValueError('merged semantic, policy or coverage audit failed')
    output.mkdir(parents=True, exist_ok=False)
    for name in ['protocol.json', 'source_manifest.json']:
        shutil.copy2(parent / name, output / name)
    (output / 'executions.jsonl').write_bytes(b''.join(selected_lines))
    for (case_id, attempt), source in pair_source.items():
        for stem in ['initial', 'recovery_before']:
            file = source / f'{stem}_{case_id}_{attempt}.json'
            if file.exists():
                shutil.copy2(file, output / file.name)
        expected = source / f'expected_{case_id}.json'
        if (output / expected.name).exists() and sha(output / expected.name) != sha(expected):
            raise ValueError('cross-segment expected task differs')
        shutil.copy2(expected, output / expected.name)
    runner.write_json(output / 'isolation.json', {
        'owner_ids': [i['owner_id'] for i in identities], 'segments': identities,
        'database': identities[0]['database'], 'server_version': identities[0]['server_version'],
    })
    summary['execution_history'] = {
        'completion_kind': 'completed_after_documented_interruption',
        'parent_batch': parent.name, 'continuation_batch': continuation.name,
        'parent_terminal_error': read(parent / 'summary.json')['terminal_error'],
        'preserved_parent_records': len(selection['selected_from_parent']),
        'added_missing_records': len(selection['selected_from_continuation']),
        'additional_diagnostic_records': len(selection['diagnostic_only_continuation']),
        'interrupted_operation_probe_attempts': 1,
        'note': 'New process after system standby; strategy pairs share a segment and original shuffle order.',
    }
    runner.write_json(output / 'summary.json', summary)
    runner.write_json(output / 'record_lineage.json', lineage)
    runner.write_json(output / 'resume_selection.json', selection)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--continuation', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--merged', type=Path)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if not args.merge_only:
        database_url = os.environ.get('TEST_DATABASE_URL')
        if not database_url:
            parser.error('explicit TEST_DATABASE_URL required')
        result = prepare_and_run(args.parent, args.continuation, args.selection, database_url)
        if not result['acceptance_passed']:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
    if args.merged:
        result = merge_batches(args.parent, args.continuation, args.selection, args.merged)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
