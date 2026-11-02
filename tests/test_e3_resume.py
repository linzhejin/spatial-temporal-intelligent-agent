"""Recovery keeps all prior measurements and the frozen strategy order."""
import random

import pytest

from scripts.research.resume_e3_product import build_selection, resumed_rng


def protocol():
    return {'cases': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
            'strategies': ['full_recompute', 'conservative_reuse'],
            'repeats': 3, 'operation_probe_repeats': 1, 'seed': 20261102}


def rows_for(p):
    return [{'case_id': c['id'], 'strategy': s, 'attempt': a,
             'operation_probe': a > p['repeats']}
            for c in p['cases'] for a in range(1, 5) for s in p['strategies']]


def test_selection_preserves_completed_keys_and_only_fills_missing():
    p = protocol()
    existing = rows_for(p)[:14]  # All a; b has three complete timed pairs.
    selection = build_selection(p, existing)
    assert selection['resume_case_ids'] == ['b', 'c']
    assert selection['shuffle_steps'] == 4
    assert len(selection['selected_from_parent']) == 14
    assert len(selection['selected_from_continuation']) == 10
    assert len(selection['diagnostic_only_continuation']) == 6
    assert {r['case_id'] for r in selection['diagnostic_only_continuation']} == {'b'}


def test_rejects_half_pair_and_duplicate_keys():
    p = protocol()
    with pytest.raises(ValueError, match='partial pair'):
        build_selection(p, rows_for(p)[:13])
    with pytest.raises(ValueError, match='duplicate'):
        build_selection(p, rows_for(p)[:14] + [rows_for(p)[0]])


def test_rejects_non_prefix_and_wrong_probe_flag():
    p = protocol()
    with pytest.raises(ValueError, match='prefix'):
        build_selection(p, rows_for(p)[8:])
    rows = rows_for(p)[:14]
    rows[0]['operation_probe'] = True
    with pytest.raises(ValueError, match='probe'):
        build_selection(p, rows)


def test_rng_continues_original_shuffle_sequence():
    p = protocol()
    original = random.Random(p['seed'])
    all_orders = []
    for _ in range(16):
        values = list(p['strategies'])
        original.shuffle(values)
        all_orders.append(values)
    resumed = resumed_rng(p['seed'], 4, p['strategies'])
    actual = []
    for _ in range(12):
        values = list(p['strategies'])
        resumed.shuffle(values)
        actual.append(values)
    assert actual == all_orders[4:]
