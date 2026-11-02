"""Wait for the frozen E3 batch, then verify it serially without timing interference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SUMMARY = ROOT / 'output/experiments/e3_product_full_20261102_02/summary.json'
ENV = os.environ.copy()
ENV.update(DATABASE_URL='', TEST_DATABASE_URL='postgresql://postgres@127.0.0.1:55439/whu_e3_20261102_test',
           PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
results = []


def save():
    (OUT / 'acceptance_commands.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


def command(label, args):
    started = time.time()
    print('START ' + label, flush=True)
    with (OUT / (label + '.log')).open('w', encoding='utf-8') as log:
        code = subprocess.run(args, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT).returncode
    results.append({'label': label, 'command': args, 'exit_code': code, 'elapsed_seconds': time.time()-started})
    save()
    print('END ' + label + ' exit=' + str(code), flush=True)
    if code:
        raise SystemExit(code)


while not SUMMARY.exists():
    time.sleep(30)
summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
if summary.get('acceptance_passed') is not True:
    print('Frozen batch did not pass; no regression or aggregation started.', flush=True)
    raise SystemExit(2)

# Archive only synthetic experiment proofs, before regression fixtures clear the
# test database. Conversation credentials are deliberately not queried.
sys.path.insert(0, str(ROOT))
from storage import database
for folder, pattern in (('e3_product_full_20261102_02', 'initial_*.json'),
                        ('e3_counterexamples_20261102_02', 'setup_*.json')):
    batch = ROOT / 'output/experiments' / folder
    owner = json.loads((batch / 'isolation.json').read_text(encoding='utf-8'))['owner_id']
    with database.connect(ENV['TEST_DATABASE_URL']) as conn:
        proofs = conn.execute('SELECT * FROM route_proof WHERE owner_id=%s ORDER BY created_at, proof_id',
                              (owner,)).fetchall()
    lookup = {str(row['proof_id']): row for row in proofs}
    checked = 0
    for path in batch.glob(pattern):
        item = json.loads(path.read_text(encoding='utf-8'))
        data = item['response']['data']
        state = data['route_state']
        proof_id = state['reuse_certificate']['proof_id']
        row = lookup.get(proof_id)
        if row is None or row['verified'] is not True or row['selected_edges'] != data['recommended_edge_ids']:
            raise RuntimeError('stored initial proof differs from archived response: ' + path.name)
        if row['graph_version'] != state['data_version'] or row['event_version'] != state['road_condition_version']:
            raise RuntimeError('stored initial proof versions differ: ' + path.name)
        checked += 1
    snapshot = {'captured_utc': datetime.now(timezone.utc).isoformat(),
                'initial_proofs_checked': checked, 'stored_proofs': [dict(row) for row in proofs],
                'note': 'Expired rows remain provenance; snapshot is not a current authorization claim.'}
    (batch / 'stored_proofs.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('STORED PROOFS ' + folder + ': initial=' + str(checked) + ', rows=' + str(len(proofs)), flush=True)

tests = [
    'tests/test_e3_product_cases.py', 'tests/test_e3_product_runner.py',
    'tests/test_e3_independent_audit.py', 'tests/test_e3_measurement.py',
    'tests/test_e3_counterexamples.py', 'tests/test_e3_result_summary.py',
    'tests/test_route_reuse.py', 'tests/test_execution_dependencies.py',
    'tests/test_route_proof_integration.py', 'tests/test_route_proof_repository.py',
    'tests/test_route_proof_repository_unit.py', 'tests/test_route_proof_worker.py',
    'tests/test_task_repository.py', 'tests/test_run_worker.py', 'tests/test_task_patch.py',
    'tests/browser/test_route_candidate_resume_e2e.py',
]
command('python_regression', [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--junitxml=' + str(OUT / 'regression.xml'), *tests])
tree = ET.parse(OUT / 'regression.xml')
suites = list(tree.getroot().iter('testsuite'))
counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
(OUT / 'regression_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
if any(counts[k] for k in ('failures', 'errors', 'skipped')):
    print('Required regression incomplete: ' + json.dumps(counts), flush=True)
    raise SystemExit(3)
command('javascript_lifecycle', ['node', '--test', 'tests/js/conversation-lifecycle.test.js'])
source = [
    'experiments/e3_product_cases.py', 'experiments/e3_product_runner.py',
    'experiments/e3_independent_audit.py', 'experiments/e3_measurement.py',
    'experiments/e3_counterexamples.py', 'scripts/research/summarize_e3_product.py',
    *tests[:6],
]
command('ruff', [sys.executable, '-m', 'ruff', 'check', *source])
command('diff_check', ['git', 'diff', '--check'])
command('aggregate', [sys.executable, 'scripts/research/summarize_e3_product.py',
    '--campus', 'output/experiments/e3_product_full_20261102_02',
    '--counterexamples', 'output/experiments/e3_counterexamples_20261102_02',
    '--include-diagnostic', 'output/experiments/e3_product_full_20261102_01',
    '--include-diagnostic', 'output/experiments/e3_counterexamples_20261102_01',
    '--include-diagnostic', 'output/experiments/e3_oracle_contract_20261102',
    '--include-diagnostic', 'output/experiments/e3_product_smoke_20261102_01',
    '--include-diagnostic', 'output/experiments/e3_product_smoke_20261102_02',
    '--include-diagnostic', 'output/experiments/e3_product_all_variants_20261102_01',
    '--output', 'research/e3/20261102'])
(OUT / 'completed.json').write_text(json.dumps({'completed': True, 'regression': counts}, indent=2), encoding='utf-8')
print('All queued acceptance steps completed; documentation and Git publication remain.', flush=True)
