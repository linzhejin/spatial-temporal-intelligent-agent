"""Archive the verified inputs and acceptance evidence after the serial checks."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
CHECKS = Path(__file__).resolve().parent
RESULT = ROOT / 'research/e3/20261102'
BATCH = ROOT / 'output/experiments/e3_product_complete_20261102'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    completion = load(CHECKS / 'completed.json')
    if completion.get('completed') is not True:
        raise RuntimeError('serial acceptance has not completed')
    result = load(RESULT / 'summary.json')
    if result['campus']['acceptance_passed'] is not True:
        raise RuntimeError('campus batch has not passed')
    reporter = ROOT / 'scripts/research/summarize_e3_product.py'
    if digest(reporter.read_bytes()) != result['derivation']['script_sha256']:
        raise RuntimeError('aggregation source changed after deriving the result')
    proof_counts = {}
    for folder, expected in [('e3_product_complete_20261102', 504),
                             ('e3_counterexamples_20261102_02', 12)]:
        batch = ROOT / 'output/experiments' / folder
        proof = load(batch / 'stored_proofs.json')
        pattern = 'setup_*.json' if 'counterexamples' in folder else 'initial_*.json'
        if proof['initial_proofs_checked'] != expected or len(list(batch.glob(pattern))) != expected:
            raise RuntimeError('initial proof coverage is incomplete: ' + folder)
        proof_counts[folder] = {'initial_proofs_checked': proof['initial_proofs_checked'],
                                'stored_rows': len(proof['stored_proofs'])}
    frozen = load(BATCH / 'source_manifest.json')['source_hashes']
    extra = [
        'scripts/research/summarize_e3_product.py',
        'scripts/research/resume_e3_product.py',
        'experiments/E3_PRODUCT_COMPARISON.md',
        'tests/js/conversation-lifecycle.test.js',
        *[str(p.relative_to(ROOT)).replace('\\', '/') for p in sorted((ROOT / 'tests').rglob('*.py'))],
    ]
    paths = {name.replace('\\', '/'): {'sha256': sha, 'role': 'frozen_execution_input'}
             for name, sha in frozen.items()}
    for name in extra:
        if name not in paths:
            paths[name] = {'sha256': digest((ROOT / name).read_bytes()), 'role': 'derivation_and_acceptance'}
    archive = RESULT / 'source_inputs.zip'
    if archive.exists():
        raise FileExistsError(archive)
    members = {}
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as target:
        for name, info in sorted(paths.items()):
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or ':' in name or name.startswith('.'):
                raise ValueError('unsafe archive path')
            file = (ROOT / name).resolve()
            file.relative_to(ROOT.resolve())
            data = file.read_bytes()
            if digest(data) != info['sha256']:
                raise RuntimeError('frozen file changed: ' + name)
            if any(part.lower() in {'.env', 'secrets', 'user_profiles.json', 'telemetry.jsonl'} for part in rel.parts):
                raise RuntimeError('unexpected runtime or credential input: ' + name)
            target.writestr(name, data)
            members[name] = {**info, 'bytes': len(data)}
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:
            raise RuntimeError('source archive CRC failure')
        for name, info in members.items():
            if digest(source.read(name)) != info['sha256']:
                raise RuntimeError('source archive hash failure: ' + name)
    save(RESULT / 'source_inputs_manifest.json', {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest': 'e3_product_complete_20261102/source_manifest.json',
        'archive_sha256': digest(archive.read_bytes()), 'files': members,
    })
    evidence = RESULT / 'acceptance'
    evidence.mkdir(exist_ok=False)
    names = ['completed.json', 'acceptance_commands.json', 'regression.xml',
             'regression_counts.json', 'python_regression.log', 'javascript_lifecycle.log',
             'ruff.log', 'diff_check.log', 'aggregate.log', 'run_acceptance.py',
             'package_acceptance.py', 'run_acceptance_initial.py',
             'power_events_during_batch.json', 'resume_selection.json']
    copied = {}
    for name in names:
        origin = CHECKS / name
        destination = evidence / name
        expected = digest(origin.read_bytes())
        shutil.copy2(origin, destination)
        actual = digest(destination.read_bytes())
        if actual != expected or digest(origin.read_bytes()) != expected:
            raise RuntimeError('acceptance copy differs from its source: ' + name)
        copied[name] = {'source': str(origin.relative_to(ROOT)).replace('\\', '/'),
                        'source_sha256': expected, 'copied_sha256': actual,
                        'bytes': destination.stat().st_size}
    save(evidence / 'copied_files_manifest.json', copied)
    distributions = sorted(
        ({'name': d.metadata.get('Name', ''), 'version': d.version} for d in metadata.distributions()),
        key=lambda item: (item['name'].casefold(), item['version']))
    save(evidence / 'environment.json', {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
        'distributions': distributions,
        'note': 'Package names and versions only; no environment variable values or private package URLs.',
    })
    (RESULT / 'SOURCE_REPRODUCTION.md').write_text(
        '# 源码与输入复现范围\n\n'
        '基准提交：`c9aa0f3f90e6a89c1e0856bd6ac1e8d059de9c37`。先检出该提交，'
        '再将 `source_inputs.zip` 解压到仓库根目录。归档提供本批冻结的145项执行输入、'
        '汇总脚本及验收测试源码，应用入口、前端资源和其他依赖源码由基准提交提供。'
        '这份归档不是脱离仓库即可启动的独立应用包。\n\n'
        '逐文件SHA-256见 `source_inputs_manifest.json`；结果派生脚本的哈希同时与'
        '`summary.json` 中的生成记录核对。软件版本见原始批次 `source_manifest.json` '
        '及 `acceptance/environment.json`。数据库只使用本机专用测试库；'
        '运行命令见 `experiments/E3_PRODUCT_COMPARISON.md`，验收的实际命令见'
        '`acceptance/acceptance_commands.json`。\n', encoding='utf-8')
    save(evidence / 'packaging_checks.json', {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_file_count': len(frozen), 'source_archive_file_count': len(members),
        'source_archive_crc_and_hashes_passed': True,
        'reporter_matches_aggregation_hash': True,
        'acceptance_copies_verified': len(copied),
        'stored_proof_checks': proof_counts, 'regression': completion['regression'],
    })
    print(json.dumps({'source_files': len(members), 'proofs': proof_counts,
                      'regression': completion['regression']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
