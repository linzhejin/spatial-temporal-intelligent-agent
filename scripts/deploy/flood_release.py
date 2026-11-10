"""Deploy a reviewed web-copy release after quiescing and verifying a fresh backup.

The release makes no database migration or APK change. Credentials and provider
responses are never written to stdout or the deployment report.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import psycopg
from psycopg.types.json import Jsonb
from dotenv import dotenv_values, set_key

APP = Path('/home/ubuntu/campus-spatial-intelligence-agent')
OPS = Path('/home/ubuntu/luojia-release-ops-20261105')
MODEL_ROOT = Path('/home/ubuntu/luojia-models')
BACKUP_ROOT = Path('/home/ubuntu/luojia-release-backups')
BACKUP_SCRIPT = OPS / 'flood_backup.py'
SERVICES = ('whu-walker', 'whu-agent-worker', 'whu-vision-worker', 'whu-vision-trial-worker')
ASSET_VERSION = '20261109-mountain-v1'
MANAGER_ASSET_VERSION = '20261110-flood-assistance-v1'
CACHE_NAME = 'whu-walker-v127'
GITEE_MAIN_URL = 'https://gitee.com/kim-in-cheol/campus-spatio-temporal-agent.git'
PRESERVED_UNTRACKED = {'?? cloudflared', '?? scripts/_probe_shortest.py', '?? wget-log'}
MODEL_SHA256 = '1ef47ccd3aef9cafee359485cfb38c65edcd6625e1552d02e23afacd98f7eb52'
REQUIRED_TESTED_SOURCES = {
    'config.py', 'storage/vision_repository.py', 'vision/analysis.py', 'vision/engine.py',
    'vision/model_release.py', 'vision/specialized.py', 'vision/worker.py',
    'vision/doodleverse_preprocessing.py', 'vision/assisted_image.py',
    'vision/assisted_image_child.py', 'vision/assisted_image_supervisor.py',
    'static/js/manager.js', 'static/manager.html', 'static/sw.js', 'api/routes.py',
    'requirements-flood-runtime.txt', 'scripts/deploy/flood_backup.py',
    'scripts/deploy/flood_release.py', 'scripts/deploy/prepare_flood_bundle.py',
    'scripts/deploy/public_flood_smoke.py',
}


def run(args, *, env=None):
    result = subprocess.run(args, cwd=APP, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f'{Path(args[0]).name} failed with code {result.returncode}')
    return result.stdout.strip()


def sha256_file(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def active_work(database_url):
    with psycopg.connect(database_url) as conn:
        agent = conn.execute("SELECT count(*) FROM run WHERE status IN ('queued','running')").fetchone()[0]
        vision = conn.execute("SELECT count(*) FROM manager_vision_job "
                              "WHERE status IN ('queued','running') AND deleted_at IS NULL").fetchone()[0]
    return agent + vision


def health_ready():
    for _ in range(40):
        try:
            with urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3) as response:
                if json.load(response).get('status') == 'ok':
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def app_package_fingerprint():
    with urllib.request.urlopen('http://127.0.0.1:5000/app/latest.json', timeout=5) as response:
        manifest = json.load(response)
    apk = APP / 'static/app/whu-walker.apk'
    return {'version_name': manifest.get('versionName'), 'version_code': manifest.get('versionCode'),
            'manifest_sha256': manifest.get('sha256'),
            'file_sha256': hashlib.sha256(apk.read_bytes()).hexdigest()}


def web_assets_ready():
    with urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5) as response:
        html = response.read().decode('utf-8')
    with urllib.request.urlopen('http://127.0.0.1:5000/sw.js', timeout=5) as response:
        worker = response.read().decode('utf-8')
    with urllib.request.urlopen('http://127.0.0.1:5000/manager', timeout=5) as response:
        manager = response.read().decode('utf-8')
    return (ASSET_VERSION in html and "serviceWorker.register('/sw.js')" in html
            and MANAGER_ASSET_VERSION in manager and CACHE_NAME in worker)


def validate_bundle(bundle_path, new_commit):
    path = Path(bundle_path).resolve(strict=True)
    root = path.parent
    if root.parent != MODEL_ROOT.resolve() or not root.name.startswith('floodnet-v7-'):
        raise ValueError('Release bundle is outside the dedicated model directory')
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('target_commit') != new_commit:
        raise ValueError('Release bundle does not identify the reviewed commit')
    model = Path(bundle['model_path'])
    runtime = Path(bundle['runtime_python'])
    manifest = Path(bundle['release_manifest'])
    smoke_image = Path(bundle['public_smoke_image'])
    expected_hash = MODEL_SHA256
    if (not model.resolve().is_relative_to(root) or not manifest.resolve().is_relative_to(root)
            or not smoke_image.resolve().is_relative_to(root)
            or not runtime.parent.resolve().is_relative_to(root)):
        raise ValueError('Release bundle references a file outside its directory')
    if (bundle.get('model_sha256') != expected_hash or not model.is_file()
            or sha256_file(model) != expected_hash):
        raise ValueError('Flood model hash differs from the reviewed model')
    if not runtime.is_file() or not manifest.is_file():
        raise ValueError('Flood runtime or release manifest is missing')
    if not smoke_image.is_file() or sha256_file(smoke_image) != bundle['public_smoke_sha256']:
        raise ValueError('Public smoke image is missing or changed')
    for name in ('parity', 'resource', 'workflow'):
        evidence = bundle['evidence'][name]
        evidence_path = Path(evidence['path'])
        if (not evidence_path.resolve().is_relative_to(root) or not evidence_path.is_file()
                or sha256_file(evidence_path) != evidence['sha256']
                or json.loads(evidence_path.read_text(encoding='utf-8')).get('status') != 'passed'):
            raise ValueError(f'{name} evidence is missing, changed, or not passed')
    entry = json.loads(manifest.read_text(encoding='utf-8')).get('models', {}).get('flood', {})
    if (entry.get('mode') != 'assisted_review' or entry.get('field_validated') is not False
            or entry.get('model_sha256') != expected_hash
            or entry.get('model_revision') != 'doodleverse-floodnet-resunet-v7'):
        raise ValueError('Flood release manifest does not match the reviewed model')
    for name in ('parity', 'resource', 'workflow'):
        item = entry.get('acceptance', {}).get(name, {})
        if (item.get('status') != 'passed' or item.get('path') != bundle['evidence'][name]['path']
                or item.get('sha256') != bundle['evidence'][name]['sha256']):
            raise ValueError('Flood release manifest is not bound to the tested evidence')
    if not REQUIRED_TESTED_SOURCES.issubset(bundle.get('tested_source_sha256', {})):
        raise ValueError('Reviewed bundle does not identify all relevant tested source files')
    for name, expected in bundle['tested_source_sha256'].items():
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', name) or '..' in Path(name).parts:
            raise ValueError('Invalid tested source path')
        content = subprocess.check_output(['git', 'show', f'{new_commit}:{name}'], cwd=APP)
        if hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError('Reviewed commit differs from the tested source snapshot')
    return bundle


def apply_flood_configuration(bundle):
    settings = {
        'VISION_ACCIDENT_ENABLED': 'false',
        'VISION_FLOOD_ENABLED': 'true',
        'VISION_FLOOD_RELEASE_MODE': 'assisted_review',
        'VISION_FLOOD_MODEL_PATH': bundle['model_path'],
        'VISION_FLOOD_REVISION': 'doodleverse-floodnet-resunet-v7',
        'VISION_FLOOD_PROBABILITY_THRESHOLD': '0.001',
        'VISION_FLOOD_MIN_AREA_RATIO': '0.08',
        'VISION_FLOOD_RUNTIME_PYTHON': bundle['runtime_python'],
        'VISION_FLOOD_TIMEOUT_SECONDS': '120',
        'VISION_FLOOD_CPU_THREADS': '2',
        'VISION_CPU_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'VISION_OPTIONAL_MODELS_RELEASE_MANIFEST': bundle['release_manifest'],
    }
    for name, value in settings.items():
        if not set_key(str(APP / '.env'), name, str(value))[0]:
            raise RuntimeError('Could not update a flood runtime setting')
    actual = dotenv_values(APP / '.env')
    if any(actual.get(name) != str(value) for name, value in settings.items()):
        raise RuntimeError('Effective flood configuration differs from the reviewed settings')


def flood_worker_ready(database_url, expected_hash):
    for _ in range(60):
        try:
            with psycopg.connect(database_url, connect_timeout=5) as conn:
                rows = conn.execute(
                    "SELECT worker_role,status_detail FROM manager_vision_worker WHERE ready=true "
                    "AND heartbeat_at>now()-interval '30 seconds'"
                ).fetchall()
            operational = any(role in ('operational', 'shared') for role, _ in rows)
            for role, detail in rows:
                status = json.loads(detail or '{}')
                if (operational and role in ('trial', 'shared')
                        and status.get('flood_assistance_supported') is True
                        and status.get('flood_assistance_release_status') == 'approved'
                        and status.get('flood_assistance_model_sha256') == expected_hash):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def cancel_active_agent_runs(database_url):
    """End remaining runs before reverting, using the current code's audit path."""
    with psycopg.connect(database_url) as conn:
        rows = conn.execute("SELECT r.run_id,c.owner_id FROM run r JOIN conversation c USING(conversation_id)"
                            " WHERE r.status IN ('queued','running')").fetchall()
    if rows:
        sys.path.insert(0, str(APP))
        from storage.task_repository import cancel_run
        for run_id, owner_id in rows:
            status = (cancel_run(database_url, str(owner_id), str(run_id)) or {}).get('status')
            if status not in {'cancelled', 'completed', 'partial', 'failed', 'needs_input', 'superseded'}:
                raise RuntimeError('Could not end an agent run safely')
    return len(rows)


def cancel_vision_jobs(database_url):
    sys.path.insert(0, str(APP))
    from storage.vision_repository import request_cancel_job
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        rows = conn.execute(
            "SELECT job_id FROM manager_vision_job WHERE status IN ('queued','running') AND deleted_at IS NULL"
        ).fetchall()
    for (job_id,) in rows:
        request_cancel_job(database_url, str(job_id))
    return len(rows)


def finalize_stopped_vision_jobs(database_url):
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        rows = conn.execute(
            "UPDATE manager_vision_job SET status='cancelled', cancel_requested=true, lease_until=NULL, "
            "error=%s, updated_at=now() WHERE status='running' AND deleted_at IS NULL RETURNING job_id",
            (Jsonb({'code': 'release_rollback_cancelled', 'message': '版本回退前已结束影像任务，可重新提交。'}),),
        ).fetchall()
    return len(rows)


def rollback_to_backup(expected, previous, backup_directory):
    """Recover an already switched release after stopping ingress and ending work."""
    directory = Path(backup_directory).resolve(strict=True)
    if directory.parent != BACKUP_ROOT.resolve():
        raise ValueError('Rollback backup is outside the private backup directory')
    record = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    saved_env = directory / 'application.env'
    if (record.get('status') != 'backup_restore_verified' or record.get('previous_commit') != previous
            or not saved_env.is_file() or sha256_file(saved_env) != record.get('application_env_sha256')):
        raise ValueError('Rollback backup has not been verified or has changed')
    if run(['git', 'rev-parse', 'HEAD']) != expected:
        raise RuntimeError('Rollback refused because production HEAD changed')
    database_url = dotenv_values(APP / '.env').get('DATABASE_URL')
    if not database_url:
        raise RuntimeError('Production database configuration is unavailable')
    run(['sudo', '-n', 'systemctl', 'stop', 'whu-walker'])
    try:
        vision_cancelled = cancel_vision_jobs(database_url)
        for _ in range(15):
            if not active_work(database_url):
                break
            time.sleep(1)
        run(['sudo', '-n', 'systemctl', 'stop', 'whu-agent-worker', 'whu-vision-worker', 'whu-vision-trial-worker'])
        finalized = finalize_stopped_vision_jobs(database_url)
        cancelled = cancel_active_agent_runs(database_url)
        shutil.copy2(saved_env, APP / '.env')
        run(['git', 'switch', '--detach', previous])
    finally:
        run(['sudo', '-n', 'systemctl', 'start', *SERVICES])
    if not health_ready() or run(['systemctl', 'is-active', *SERVICES]).splitlines() != ['active'] * len(SERVICES):
        raise RuntimeError('Services did not recover after rollback')
    return {'status': 'rolled_back', 'head': run(['git', 'rev-parse', 'HEAD']),
            'ended_agent_runs': cancelled, 'cancelled_vision_jobs': vision_cancelled,
            'finalized_vision_jobs': finalized, 'health_after_recovery': True}


def main(old, new, rollback, bundle_path):
    os.umask(0o077)
    def interrupted(signum, _frame):
        raise RuntimeError('Deployment interrupted by signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    stopped = switched = False
    database_url = None
    backup = None
    for commit in (old, new, rollback):
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Invalid release commit identifier')
    if rollback != old:
        raise ValueError('Rollback must use the exact pre-release commit')
    if run(['git', 'rev-parse', 'HEAD']) != old:
        raise RuntimeError('Production HEAD differs from the reviewed baseline')
    run(['git', 'diff', '--quiet'])
    run(['git', 'diff', '--cached', '--quiet'])
    status = run(['git', 'status', '--porcelain', '--untracked-files=normal']).splitlines()
    if set(status) - PRESERVED_UNTRACKED:
        raise RuntimeError('Production worktree has changes outside the preserved release files')
    run(['git', 'fetch', 'origin', 'main'])
    if run(['git', 'rev-parse', 'origin/main']) != new:
        raise RuntimeError('GitHub main is not the reviewed release')
    gitee_ref = run(['git', 'ls-remote', GITEE_MAIN_URL, 'refs/heads/main']).split()
    if not gitee_ref or gitee_ref[0] != new:
        raise RuntimeError('Gitee main is not the reviewed release')
    bundle = validate_bundle(bundle_path, new)
    run(['git', 'cat-file', '-e', rollback + '^{commit}'])
    run(['git', 'merge-base', '--is-ancestor', old, new])
    run(['git', 'merge-base', '--is-ancestor', old, rollback])
    if not BACKUP_SCRIPT.is_file():
        raise RuntimeError('The reviewed backup-and-restore-check script is unavailable')
    database_url = dotenv_values(APP / '.env').get('DATABASE_URL')
    if not database_url:
        raise RuntimeError('Production database configuration is unavailable')
    package_before = app_package_fingerprint()
    if (package_before['version_name'] != '1.4.2' or package_before['version_code'] != 7
            or package_before['manifest_sha256'] != package_before['file_sha256']):
        raise RuntimeError('The installed App package metadata is not the reviewed 1.4.2 build')
    if active_work(database_url):
        raise RuntimeError('Production has active work')
    try:
        run(['sudo', '-n', 'systemctl', 'stop', 'whu-walker'])
        stopped = True
        for _ in range(45):
            if not active_work(database_url):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Active work did not drain')
        run(['sudo', '-n', 'systemctl', 'stop', 'whu-agent-worker', 'whu-vision-worker', 'whu-vision-trial-worker'])
        backup = json.loads(run([str(APP / 'venv/bin/python'), str(BACKUP_SCRIPT)]))
        if backup.get('status') != 'backup_restore_verified' or backup.get('previous_commit') != old:
            raise RuntimeError('Fresh quiesced production backup failed restore verification')
        run(['git', 'merge', '--ff-only', new])
        switched = True
        apply_flood_configuration(bundle)
        run([str(APP / 'venv/bin/python'), '-m', 'compileall', '-q', 'agents', 'api', 'jobs', 'spatial', 'storage', 'vision'])
        run(['sudo', '-n', 'systemctl', 'start', *SERVICES])
        if not health_ready():
            raise RuntimeError('Health endpoint did not recover')
        states = run(['systemctl', 'is-active', *SERVICES]).splitlines()
        if states != ['active'] * len(SERVICES):
            raise RuntimeError('A production service is inactive')
        if not web_assets_ready():
            raise RuntimeError('The public web cache version or Service Worker entry point is not live')
        if not flood_worker_ready(database_url, MODEL_SHA256):
            raise RuntimeError('The approved flood worker did not become ready')
        package_after = app_package_fingerprint()
        if package_after != package_before:
            raise RuntimeError('The App package changed during a web-only release')
        public_smoke = json.loads(run([str(APP / 'venv/bin/python'),
                                       str(APP / 'scripts/deploy/public_flood_smoke.py'),
                                       bundle['public_smoke_image']]))
        if public_smoke.get('status') != 'passed' or public_smoke.get('road_event_published') is not False:
            raise RuntimeError('Public model smoke check did not pass')
        report = {'status': 'deployed', 'previous_commit': old, 'deployed_commit': run(['git', 'rev-parse', 'HEAD']),
                  'compatible_rollback_commit': rollback, 'backup_directory': backup['backup_directory'],
                  'services': dict(zip(SERVICES, states)), 'local_health': 'ok',
                  'database_migration': 'none', 'apk_version': package_after['version_name'],
                  'apk_version_code': package_after['version_code'], 'apk_sha256': package_after['file_sha256'],
                  'apk_unchanged': package_after == package_before,
                  'service_worker_cache': CACHE_NAME, 'web_asset_version': ASSET_VERSION,
                  'flood_mode': 'assisted_review', 'flood_model_sha256': bundle['model_sha256'],
                  'public_smoke': public_smoke,
                  'completed_at': datetime.now(timezone.utc).isoformat(),
                  'database_restored_during_deployment': False}
        OPS.mkdir(parents=True, exist_ok=True)
        report_path = OPS / ('model_release_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({**report, 'report_path': str(report_path)}))
        return 0
    except Exception as error:
        if stopped:
            try:
                if switched:
                    recovered = rollback_to_backup(new, rollback, backup['backup_directory'])
                else:
                    run(['sudo', '-n', 'systemctl', 'start', *SERVICES])
                    recovered = {'status': 'aborted', 'health_after_recovery': health_ready(),
                                 'head': run(['git', 'rev-parse', 'HEAD'])}
                print(json.dumps({**recovered, 'error_type': type(error).__name__}))
            except Exception as recovery_error:
                print(json.dumps({'status': 'recovery_blocked', 'error_type': type(error).__name__,
                                  'recovery_error_type': type(recovery_error).__name__}))
            return 1
        raise


if __name__ == '__main__':
    try:
        if len(sys.argv) == 5 and sys.argv[1] == '--rollback':
            print(json.dumps(rollback_to_backup(*sys.argv[2:])))
            raise SystemExit(0)
        if len(sys.argv) != 5:
            raise ValueError('expected old, new, rollback and reviewed bundle path')
        raise SystemExit(main(*sys.argv[1:]))
    except Exception as error:
        print(json.dumps({'status': 'precondition_failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
