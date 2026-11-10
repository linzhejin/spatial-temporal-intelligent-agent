"""Create and restore-check a private production backup without emitting secrets."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from datetime import datetime, timezone

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from dotenv import dotenv_values

APP = Path('/home/ubuntu/campus-spatial-intelligence-agent')
BACKUP_ROOT = Path('/home/ubuntu/luojia-release-backups')
TABLES = ('conversation', 'task', 'run', 'manager_vision_job')


def run(arguments, **kwargs):
    result = subprocess.run(arguments, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{Path(arguments[0]).name} failed with code {result.returncode}')
    return result.stdout.strip()


def snapshot(conn):
    counts = {table: conn.execute(sql.SQL('SELECT count(*) FROM {}').format(
        sql.Identifier(table))).fetchone()[0] for table in TABLES}
    fingerprint = conn.execute(
        "SELECT md5(COALESCE(string_agg(md5(task_id::text || state::text), '' "
        "ORDER BY task_id::text), '')) FROM task"
    ).fetchone()[0]
    return {'counts': counts, 'task_state_md5': fingerprint}


def main():
    os.umask(0o077)
    settings = dotenv_values(APP / '.env')
    database_url = settings['DATABASE_URL']
    params = conninfo_to_dict(database_url)
    env = os.environ.copy()
    for source, target in [('host', 'PGHOST'), ('port', 'PGPORT'), ('user', 'PGUSER'),
                           ('password', 'PGPASSWORD'), ('dbname', 'PGDATABASE'),
                           ('sslmode', 'PGSSLMODE')]:
        if params.get(source):
            env[target] = params[source]
    env['PGCONNECT_TIMEOUT'] = '10'
    head = run(['git', '-C', str(APP), 'rev-parse', 'HEAD'])
    with psycopg.connect(database_url) as conn:
        active = dict(conn.execute(
            "SELECT status, count(*) FROM run WHERE status IN ('queued','running') GROUP BY status"
        ).fetchall())
        vision_active = dict(conn.execute(
            "SELECT status, count(*) FROM manager_vision_job "
            "WHERE status IN ('queued','running') AND deleted_at IS NULL GROUP BY status"
        ).fetchall())
        database_bytes = conn.execute('SELECT pg_database_size(current_database())').fetchone()[0]
    if active or vision_active:
        print(json.dumps({'status': 'busy', 'agent_runs': active, 'vision_jobs': vision_active}))
        return 2
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = BACKUP_ROOT / (stamp + '-' + head[:8])
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    dump_path = directory / 'database.dump'
    with psycopg.connect(database_url) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        exported_snapshot = conn.execute('SELECT pg_export_snapshot()').fetchone()[0]
        original = snapshot(conn)
        run(['pg_dump', '--format=custom', '--no-owner', '--no-acl',
             '--snapshot=' + exported_snapshot, '--file=' + str(dump_path)], env=env)
    run(['pg_restore', '--list', str(dump_path)])
    shutil.copy2(APP / '.env', directory / 'application.env')
    events = Path(settings.get('ROAD_CONDITIONS_FILE') or APP / 'data/road_conditions.json')
    if not events.is_absolute():
        events = APP / events
    if not events.is_file():
        raise RuntimeError('Configured road-condition file does not exist')
    shutil.copy2(events, directory / 'road_conditions.json')
    service_hashes = {}
    for service in ('whu-walker', 'whu-agent-worker', 'whu-vision-worker', 'whu-vision-trial-worker'):
        unit = Path('/etc/systemd/system') / (service + '.service')
        if not unit.is_file():
            raise RuntimeError('A required application service configuration is missing')
        shutil.copy2(unit, directory / unit.name)
        service_hashes[service] = hashlib.sha256(unit.read_bytes()).hexdigest()
        dropin = Path('/etc/systemd/system') / (service + '.service.d')
        if dropin.is_dir():
            shutil.copytree(dropin, directory / dropin.name)
    verification_db = 'luojia_restore_verify_' + uuid.uuid4().hex[:12]
    created = False
    try:
        run(['sudo', '-n', '-u', 'postgres', 'createdb', '--owner', params['user'], verification_db])
        created = True
        restore_env = dict(env, PGDATABASE=verification_db)
        run(['pg_restore', '--exit-on-error', '--no-owner', '--no-acl',
             '--dbname=' + verification_db, str(dump_path)], env=restore_env)
        verify_params = dict(params, dbname=verification_db)
        with psycopg.connect(**verify_params) as restored:
            recovered = snapshot(restored)
        if recovered != original:
            raise RuntimeError('Restored counts or task-state checksum differ')
    finally:
        if created:
            run(['sudo', '-n', '-u', 'postgres', 'dropdb', verification_db])
    metadata = {
        'status': 'backup_restore_verified', 'backup_directory': str(directory),
        'previous_commit': head, 'database_bytes': database_bytes,
        'dump_bytes': dump_path.stat().st_size,
        'dump_sha256': hashlib.sha256(dump_path.read_bytes()).hexdigest(),
        'application_env_sha256': hashlib.sha256((directory / 'application.env').read_bytes()).hexdigest(),
        'service_unit_sha256': service_hashes,
        'snapshot': original, 'road_conditions_path': str(events),
        'road_conditions_sha256': hashlib.sha256(events.read_bytes()).hexdigest(),
        'active_agent_runs': active, 'active_vision_jobs': vision_active,
    }
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'backup_failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
