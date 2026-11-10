"""Check a public trial image against the deployed worker without publishing a road event."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dotenv import dotenv_values
import requests

BASE = 'https://whuspati.online'
APP = Path('/home/ubuntu/campus-spatial-intelligence-agent')
MODEL_SHA256 = '1ef47ccd3aef9cafee359485cfb38c65edcd6625e1552d02e23afacd98f7eb52'
SERVICES = ('whu-walker', 'whu-agent-worker', 'whu-vision-worker', 'whu-vision-trial-worker')
REGION = [{'id': 'public-smoke-road', 'kind': 'road_surface',
           'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}]


def check_host():
    memory = Path('/proc/meminfo').read_text(encoding='ascii')
    available = next(int(line.split()[1]) for line in memory.splitlines() if line.startswith('MemAvailable:'))
    if available < 128 * 1024:
        raise RuntimeError('Host memory fell below the acceptance limit')
    state = subprocess.run(['systemctl', 'is-active', *SERVICES], capture_output=True, text=True, timeout=10)
    if state.returncode or state.stdout.splitlines() != ['active'] * len(SERVICES):
        raise RuntimeError('An application service is inactive')


def get_data(response, expected=200):
    if response.status_code != expected:
        raise RuntimeError('Public API returned unexpected HTTP status ' + str(response.status_code))
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get('data'), dict):
        raise RuntimeError('Public API response is missing its data')
    return body['data']


def authenticate(session, settings):
    password = settings.get('ROAD_CONDITION_ADMIN_PASSWORD')
    if not password:
        raise RuntimeError('Administrator login is not configured')
    data = get_data(session.post(BASE + '/api/admin/login', json={'password': password}, timeout=15))
    if not data.get('is_admin') or not data.get('csrf_token'):
        raise RuntimeError('Administrator login failed')
    session.headers['X-CSRF-Token'] = data['csrf_token']


def submit(session, image, name):
    with image.open('rb') as stream:
        data = get_data(session.post(
            BASE + '/api/manager/vision-jobs',
            data={'purpose': 'trial', 'observation_regions': json.dumps(REGION)},
            files={'media': (name, stream, 'image/png')}, timeout=30,
        ), expected=202)
    return data['job']['job_id']


def read_job(session, job_id):
    return get_data(session.get(BASE + '/api/manager/vision-jobs/' + job_id, timeout=15))['job']


def wait_for(session, job_id, statuses, timeout=160):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_host()
        job = read_job(session, job_id)
        if job['status'] in statuses:
            return job
        if job['status'] in {'failed', 'cancelled', 'completed', 'needs_review'}:
            raise RuntimeError('Public trial ended with unexpected status ' + job['status'])
        time.sleep(0.35)
    raise RuntimeError('Public trial did not reach the expected state')


def main(image):
    image = Path(image).resolve(strict=True)
    root = Path('/home/ubuntu/luojia-models').resolve()
    if not image.is_relative_to(root) or image.suffix.lower() != '.png':
        raise ValueError('Public smoke image must belong to the reviewed model bundle')
    check_host()
    session = requests.Session()
    session.trust_env = False
    authenticate(session, dotenv_values(APP / '.env'))
    capability = get_data(session.get(BASE + '/api/manager/vision-status', timeout=15))['flood_assistance']
    if (capability.get('ready') is not True or capability.get('mode') != 'assisted_review'
            or capability.get('supported_media') != ['image'] or capability.get('field_validated') is not False
            or capability.get('model_sha256') != MODEL_SHA256):
        raise RuntimeError('Public flood capability differs from the accepted model')
    pending = []
    try:
        job_id = submit(session, image, 'public-flood-validation.png')
        pending.append(job_id)
        started = time.monotonic()
        job = wait_for(session, job_id, {'needs_review'})
        result = job.get('result') or {}
        if ((result.get('trial_coverage') or {}).get('flood') != 'checked'
                or (result.get('model', {}).get('flood_model') or {}).get('sha256') != MODEL_SHA256):
            raise RuntimeError('Public trial did not use the accepted flood model')
        candidates = result.get('candidates') or []
        flood = next((index for index, candidate in enumerate(candidates)
                      if candidate.get('kind') == 'possible_flooding'), None)
        if flood is None:
            raise RuntimeError('Public trial did not produce a flood candidate')
        review = get_data(session.post(
            BASE + f'/api/manager/vision-jobs/{job_id}/candidates/{flood}/review',
            json={'status': 'confirmed', 'note': '公开图片工程验收：确认已查看模型候选，不发布路况。'}, timeout=15,
        ))
        if not review.get('job'):
            raise RuntimeError('Public candidate review was not recorded')
        pending.remove(job_id)
        elapsed = time.monotonic() - started
        cancel_id = submit(session, image, 'public-flood-cancellation.png')
        pending.append(cancel_id)
        wait_for(session, cancel_id, {'running'}, timeout=30)
        cancel_started = time.monotonic()
        get_data(session.post(BASE + '/api/manager/vision-jobs/' + cancel_id + '/cancel', timeout=15))
        cancelled = wait_for(session, cancel_id, {'cancelled'}, timeout=5)
        cancel_seconds = time.monotonic() - cancel_started
        if cancelled['status'] != 'cancelled' or cancel_seconds > 5:
            raise RuntimeError('Public cancellation exceeded five seconds')
        pending.remove(cancel_id)
        check_host()
        return {'status': 'passed', 'purpose': 'trial', 'model_sha256': MODEL_SHA256,
                'job_id': job_id, 'candidate_index': flood, 'review': 'confirmed',
                'elapsed_seconds': round(elapsed, 3), 'cancelled_job_id': cancel_id,
                'cancel_seconds': round(cancel_seconds, 3), 'road_event_published': False}
    finally:
        for job_id in pending:
            try:
                session.post(BASE + '/api/manager/vision-jobs/' + job_id + '/cancel', timeout=5)
            except requests.RequestException:
                pass
        session.close()


if __name__ == '__main__':
    os.umask(0o077)
    try:
        if len(sys.argv) != 2:
            raise ValueError('expected path to the public validation image')
        print(json.dumps(main(sys.argv[1]), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}))
        raise SystemExit(1)
