"""Exercise the public smoke client against real models and an isolated HTTP/database stack."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

import pytest
from werkzeug.serving import make_server

from storage import database, vision_repository

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures('isolated_test_database')


@pytest.mark.skipif(os.getenv('RUN_FLOOD_PUBLIC_SMOKE_E2E') != '1',
                    reason='requires isolated PostgreSQL and the real flood model')
def test_public_smoke_runs_real_trial_review_and_cancellation(monkeypatch, tmp_path):
    import api.routes as routes
    import config
    from app import create_app
    from scripts.deploy import public_flood_smoke as smoke
    from vision import worker
    from vision.specialized import DoodleverseFloodSegmenter

    root = Path(r'D:\LuoJiaValidation')
    source = root / 'model_comparison_20261109/flood/road_inundation_delta_ian_2024/samples_v1_2/test/HURRICANE-IAN_000352_image.png'
    flood_path = root / 'doodleverse_floodnet_v7_fp32.onnx'
    runtime = root / 'flood_runtime_release_20261110/Scripts/python.exe'
    traffic = Path(r'D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx')
    assert all(path.is_file() for path in (source, flood_path, runtime, traffic))
    image = tmp_path / 'public-smoke.png'
    shutil.copy2(source, image)
    uploads = tmp_path / 'uploads'
    uploads.mkdir()
    (tmp_path / '.env').write_text('ROAD_CONDITION_ADMIN_PASSWORD=isolated-public-smoke\n', encoding='utf-8')
    model_hash = hashlib.sha256(flood_path.read_bytes()).hexdigest()
    for name, value in {
        'ROAD_CONDITION_ADMIN_PASSWORD': 'isolated-public-smoke', 'ROAD_CONDITION_ADMIN_TOKEN': '',
        'VISION_UPLOAD_DIR': str(uploads), 'VISION_MODEL_PATH': str(traffic),
        'VISION_TRIAL_ENABLED': True, 'VISION_FLOOD_ENABLED': True,
        'VISION_FLOOD_RELEASE_MODE': 'assisted_review', 'VISION_FLOOD_MODEL_PATH': str(flood_path),
        'VISION_FLOOD_REVISION': 'doodleverse-floodnet-resunet-v7',
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(routes, '_vision_flood_assistance_ready', lambda caps: bool(
        caps.get('flood_assistance_supported') and caps.get('flood_assistance_model_sha256') == model_hash
    ))
    monkeypatch.setenv('DATABASE_URL', DATABASE_URL)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute('TRUNCATE manager_vision_job, manager_vision_worker')
    capability = {'flood_assistance_supported': True, 'flood_assistance_release_status': 'test_injected',
                  'flood_assistance_model_id': 'doodleverse-floodnet-resunet-v7',
                  'flood_assistance_model_version': config.VISION_FLOOD_REVISION,
                  'flood_assistance_model_sha256': model_hash, 'trial_flood_supported': True,
                  'trial_flood_media': 'image'}
    stop = threading.Event()
    model = DoodleverseFloodSegmenter(str(flood_path), model_version=config.VISION_FLOOD_REVISION,
                                     runtime_python=str(runtime))
    model.model_sha256 = model_hash

    def worker_loop():
        while not stop.is_set():
            vision_repository.heartbeat_worker(DATABASE_URL, 'smoke-operational', ready=True, role='operational')
            vision_repository.heartbeat_worker(DATABASE_URL, 'smoke-trial', ready=True, role='trial',
                                               status_detail=json.dumps(capability))
            outcome = worker.process_next_job(DATABASE_URL, 'smoke-actual-worker',
                                              trial_flood_segmenter=model, trial_only=True,
                                              assisted_flood_supported=True)
            if outcome is None:
                stop.wait(0.2)

    thread = threading.Thread(target=worker_loop, daemon=True)
    thread.start()
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY='isolated-public-smoke')
    server = make_server('127.0.0.1', 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setattr(smoke, 'BASE', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setattr(smoke, 'APP', tmp_path)
    monkeypatch.setattr(smoke, 'MODEL_ROOT', tmp_path)
    monkeypatch.setattr(smoke, 'check_host', lambda: None)
    pids = iter([100, 101])
    monkeypatch.setattr(smoke, 'trial_worker_pid', lambda: next(pids))
    try:
        for _ in range(50):
            statuses = vision_repository.ready_worker_statuses(DATABASE_URL)
            if any(status.get('worker_role') == 'trial' for status in statuses):
                break
            stop.wait(0.1)
        result = smoke.main(image)
        assert result['status'] == 'passed' and result['road_event_published'] is False
        assert result['cancel_seconds'] <= 5 and result['model_sha256'] == model_hash
        reviewed = vision_repository.get_job(DATABASE_URL, result['job_id'])
        assert reviewed['purpose'] == 'trial'
        assert reviewed['candidate_reviews'][str(result['candidate_index'])]['status'] == 'confirmed'
        cancelled = vision_repository.get_job(DATABASE_URL, result['cancelled_job_id'])
        assert cancelled['status'] == 'cancelled'
    finally:
        stop.set()
        thread.join(timeout=15)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
