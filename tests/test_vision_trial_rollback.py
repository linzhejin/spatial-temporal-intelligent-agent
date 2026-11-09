"""Check that the last deployed server still rejects guarded trial results."""
import io
import json
import os
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

from psycopg import sql
import pytest

from storage import database, vision_repository

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_upgrade_preserves_old_jobs_and_workers_in_a_rolled_back_test_schema():
    migrations = Path(__file__).resolve().parents[1] / "storage" / "migrations"
    schema = "trial_upgrade_test_" + uuid.uuid4().hex[:12]
    with database.connect(DATABASE_URL) as conn:
        try:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            for path in sorted(migrations.glob("*.sql")):
                if int(path.name[:3]) <= 17:
                    conn.execute(path.read_text(encoding="utf8"))
            old_job = str(uuid.uuid4())
            conn.execute("INSERT INTO manager_vision_job(job_id,created_by,original_name,media_kind,media_path,sha256,status,result) "
                         "VALUES (%s,'old','original.png','image','original.png',%s,'completed',%s::jsonb)",
                         (old_job, "e" * 64, '{"legacy":"preserved"}'))
            conn.execute("INSERT INTO manager_vision_worker(worker_id,ready,status_detail) VALUES ('old',true,'old-detail')")
            for number in (18, 19):
                path = next(migrations.glob(f"{number:03d}_*.sql"))
                conn.execute(path.read_text(encoding="utf8"))
            saved = conn.execute("SELECT original_name,purpose,trial_feedback,result FROM manager_vision_job WHERE job_id=%s",
                                 (old_job,)).fetchone()
            assert saved == {"original_name": "original.png", "purpose": "operational", "trial_feedback": [],
                             "result": {"legacy": "preserved"}}
            worker = conn.execute("SELECT worker_role,status_detail FROM manager_vision_worker WHERE worker_id='old'").fetchone()
            assert worker == {"worker_role": "shared", "status_detail": "old-detail"}
        finally:
            conn.rollback()


def test_previous_release_cannot_publish_trial_result_after_compatible_migration(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    baseline = "505a5d49c5bfa25a850814c60638f5d96b283405"
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job")
    job = vision_repository.create_job(DATABASE_URL, created_by="rollback-test", original_name="public.png",
        media_kind="image", media_path="rollback-unused.png", sha256="d" * 64, anchor_gcj=None, purpose="trial")
    claimed = vision_repository.claim_next_job(DATABASE_URL, "rollback-trial", allow_trial=True, trial_only=True)
    assert claimed["job_id"] == job["job_id"]
    candidate = {"kind": "possible_accident", "review_required": True, "auto_publish": False}
    vision_repository.finish_job(DATABASE_URL, job["job_id"], "rollback-trial", status="needs_review",
                                 result={"candidates": [candidate]})
    vision_repository.review_candidate(DATABASE_URL, job["job_id"], candidate_index=0,
        review_status="confirmed", review_note="已查看公开测试视频并确认", reviewed_by="rollback-test")
    archived = tmp_path / "old-release"
    archived.mkdir()
    bundle = subprocess.check_output(["git", "archive", baseline], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:") as archive:
        archive.extractall(archived, filter="data")
    code = """
import json, os
import config
config.ROAD_CONDITION_ADMIN_PASSWORD = 'rollback-secret'
config.ROAD_CONDITION_ADMIN_TOKEN = ''
from app import create_app
app = create_app()
app.config.update(TESTING=True, DATABASE_URL=os.environ['DATABASE_URL'], SECRET_KEY='rollback-secret')
client = app.test_client()
login = client.post('/api/admin/login', json={'password':'rollback-secret'})
csrf = login.get_json()['data']['csrf_token']
response = client.post('/api/road-conditions', headers={'X-CSRF-Token':csrf}, json={
  'type':'event', 'name':'试用隔离检查', 'lng':114.363, 'lat':30.5365,
  'source_vision_job_id':os.environ['TRIAL_TEST_JOB_ID'], 'source_vision_candidate_index':0,
  'field_confirmation':'测试确认已查看原始视频和道路'})
print(json.dumps({'status':response.status_code,'error':response.get_json().get('error')}))
"""
    env = dict(os.environ, DATABASE_URL=DATABASE_URL, TRIAL_TEST_JOB_ID=job["job_id"], PYTHONPATH=str(archived))
    run = subprocess.run([sys.executable, "-c", code], cwd=archived, env=env,
                         text=True, encoding="utf8", capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr[-3000:]
    answer = json.loads(run.stdout.strip().splitlines()[-1])
    assert answer == {"status": 400, "error": "simulation_not_allowed"}
