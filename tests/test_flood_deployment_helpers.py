"""Safety checks for the flood-only release configuration and evidence gate."""
import hashlib
import json
from pathlib import Path

import pytest
from dotenv import dotenv_values

from scripts.deploy import flood_release, public_flood_smoke
from scripts.deploy.prepare_flood_bundle import validate_resource


def test_release_configuration_preserves_existing_values_and_disables_accident(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("KEEP_VALUE=preserved\nVISION_ACCIDENT_ENABLED=true\n", encoding="utf-8")
    monkeypatch.setattr(flood_release, "APP", tmp_path)
    flood_release.apply_flood_configuration({
        "model_path": "/model/flood.onnx", "runtime_python": "/runtime/python",
        "release_manifest": "/model/release.json",
    })
    values = dotenv_values(env)
    assert values["KEEP_VALUE"] == "preserved"
    assert values["VISION_ACCIDENT_ENABLED"] == "false"
    assert values["VISION_FLOOD_ENABLED"] == "true"
    assert values["VISION_FLOOD_RELEASE_MODE"] == "assisted_review"
    assert values["VISION_FLOOD_PROBABILITY_THRESHOLD"] == "0.001"
    assert values["VISION_FLOOD_MIN_AREA_RATIO"] == "0.08"
    assert values["VISION_FLOOD_TIMEOUT_SECONDS"] == "120"


def test_release_rejects_failed_evidence_even_when_manifest_claims_passed(monkeypatch, tmp_path):
    root = tmp_path / "floodnet-v7-test"
    root.mkdir()
    model = root / "model.onnx"
    model.write_bytes(b"fixture model")
    runtime = root / "runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"python")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    evidence = {}
    for kind in ("parity", "resource", "workflow"):
        path = root / (kind + ".json")
        path.write_text(json.dumps({"status": "failed" if kind == "resource" else "passed"}), encoding="utf-8")
        evidence[kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = root / "release.json"
    manifest.write_text(json.dumps({"models": {"flood": {
        "mode": "assisted_review", "field_validated": False, "model_sha256": model_hash,
        "model_revision": "doodleverse-floodnet-resunet-v7",
        "acceptance": {kind: {"status": "passed", **item} for kind, item in evidence.items()},
    }}}), encoding="utf-8")
    target = "a" * 40
    bundle = root / "bundle.json"
    smoke = root / "public-smoke.png"
    smoke.write_bytes(b"fixture image")
    bundle.write_text(json.dumps({"target_commit": target, "model_path": str(model),
        "model_sha256": model_hash, "runtime_python": str(runtime), "release_manifest": str(manifest),
        "public_smoke_image": str(smoke), "public_smoke_sha256": hashlib.sha256(smoke.read_bytes()).hexdigest(),
        "evidence": evidence, "tested_source_sha256": {}}), encoding="utf-8")
    monkeypatch.setattr(flood_release, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(flood_release, "MODEL_SHA256", model_hash)
    with pytest.raises(ValueError, match="resource evidence"):
        flood_release.validate_bundle(bundle, target)


def _valid_resource_report():
    cases = []
    for name in ["preflight", *(str(index) for index in range(10)), "cancel", "timeout", "failure_retry"]:
        worker = {"status_check": "passed"}
        if name == "cancel":
            worker["cancel_to_terminal_seconds"] = 0.5
        if name == "timeout":
            worker["status"] = "failed"
        if name == "failure_retry":
            worker["retry_terminal_status"] = "needs_review"
        cases.append({"case": name, "exit_code": 0, "peak_bytes": 620 * 1024 * 1024,
                      "swap_max": "0", "cpu_max": "75000 100000",
                      "memory_events": {"oom_kill": 0}, "worker": worker})
    return {"status": "passed", "vision_services_active": True, "other_services_healthy": True,
            "cases": cases}


def test_bundle_accepts_only_complete_resource_evidence():
    valid = _valid_resource_report()
    validate_resource(valid)
    valid["cases"][3]["memory_events"]["oom_kill"] = 1
    with pytest.raises(ValueError, match="violated"):
        validate_resource(valid)


def test_bundle_rejects_incomplete_or_too_slow_cancellation():
    incomplete = _valid_resource_report()
    incomplete["cases"] = [case for case in incomplete["cases"] if case["case"] != "9"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_resource(incomplete)
    slow = _valid_resource_report()
    next(case for case in slow["cases"] if case["case"] == "cancel")["worker"]["cancel_to_terminal_seconds"] = 5
    with pytest.raises(ValueError, match="Cancellation"):
        validate_resource(slow)


def test_public_smoke_submits_only_an_unlocated_trial_image(tmp_path):
    class Response:
        status_code = 202

        def json(self):
            return {"data": {"job": {"job_id": "test-job"}}}

    class Session:
        def post(self, url, **kwargs):
            assert url.endswith("/api/manager/vision-jobs")
            assert kwargs["data"]["purpose"] == "trial"
            assert set(kwargs["data"]) == {"purpose", "observation_regions"}
            assert json.loads(kwargs["data"]["observation_regions"])[0]["kind"] == "road_surface"
            return Response()

    image = tmp_path / "public.png"
    image.write_bytes(b"fixture image")
    assert public_flood_smoke.submit(Session(), image, "public.png") == "test-job"


def test_public_smoke_treats_failed_inference_as_error(monkeypatch):
    monkeypatch.setattr(public_flood_smoke, "check_host", lambda: None)
    monkeypatch.setattr(public_flood_smoke, "read_job", lambda *_: {"status": "failed"})
    with pytest.raises(RuntimeError, match="unexpected status failed"):
        public_flood_smoke.wait_for(object(), "test-job", {"needs_review"})


def test_rollback_restores_verified_environment_and_starts_all_services(monkeypatch, tmp_path):
    root = tmp_path / 'backups'
    backup = root / 'accepted'
    backup.mkdir(parents=True)
    saved = backup / 'application.env'
    saved.write_text('DATABASE_URL=postgresql://previous\nVISION_FLOOD_ENABLED=false\n', encoding='utf-8')
    (backup / 'manifest.json').write_text(json.dumps({
        'status': 'backup_restore_verified', 'previous_commit': 'old',
        'application_env_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
    }), encoding='utf-8')
    app = tmp_path / 'app'
    app.mkdir()
    (app / '.env').write_text('DATABASE_URL=postgresql://current\nVISION_FLOOD_ENABLED=true\n', encoding='utf-8')
    monkeypatch.setattr(flood_release, 'APP', app)
    monkeypatch.setattr(flood_release, 'BACKUP_ROOT', root)
    head = ['new']
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ['git', 'rev-parse', 'HEAD']:
            return head[0]
        if args[:3] == ['git', 'switch', '--detach']:
            head[0] = args[3]
        if args[:2] == ['systemctl', 'is-active']:
            return '\n'.join(['active'] * len(flood_release.SERVICES))
        return ''

    monkeypatch.setattr(flood_release, 'run', fake_run)
    monkeypatch.setattr(flood_release, 'cancel_vision_jobs', lambda _url: 2)
    monkeypatch.setattr(flood_release, 'active_work', lambda _url: 0)
    monkeypatch.setattr(flood_release, 'finalize_stopped_vision_jobs', lambda _url: 1)
    monkeypatch.setattr(flood_release, 'cancel_active_agent_runs', lambda _url: 3)
    monkeypatch.setattr(flood_release, 'health_ready', lambda: True)
    result = flood_release.rollback_to_backup('new', 'old', backup)
    assert result['status'] == 'rolled_back' and result['head'] == 'old'
    assert (app / '.env').read_bytes() == saved.read_bytes()
    assert ['sudo', '-n', 'systemctl', 'start', *flood_release.SERVICES] in calls
    assert calls.index(['sudo', '-n', 'systemctl', 'stop', 'whu-walker']) < calls.index(
        ['git', 'switch', '--detach', 'old'])


def test_rollback_rejects_changed_backup_before_stopping_services(monkeypatch, tmp_path):
    root = tmp_path / 'backups'
    backup = root / 'accepted'
    backup.mkdir(parents=True)
    (backup / 'application.env').write_text('changed', encoding='utf-8')
    (backup / 'manifest.json').write_text(json.dumps({
        'status': 'backup_restore_verified', 'previous_commit': 'old', 'application_env_sha256': 'wrong',
    }), encoding='utf-8')
    monkeypatch.setattr(flood_release, 'BACKUP_ROOT', root)
    monkeypatch.setattr(flood_release, 'run', lambda *_args: pytest.fail('must not stop or switch services'))
    with pytest.raises(ValueError, match='has changed'):
        flood_release.rollback_to_backup('new', 'old', backup)
