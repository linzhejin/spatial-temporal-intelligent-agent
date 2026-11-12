"""Release evidence must correspond to the tested automatic workflow."""
import hashlib
import json

import pytest

from scripts.deploy.prepare_automatic_flood_bundle import REQUIRED_WORKFLOW_REPORTS, validate_workflow
from vision.model_release import automatic_flood_runtime_config


def _workflow(tmp_path):
    model_hash = "a" * 64
    reports = []
    for name, count in REQUIRED_WORKFLOW_REPORTS.items():
        path = tmp_path / (name + ".xml")
        path.write_text(f'<testsuites><testsuite tests="{count}" failures="0" errors="0" skipped="0" />'
                        '</testsuites>', encoding="utf-8")
        reports.append({"name": name, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    js = tmp_path / "js.log"
    js.write_text("tests 238\npass 238\nfail 0\nskipped 0\n", encoding="utf-8")
    data = {"status": "passed", "scope": "automatic_flood", "model_sha256": model_hash,
            "runtime_config": automatic_flood_runtime_config(), "reports": reports,
            "javascript": {"file": str(js), "sha256": hashlib.sha256(js.read_bytes()).hexdigest(),
                           "tests": 238, "passed": 238, "failed": 0, "skipped": 0}}
    return model_hash, data


def test_workflow_accepts_only_complete_matching_reports(tmp_path):
    model_hash, data = _workflow(tmp_path)
    validate_workflow(data, tmp_path, model_hash)
    data["reports"] = data["reports"][:-1]
    with pytest.raises(ValueError, match="workflow report"):
        validate_workflow(data, tmp_path, model_hash)


def test_workflow_rejects_report_that_changed_after_review(tmp_path):
    model_hash, data = _workflow(tmp_path)
    first = data["reports"][0]
    (tmp_path / "regression.xml").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="workflow report"):
        validate_workflow(data, tmp_path, model_hash)
    assert first["sha256"] != hashlib.sha256((tmp_path / "regression.xml").read_bytes()).hexdigest()
