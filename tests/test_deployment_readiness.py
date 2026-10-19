"""发布门槛只接受有来源和现场证据的数据。"""

from scripts.validate.check_deployment_readiness import assess
from scripts.validate.check_agent_research_readiness import (
    CHECKS, REQUIRED_INPUTS, _sha256_paths, _versions, assess_research_readiness,
)


REQUIRED_CHECKS = tuple(CHECKS)


def test_agent_gate_covers_core_task_and_tool_suites_without_db_integration():
    commands = {name: set(command) for name, command in CHECKS.items()}
    assert {"task_planning", "poi_activity", "multi_via", "tool_registry"} <= commands.keys()
    assert {"tests/test_task_planner.py", "tests/test_task_patch.py"} <= commands["task_planning"]
    assert {"tests/test_poi_activity_recommendations.py",
            "tests/test_poi_activity_taxonomy.py"} <= commands["poi_activity"]
    assert "tests/test_multi_via_agent.py" in commands["multi_via"]
    assert "tests/test_agent_tool_registry.py" in commands["tool_registry"]
    assert not any("test_task_repository.py" in command
                   or "test_run_worker.py" in command
                   for command in commands.values())


def _verified_inputs():
    validation = {
        "poi_count_matches_declared": True,
        "annotations": {"coverage_pct_of_edges": 100},
    }
    audit = {
        "poi": {"osm_candidate_source_available": True,
                "missing_open_candidates": [],
                "by_verification_status": {"field_verified": 1}},
        "road": {"placeholder_annotation_count": 0,
                 "review_queue": [], "graph": {"poi_snap_over_50m": []}},
    }
    samples = {"samples": [{"id": "od_01", "field_verification": {
        "status": "field_verified", "evidence": "现场记录-001"}}]}
    comparison = {"samples": [{"id": "od_01", "status": "compared"}]}
    return validation, audit, comparison, samples, {}


def test_runtime_flask_floor_supports_per_request_upload_limits():
    from pathlib import Path
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    flask_requirement = next(
        line.strip().lower() for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip().lower().startswith("flask>=")
    )
    assert flask_requirement.startswith("flask>=3.1")


def test_agent_worker_has_a_managed_service_and_deploy_installs_it():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    worker_unit = root / "ops" / "systemd" / "whu-agent-worker.service"
    assert worker_unit.exists(), "Agent queue needs a systemd worker unit"
    unit_text = worker_unit.read_text(encoding="utf-8")
    assert "python -m jobs.worker" in unit_text
    assert "@APP_DIR@" in unit_text
    deploy_script = (root / "deploy.sh").read_text(encoding="utf-8")
    assert "whu-agent-worker.service" in deploy_script
    assert "systemctl enable --now" in deploy_script
    assert "DATABASE_URL" in deploy_script
    assert "SECRET_KEY" in deploy_script
    worker_source = (root / "jobs" / "worker.py").read_text(encoding="utf-8")
    assert "load_dotenv" in worker_source


def test_vision_readiness_and_worker_share_optional_runtime_dependencies():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    optional_requirements = (root / "requirements-vision.txt").read_text(encoding="utf-8")
    deployment_guide = (root / "docs" / "03_部署指南.md").read_text(encoding="utf-8")
    assert "same venv" in optional_requirements
    assert "Web 共用的虚拟环境" in deployment_guide
    assert "venv/bin/pip install -r requirements-vision.txt" in deployment_guide
    vision_unit = root / "ops" / "systemd" / "whu-vision-worker.service"
    assert vision_unit.exists()
    vision_unit_text = vision_unit.read_text(encoding="utf-8")
    assert "python -m vision.worker" in vision_unit_text
    assert "TimeoutStopSec=300" in vision_unit_text
    assert "manager_vision_worker" in (root / "storage" / "migrations" / "006_vision_worker_heartbeat.sql").read_text(encoding="utf-8")
    deploy_script = (root / "deploy.sh").read_text(encoding="utf-8")
    assert "VISION_WORKER_ENABLED" in deploy_script
    assert 'systemctl restart "${SERVICE_NAME}" "${AGENT_SERVICE_NAME}"' in deploy_script


def test_verified_campus_data_passes_readiness_gate():
    assert assess(*_verified_inputs())["ready"] is True


def test_complete_attribute_master_allows_explicit_unknowns_on_new_edges():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    validation["annotations"]["coverage_pct_of_edges"] = 99.9
    validation["edge_attribute_master"] = {
        "current_graph_edges": 12568,
        "bound_current_edges": 12568,
        "coverage_pct": 100.0,
        "duplicate_bindings": 0,
        "stale_bindings": 0,
        "geometry_binding_mismatches": 0,
        "network_version_matches_current_graph": True,
    }

    result = assess(validation, audit, comparison, samples, reviews)

    assert "road_annotation_gap" not in {row["code"] for row in result["blockers"]}


def test_unreviewed_candidate_and_route_block_deployment():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    audit["poi"]["missing_open_candidates"] = [{"osm_id": "way/123"}]
    samples["samples"][0]["field_verification"] = None
    result = assess(validation, audit, comparison, samples, reviews)
    assert result["ready"] is False
    assert {row["code"] for row in result["blockers"]} >= {
        "unreviewed_open_candidates", "unverified_reference_routes"}


def test_candidate_network_losing_existing_poi_coverage_blocks_deployment():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    audit["road"]["graph"]["candidate_graph_regressions"] = [{
        "poi_id": "poi_001", "production_snap_m": 10.0,
        "candidate_snap_m": 120.0,
    }]
    result = assess(validation, audit, comparison, samples, reviews)
    assert result["ready"] is False
    assert "candidate_graph_coverage_regression" in {
        row["code"] for row in result["blockers"]}


def test_foreign_university_poi_blocks_deployment():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    audit["poi"]["foreign_campus_conflicts"] = [{"poi_id": "poi_001"}]
    result = assess(validation, audit, comparison, samples, reviews)
    assert "foreign_campus_poi_conflicts" in {
        row["code"] for row in result["blockers"]}


def test_unreviewed_large_provider_route_disagreement_blocks_deployment():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    samples["samples"][0]["field_verification"] = None
    comparison["samples"][0].update({
        "geometry_deviation": {"p95_m": 180.0},
        "distance_difference_m": 220.0,
    })
    result = assess(validation, audit, comparison, samples, reviews)
    assert "unresolved_route_disagreements" in {
        row["code"] for row in result["blockers"]}


def test_release_route_without_building_geometry_check_blocks_deployment():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    comparison["own_network"] = {"course_release_fingerprint": "release-123"}
    comparison["samples"][0]["hazard_assessment"] = {
        "status": "building_layer_missing",
    }

    result = assess(validation, audit, comparison, samples, reviews)

    assert "unassessed_route_geometry_hazards" in {
        row["code"] for row in result["blockers"]}


def test_unreviewed_building_crossing_edge_blocks_release_route():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    comparison["own_network"] = {"course_release_fingerprint": "release-123"}
    comparison["samples"][0]["hazard_assessment"] = {
        "status": "checked",
        "suspected_crossing_edge_ids": [{"edge_id": [1, 2, 0]}],
    }

    result = assess(validation, audit, comparison, samples, reviews)

    assert "unreviewed_route_building_crossings" in {
        row["code"] for row in result["blockers"]}


def test_evidence_backed_review_resolves_building_crossing_gate():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    comparison["own_network"] = {"course_release_fingerprint": "release-123"}
    comparison["samples"][0]["hazard_assessment"] = {
        "status": "checked",
        "suspected_crossing_edge_ids": [{"edge_id": [1, 2, 0]}],
    }
    reviews["road_decisions"] = [{
        "edge_id": [1, 2, 0], "verification_status": "institution_verified",
        "evidence": "校方通行说明", "verified_at": "2026-09-28",
    }]

    result = assess(validation, audit, comparison, samples, reviews)

    assert "unreviewed_route_building_crossings" not in {
        row["code"] for row in result["blockers"]}


def test_candidate_graph_deviation_and_building_crossing_block_replacement():
    validation, audit, comparison, samples, reviews = _verified_inputs()
    candidate = {"samples": [{"id": "od_01", "status": "compared",
               "geometry_deviation": {"p95_m": 180.0},
               "distance_difference_m": 200.0,
               "hazard_assessment": {"suspected_crossing_edge_ids": [
                   {"edge_id": [1, 2, 3]}]}}]}
    samples["samples"][0]["field_verification"] = None

    result = assess(validation, audit, comparison, samples, reviews, candidate)

    assert {"candidate_route_disagreements", "candidate_building_crossing_suspicions"} <= {
        row["code"] for row in result["blockers"]}


def test_research_readiness_separates_agent_and_preference_gates():
    ready_checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    result = assess_research_readiness(
        ready_checks,
        deployment={"ready": True, "blockers": []},
        attributes={"ready": False, "degraded": True, "blockers": ["placeholder data"]},
    )
    assert result["ready"] is True
    assert result["scope"] == "agent_main_experiment"
    assert result["scopes"]["agent_main_experiment"]["ready"] is True
    assert result["scopes"]["preference_attributes"]["ready"] is False
    assert "attribute_master_data" in result["scopes"]["preference_attributes"]["blockers"]


def test_research_readiness_does_not_claim_all_scopes_pass_from_agent_inputs():
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    result = assess_research_readiness(
        checks,
        deployment={"ready": True, "blockers": []},
        attributes={"ready": True, "degraded": False, "blockers": []},
    )
    assert result["ready"] is True
    assert result["blockers"] == {}
    assert result["scopes"]["sampled_route_evidence"]["ready"] is True
    assert result["scopes"]["vision"]["detection"]["ready"] is False


def test_research_readiness_route_and_campus_evidence_do_not_block_agent():
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    deployment = {"ready": False, "blockers": [
        {"code": "unverified_pois", "count": 2},
        {"code": "unverified_reference_routes", "count": 3},
    ]}
    result = assess_research_readiness(checks, deployment, {"ready": False})
    assert result["ready"] is True
    assert result["scopes"]["sampled_route_evidence"]["ready"] is False
    assert result["scopes"]["preference_attributes"]["ready"] is False


def test_research_readiness_missing_code_check_and_structural_error_block_agent():
    result = assess_research_readiness(
        {"routing_policy": {"passed": True}},
        {"ready": False, "blockers": [{"code": "structural_errors", "count": 1}]},
        {"ready": True, "degraded": False},
    )
    assert result["ready"] is False
    assert "code_checks" in result["blockers"]
    assert "campus_structure" in result["blockers"]


def test_missing_tool_registry_result_blocks_agent_experiment():
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS
              if name != "tool_registry"}
    result = assess_research_readiness(
        checks, {"ready": True, "blockers": []}, {"ready": True},
    )
    assert result["ready"] is False
    assert result["blockers"]["code_checks"] == ["tool_registry"]


def test_research_versions_hash_changed_inputs_and_report_missing_files(tmp_path):
    (tmp_path / "data").mkdir()
    source = tmp_path / "data" / "edge_overrides.json"
    source.write_text("{}", encoding="utf-8")
    first = _sha256_paths(["data/edge_overrides.json", "data/missing.json"], root=tmp_path)
    source.write_text('{"blocked": true}', encoding="utf-8")
    second = _sha256_paths(["data/edge_overrides.json", "data/missing.json"], root=tmp_path)
    assert first["sha256"] != second["sha256"]
    assert first["missing"] == ["data/missing.json"]


def test_effective_snapshot_changes_with_runtime_code_requirements_and_environment(
        monkeypatch, tmp_path):
    import scripts.validate.check_agent_research_readiness as readiness

    (tmp_path / "agents").mkdir()
    planner = tmp_path / "agents" / "planner.py"
    planner.write_text("ROUTE_POLICY = 1\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("networkx==3.0\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "_runtime_model_config", lambda: {
        "sha256": "a" * 64, "values": {},
    })
    monkeypatch.setattr(readiness, "_event_state_version", lambda: {
        "sha256": "b" * 64, "active_sha256": "c" * 64,
        "files": ["event_state"], "missing": [],
    })
    environment = {"sha256": "e" * 64, "files": ["runtime_environment"],
                   "missing": [], "values": {"python": "3.13"}}
    monkeypatch.setattr(readiness, "_runtime_environment_version", lambda: environment)
    monkeypatch.setattr(readiness, "_git", lambda *_args: "d" * 40)

    first = _versions()
    planner.write_text("ROUTE_POLICY = 2\n", encoding="utf-8")
    changed_code = _versions()
    requirements.write_text("networkx==3.1\n", encoding="utf-8")
    changed_requirements = _versions()
    environment = {"sha256": "f" * 64, "files": ["runtime_environment"],
                   "missing": [], "values": {"python": "3.14"}}
    changed_environment = _versions()

    assert {"runtime_code", "requirements", "runtime_environment"} <= set(REQUIRED_INPUTS)
    assert first["runtime_code"]["files"] == ["agents/planner.py"]
    assert "vision" not in readiness.RUNTIME_SOURCE_PATHS
    assert first["effective_inputs_sha256"] != changed_code["effective_inputs_sha256"]
    assert changed_code["effective_inputs_sha256"] != changed_requirements["effective_inputs_sha256"]
    assert changed_requirements["effective_inputs_sha256"] != changed_environment[
        "effective_inputs_sha256"]


def test_research_readiness_vision_evidence_is_independent_by_category():
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    vision = {"task": "campus_uav_field_readiness", "campus_field_data_present": True,
              "checks": {
                  "object_detection": {"accuracy_claim_ready": True},
                  "event_evaluation": {"status": "ready", "event_metric_targets": {
                      "congestion": {"met": True, "flight_coverage_met": True,
                                     "positive_test_flight_count": 1},
                      "crowding": {"met": False},
                  }},
                  "visible_water_segmentation": {"status": "missing"},
              }}
    result = assess_research_readiness(
        checks, {"ready": True, "blockers": []}, {"ready": True}, vision=vision,
    )
    categories = result["scopes"]["vision"]
    assert categories["detection"]["ready"] is True
    assert categories["congestion"]["ready"] is True
    assert categories["crowding"]["ready"] is False
    assert categories["accident"]["ready"] is False
    assert categories["flooding"]["ready"] is False


def test_malformed_vision_report_fails_closed_for_each_category():
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    vision = {"task": "campus_uav_field_readiness", "campus_field_data_present": True,
              "checks": {"object_detection": ["ready"],
                         "event_evaluation": {"event_metric_targets": ["ready"]},
                         "models": ["ready"]}}
    result = assess_research_readiness(
        checks, {"ready": True, "blockers": []}, {"ready": True}, vision=vision,
    )
    assert all(not item["ready"] for item in result["scopes"]["vision"].values())


def test_missing_snapshot_input_blocks_agent_gate():
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    versions = {name: {"files": [name], "missing": []} for name in REQUIRED_INPUTS}
    versions["edge_overrides"] = {"files": [], "missing": ["data/edge_overrides.json"]}
    result = assess_research_readiness(
        checks, {"ready": True, "blockers": []}, {"ready": True}, versions=versions,
    )
    assert result["ready"] is False
    assert result["blockers"]["snapshot_inputs"] == ["edge_overrides"]


def test_agent_snapshot_does_not_require_preference_or_vision_evidence(
        monkeypatch, tmp_path):
    import json
    import scripts.validate.check_agent_research_readiness as readiness

    audit = tmp_path / "audit"
    audit.mkdir()
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "AUDIT", audit)
    monkeypatch.setattr(readiness, "_run_checks", lambda: {
        name: {"passed": True} for name in REQUIRED_CHECKS
    })
    monkeypatch.setattr(readiness, "_deployment_result", lambda: {
        "ready": False, "blockers": [{"code": "unverified_reference_routes", "count": 1}],
    })
    monkeypatch.setattr(readiness, "_versions", lambda: {
        **{name: {"files": [name], "missing": []} for name in REQUIRED_INPUTS},
        "effective_inputs_sha256": "a" * 64,
    })
    monkeypatch.setattr(readiness, "_git", lambda *_args: "")

    assert readiness.main(["--write-snapshot"]) == 0
    snapshot = json.loads((tmp_path / "experiments" / "research_snapshot.json").read_text(
        encoding="utf-8"))
    assert snapshot["ready"] is True
    assert snapshot["scopes"]["sampled_route_evidence"]["ready"] is False
    assert snapshot["scopes"]["preference_attributes"]["ready"] is False
    assert snapshot["scopes"]["vision"]["detection"]["ready"] is False


def test_diagnostic_input_snapshot_is_explicit_and_never_overwritten(
        monkeypatch, tmp_path, capsys):
    import json
    import pytest
    import scripts.validate.check_agent_research_readiness as readiness

    monkeypatch.setattr(readiness, "_versions", lambda: {
        "effective_inputs_sha256": "a" * 64,
        "poi": {"sha256": "b" * 64, "files": ["data/pois.json"], "missing": []},
    })
    monkeypatch.setattr(readiness, "_git", lambda *_args: "c" * 40)
    monkeypatch.setattr(readiness, "_run_checks", lambda: pytest.fail(
        "diagnostic capture must not run the readiness test suite"))
    path = tmp_path / "diagnostic.json"

    assert readiness.main(["--capture-inputs", str(path)]) == 0
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["snapshot_kind"] == "diagnostic"
    assert row["formal_eligible"] is False
    assert row["versions"]["effective_inputs_sha256"] == "a" * 64
    assert "api_key" not in json.dumps(row).lower()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        readiness.main(["--capture-inputs", str(path)])


def test_inline_deploy_checks_load_dotenv_from_the_project_directory():
    from pathlib import Path
    deploy_script = (Path(__file__).resolve().parents[1] / "deploy.sh").read_text(encoding="utf-8")
    assert deploy_script.count('load_dotenv(".env")') == 2


def test_first_road_event_migration_stops_old_web_writer_before_copy():
    from pathlib import Path
    deploy_script = (Path(__file__).resolve().parents[1] / "deploy.sh").read_text(encoding="utf-8")
    stop = deploy_script.index('sudo systemctl stop "$SERVICE_NAME"')
    copy = deploy_script.index('scripts/deploy/migrate_road_conditions.py')
    restart = deploy_script.index('sudo systemctl restart "${SERVICE_NAME}"')
    assert stop < copy < restart
    assert 'cp "$APP_DIR/data/road_conditions.json" "$RUNTIME_CONDITIONS"' not in deploy_script
    assert '"$APP_DIR/venv/bin/python" scripts/deploy/migrate_road_conditions.py' in deploy_script
