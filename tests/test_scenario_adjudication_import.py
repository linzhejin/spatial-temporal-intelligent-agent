from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

def _importer():
    module_name = "experiments.import_scenario_adjudication"
    assert importlib.util.find_spec(module_name) is not None, (
        "scenario adjudication importer should exist"
    )
    from experiments.import_scenario_adjudication import import_review_workbook

    return import_review_workbook


def _source_dataset() -> dict:
    return {
        "protocol_version": "scenario-v1",
        "version": "test-dev-draft",
        "split": "development",
        "repeats": 1,
        "authoring_status": "draft",
        "cases": [{
            "id": "case-1",
            "category": "mode_access",
            "template_group": "walk_route",
            "od_group": "a_to_b",
            "snapshot_id": "PENDING_FROZEN_SNAPSHOT",
            "adjudication": {
                "status": "draft", "method": None,
                "reviewer": None, "evidence": [],
            },
            "messages": [{"role": "user", "content": "从甲到乙步行"}],
            "expected_turns": [{
                "status": "completed", "kind": "route", "feasibility": "unknown",
                "start_poi_id": None, "end_poi_id": None, "via_poi_ids": [],
                "travel_mode": None, "strategy": None, "hard_constraints": {},
                "required_outputs": [], "candidate_poi_ids": [],
                "clarification_slot": None, "allowed_unknowns": [],
                "intent_label": None, "expected_tools": None,
            }],
        }],
    }


CASE_HEADERS = [
    "场景ID", "源快照标记", "用户轮次数", "源审查状态", "审查状态", "结论摘要",
    "审查人", "审查方法", "证据/参考", "理由与备注",
]
TURN_HEADERS = [
    "场景ID", "轮次", "源快照标记", "此前对话", "本轮用户原话", "审查状态",
    "预期响应状态", "预期输出类型", "任务判读", "可行性", "起点 POI ID",
    "终点 POI ID", "途经 POI ID（有序）", "出行方式", "路线策略", "硬约束 JSON",
    "必需输出（JSON列表）", "候选 POI ID（JSON有序列表）", "澄清槽位",
    "允许未知项（JSON列表）", "意图标签", "可接受工具调用序列（JSON二维数组）",
    "事件快照/证据ID", "证据来源/参考", "审查人",
    "审查日期", "理由与备注",
]


class _FakeSheet:
    def __init__(self, title, rows):
        self.title = title
        self._rows = rows

    def iter_rows(self, *, min_row=1, values_only=False):
        return iter(self._rows[min_row - 1:])


class _FakeWorkbook:
    def __init__(self, source_hash, *, complete=True, expected_tools='[["resolve_poi","plan_route"]]', intent="route_planning"):
        self.sheetnames = ["审查说明", "场景审查", "逐轮裁决"]
        self.sheets = {
            "审查说明": _FakeSheet("审查说明", [["源文件", f"experiments/datasets/scenarios_dev_v1.json | SHA-256 {source_hash}"]]),
            "场景审查": _FakeSheet("场景审查", [CASE_HEADERS, [
                "case-1", "PENDING_FROZEN_SNAPSHOT", 1, "draft", "已审", "独立结论-actual",
                "reviewer-1", "manual_independent", "review-log-1", "case rationale-actual",
            ]]),
            "逐轮裁决": _FakeSheet("逐轮裁决", [TURN_HEADERS, [
                "case-1", 1, "PENDING_FROZEN_SNAPSHOT", "", "从甲到乙步行",
                "已审" if complete else "未审", "completed", "route", "直接规划步行路线",
                "feasible", "poi-a", "poi-b", "[]", "walk", "shortest", "{}",
                '["route"]', "[]", None, "[]", intent, expected_tools,
                None, "route-review-1", "reviewer-1", "2026-10-17", "路径判据已复核",
            ]]),
        }

    def __getitem__(self, key):
        return self.sheets[key]

    def close(self):
        pass


def _write_review_workbook(path: Path, source_path: Path, *, complete: bool = True,
                           expected_tools='[["resolve_poi","plan_route"]]',
                           intent="route_planning"):
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.write_bytes(b"mock-xlsx-payload")
    return _FakeWorkbook(source_hash, complete=complete, expected_tools=expected_tools,
                         intent=intent)


def test_imports_complete_manual_review_to_new_adjudicated_development_file(tmp_path, monkeypatch):
    import_review_workbook = _importer()
    from experiments import import_scenario_adjudication as importer
    source = _source_dataset()
    source_path = tmp_path / "scenarios_dev_v1.json"
    source_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    workbook_path = tmp_path / "review.xlsx"
    output_path = tmp_path / "adjudicated.json"
    workbook = _write_review_workbook(workbook_path, source_path)
    monkeypatch.setattr(importer, "load_workbook", lambda *_args, **_kwargs: workbook)

    imported = import_review_workbook(
        source_path=source_path,
        workbook_path=workbook_path,
        output_path=output_path,
        snapshot_id="a" * 64,
    )

    assert imported["split"] == "development"
    case = imported["cases"][0]
    assert case["snapshot_id"] == "a" * 64
    assert case["adjudication"] == {
        "status": "adjudicated",
        "method": "manual_independent",
        "reviewer": "reviewer-1",
        "evidence": [
            "case conclusion: 独立结论-actual",
            "review-log-1",
            "case rationale: case rationale-actual",
            "turn 1: route-review-1",
            "turn 1 task: 直接规划步行路线",
            "turn 1 rationale: 路径判据已复核",
        ],
    }
    assert case["expected_turns"][0] == {
        "status": "completed", "kind": "route", "feasibility": "feasible",
        "start_poi_id": "poi-a", "end_poi_id": "poi-b", "via_poi_ids": [],
        "travel_mode": "walk", "strategy": "shortest", "hard_constraints": {},
        "required_outputs": ["route"], "candidate_poi_ids": [],
        "clarification_slot": None, "allowed_unknowns": [],
        "intent_label": "route_planning",
        "expected_tools": [["resolve_poi", "plan_route"]],
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == imported
    assert json.loads(source_path.read_text(encoding="utf-8"))["cases"][0]["adjudication"]["status"] == "draft"


def test_import_rejects_unreviewed_turn_without_creating_output(tmp_path, monkeypatch):
    import_review_workbook = _importer()
    from experiments import import_scenario_adjudication as importer
    source_path = tmp_path / "scenarios_dev_v1.json"
    source_path.write_text(json.dumps(_source_dataset()), encoding="utf-8")
    workbook_path = tmp_path / "review.xlsx"
    output_path = tmp_path / "adjudicated.json"
    workbook = _write_review_workbook(workbook_path, source_path, complete=False)
    monkeypatch.setattr(importer, "load_workbook", lambda *_args, **_kwargs: workbook)

    try:
        import_review_workbook(
            source_path=source_path,
            workbook_path=workbook_path,
            output_path=output_path,
            snapshot_id="a" * 64,
        )
    except ValueError as exc:
        assert "unreviewed" in str(exc)
    else:
        raise AssertionError("unreviewed turn must be rejected")

    assert not output_path.exists()


def test_import_rejects_source_changed_after_workbook_creation(tmp_path, monkeypatch):
    import_review_workbook = _importer()
    from experiments import import_scenario_adjudication as importer
    source_path = tmp_path / "scenarios_dev_v1.json"
    source_path.write_text(json.dumps(_source_dataset()), encoding="utf-8")
    workbook_path = tmp_path / "review.xlsx"
    output_path = tmp_path / "adjudicated.json"
    workbook = _write_review_workbook(workbook_path, source_path)
    monkeypatch.setattr(importer, "load_workbook", lambda *_args, **_kwargs: workbook)
    source_path.write_text(source_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    try:
        import_review_workbook(
            source_path=source_path,
            workbook_path=workbook_path,
            output_path=output_path,
            snapshot_id="a" * 64,
        )
    except ValueError as exc:
        assert "changed after" in str(exc)
    else:
        raise AssertionError("stale workbook must be rejected")

    assert not output_path.exists()


def test_import_never_overwrites_an_existing_output(tmp_path, monkeypatch):
    import_review_workbook = _importer()
    from experiments import import_scenario_adjudication as importer
    source_path = tmp_path / "scenarios_dev_v1.json"
    source_path.write_text(json.dumps(_source_dataset()), encoding="utf-8")
    workbook_path = tmp_path / "review.xlsx"
    output_path = tmp_path / "adjudicated.json"
    workbook = _write_review_workbook(workbook_path, source_path)
    monkeypatch.setattr(importer, "load_workbook", lambda *_args, **_kwargs: workbook)
    output_path.write_text("keep", encoding="utf-8")

    try:
        import_review_workbook(
            source_path=source_path,
            workbook_path=workbook_path,
            output_path=output_path,
            snapshot_id="a" * 64,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output must not be overwritten")

    assert output_path.read_text(encoding="utf-8") == "keep"


def test_import_rejects_invalid_tool_sequences_and_intent_labels(tmp_path, monkeypatch):
    import_review_workbook = _importer()
    from experiments import import_scenario_adjudication as importer

    source_path = tmp_path / "scenarios_dev_v1.json"
    source_path.write_text(json.dumps(_source_dataset()), encoding="utf-8")
    workbook_path = tmp_path / "review.xlsx"
    output_path = tmp_path / "adjudicated.json"
    workbook = _write_review_workbook(workbook_path, source_path, expected_tools='["plan_route"]')
    monkeypatch.setattr(importer, "load_workbook", lambda *_args, **_kwargs: workbook)
    try:
        import_review_workbook(source_path=source_path, workbook_path=workbook_path,
                               output_path=output_path, snapshot_id="a" * 64)
    except ValueError as exc:
        assert "expected_tools" in str(exc) or "tool" in str(exc)
    else:
        raise AssertionError("one-dimensional tool sequence must be rejected")
    assert not output_path.exists()

    workbook = _write_review_workbook(workbook_path, source_path, intent="made_up")
    monkeypatch.setattr(importer, "load_workbook", lambda *_args, **_kwargs: workbook)
    try:
        import_review_workbook(source_path=source_path, workbook_path=workbook_path,
                               output_path=output_path, snapshot_id="a" * 64)
    except ValueError as exc:
        assert "intent_label" in str(exc)
    else:
        raise AssertionError("unknown intent label must be rejected")
