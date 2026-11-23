import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from spatial import road_clues as clues
from spatial import road_conditions as rc


@pytest.fixture(autouse=True)
def clue_store(tmp_path, monkeypatch):
    monkeypatch.setattr(clues, "_CLUES_FILE", str(tmp_path / "clues.json"))
    conditions = tmp_path / "road_conditions.json"
    conditions.write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    clues.reset_collect_attempts()
    yield


def sample(**overrides):
    data = {"kind": "ugc_report", "summary": "施工", "valid_from": 4102444800,
            "valid_to": 4102444900, "draft": {"event_type": "construction"},
            "edge_candidates": [{"u": 1, "v": 2, "key": 0, "road_name": "自强大道",
                                 "distance_m": 5}]}
    data.update(overrides)
    return data


def edge_payload(u=1, v=2, key=0):
    return {
        "u": u, "v": v, "key": key, "road_name": "自强大道", "dist_m": 2.0,
        "snap_lng_gcj": 114.36, "snap_lat_gcj": 30.53,
        "geometry_gcj": [[114.36, 30.53], [114.361, 30.53]],
        "edges": [[u, v, key]], "chain_length_m": 80.0,
    }


def reviewing(**overrides):
    values = {"valid_from": time.time() - 60, "valid_to": time.time() + 7200}
    values.update(overrides)
    row = clues.create_clue(sample(**values))
    return clues.update_clue(row["clue_id"], {"status": "reviewing"})


def test_status_transitions_require_review_and_rejection_reason():
    clue = clues.create_clue(sample())
    with pytest.raises(ValueError):
        clues.update_clue(clue["clue_id"], {"status": "adopted"})
    with pytest.raises(ValueError):
        clues.update_clue(clue["clue_id"], {"status": "rejected"})
    updated = clues.update_clue(clue["clue_id"], {"status": "reviewing"})
    assert updated["status"] == "reviewing"
    assert clues.update_clue(clue["clue_id"], {"status": "rejected", "review_note": "现场不符"})["status"] == "rejected"
    with pytest.raises(ValueError):
        clues.update_clue(clue["clue_id"], {"status": "pending"})


def test_invalid_times_duplicate_and_corrupt_file():
    with pytest.raises(ValueError):
        clues.create_clue(sample(valid_from=200, valid_to=100))
    first = clues.create_clue(sample())
    second = clues.create_clue(sample(valid_from=4102444820, valid_to=4102444890))
    assert second["duplicate_of"] == first["clue_id"]
    with open(clues._CLUES_FILE, "w", encoding="utf-8") as f:
        f.write("{")
    with pytest.raises(clues.ClueStoreUnavailable):
        clues.create_clue(sample())
    with open(clues._CLUES_FILE, encoding="utf-8") as f:
        assert f.read() == "{"


def test_time_must_carry_timezone_and_summary_is_bounded():
    with pytest.raises(ValueError):
        clues.create_clue(sample(valid_from="2026-01-01T00:00:00",
                                 valid_to="2026-01-01T01:00:00"))
    row = clues.create_clue(sample(valid_from="2026-01-01T00:00:00+08:00",
                                   valid_to="2026-01-01T01:00:00+08:00"))
    assert row["valid_from"].endswith("Z")
    with pytest.raises(ValueError):
        clues.create_clue(sample(summary="长" * 61))


def test_duplicate_is_not_guessed_without_road_type_or_window():
    clues.create_clue(sample())
    no_window = clues.create_clue(sample(valid_from=None, valid_to=None))
    assert no_window["duplicate_of"] is None
    different_type = clues.create_clue(sample(draft={"event_type": "flooding"}))
    assert different_type["duplicate_of"] is None
    different_road = clues.create_clue(sample(edge_candidates=[{
        "u": 7, "v": 8, "key": 0, "road_name": "北环路", "distance_m": 4,
    }]))
    assert different_road["duplicate_of"] is None


def test_expiry_is_persisted_on_read():
    clue = clues.create_clue(sample(valid_from=time.time() - 20, valid_to=time.time() - 10))
    assert clues.get_clue(clue["clue_id"])["status"] == "expired"
    with open(clues._CLUES_FILE, encoding="utf-8") as f:
        assert json.load(f)[0]["status"] == "expired"


def test_web_collector_uses_existing_search_and_fetch(monkeypatch):
    from agents import web_evidence

    calls = []

    def execute(name, args, sources, **kwargs):
        calls.append((name, args))
        if name == "search_web":
            sources["S1"] = {"url": "https://example.org/a"}
            return {"results": [{"source_id": "S1", "url": "https://example.org/a"}]}
        return {"title": "施工通知", "content": "正文"}

    monkeypatch.setattr(web_evidence, "execute", execute)
    results = clues.registry["web"].fetch(query="校内施工")
    assert results[0]["raw_text"] == "正文"
    assert results[0]["search_query"] == "校内施工"
    assert [call[0] for call in calls] == ["search_web", "fetch_web"]


def test_web_collector_does_not_hide_provider_failure(monkeypatch):
    from agents import web_evidence

    monkeypatch.setattr(web_evidence, "execute", lambda *args, **kwargs: {"error": "unavailable"})
    with pytest.raises(RuntimeError):
        clues.registry["web"].fetch(query="施工")


def test_web_direct_url_goes_through_source_validation_and_fetch(monkeypatch):
    from agents import web_evidence

    calls = []
    sources_seen = []

    def execute(name, args, sources, **kwargs):
        calls.append(name)
        sources_seen.append(dict(sources))
        if name == "fetch_web":
            return {"title": "道路通知", "content": "通知正文"}
        raise AssertionError("direct URL must not search first")

    monkeypatch.setattr(web_evidence, "execute", execute)
    results = clues.registry["web"].fetch(url="https://example.org/notice")
    assert calls == ["fetch_web"]
    assert results[0]["source_url"] == "https://example.org/notice"
    only_source = next(iter(sources_seen[0].values()))
    assert only_source["url"] == "https://example.org/notice"


def test_disabled_sources_cannot_create_clues():
    assert clues.registry["weather_alert"].enabled is False
    with pytest.raises(ValueError):
        clues.create_clue({
            "kind": "weather_alert", "summary": "暴雨", "draft": {},
            "edge_candidates": [],
        })


def test_official_notice_low_confidence_requires_human():
    candidates = clues.registry["official_notice"].fetch(
        text="学校通知：自强大道因施工封闭，请绕行。",
        extractor=lambda _text: {
            "summary": "自强大道施工封闭", "confidence": 0.4,
            "draft": {"event_type": "construction", "road_hint": "自强大道"},
        },
    )
    row = clues.create_clue(candidates[0])
    assert row["status"] == "reviewing"
    assert row["need_human"] is True


def test_adoption_merges_same_road_and_marks_all_clues():
    first = reviewing()
    second = reviewing(valid_from=time.time() - 30, valid_to=time.time() + 7000)
    condition, rows, already = clues.adopt_clues(
        first["clue_id"], [second["clue_id"]], edge=edge_payload(),
        cond_type="construction", name="合并施工", start_time=time.time() + 10,
        end_time=time.time() + 1800, action="closure", affected_modes=["walk"],
        review_note="核对通知与现场照片，确认同一路段同一时段施工。",
    )
    assert already is False
    assert condition["source"]["kind"] == "clue"
    assert set(condition["source"]["clue_ids"]) == {first["clue_id"], second["clue_id"]}
    assert all(row["status"] == "adopted" for row in rows)
    assert all(row["condition_ids"] == [condition["id"]] for row in rows)


def test_adoption_rejects_unreviewed_clue_and_foreign_edge():
    pending = clues.create_clue(sample(valid_from=time.time() - 60,
                                       valid_to=time.time() + 7200))
    with pytest.raises(ValueError):
        clues.adopt_clues(
            pending["clue_id"], [], edge=edge_payload(), cond_type="construction",
            name="未复核", end_time=time.time() + 1800, action="closure",
            affected_modes=["walk"], review_note="尚未复核就尝试采纳的测试。",
        )
    target = reviewing()
    with pytest.raises(ValueError):
        clues.adopt_clues(
            target["clue_id"], [], edge=edge_payload(u=7, v=8), cond_type="construction",
            name="错误道路", end_time=time.time() + 1800, action="closure",
            affected_modes=["walk"], review_note="核对后选择不在候选中的道路测试。",
        )


def test_adoption_retry_reuses_event_after_write_before_clue_writback(monkeypatch):
    clue = reviewing()
    token = "crash-recovery-token"
    rows = clues._load()
    for row in rows:
        if row["clue_id"] == clue["clue_id"]:
            row["adoption"] = {"token": token, "started_at": time.time()}
    clues._save(rows)
    condition = rc.add_condition(
        cond_type="construction", name="已写入事件", edge=edge_payload(),
        start_time=time.time() + 10, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
        source={"kind": "clue", "clue_ids": [clue["clue_id"]],
                "primary_clue_id": clue["clue_id"], "adoption_token": token},
    )
    reused, adopted, already = clues.adopt_clues(
        clue["clue_id"], [], edge=edge_payload(), cond_type="construction",
        name="重试", start_time=time.time() + 10, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
        review_note="模拟事件已写入但线索尚未回写后的重试。",
    )
    assert reused["id"] == condition["id"]
    assert already is True
    assert adopted[0]["status"] == "adopted"
    assert len(rc.list_conditions(include_inactive=True, strict=True)) == 1


def test_adopted_replay_rejects_clues_linked_to_several_events():
    first = reviewing()
    second = reviewing(valid_from=time.time() - 30, valid_to=time.time() + 7000)
    condition, _, _ = clues.adopt_clues(
        first["clue_id"], [second["clue_id"]], edge=edge_payload(),
        cond_type="construction", name="合并施工", start_time=time.time() + 10,
        end_time=time.time() + 1800, action="closure", affected_modes=["walk"],
        review_note="核对通知与现场照片，确认同一路段同一时段施工。",
    )
    extra = rc.add_condition(
        cond_type="accident", name="追加事件", edge=edge_payload(),
        start_time=time.time() + 10, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
    )
    stored = clues._load()
    for row in stored:
        row["condition_ids"] = [condition["id"], extra["id"]]
    clues._save(stored)
    with pytest.raises(ValueError, match="关联事件不一致"):
        clues.adopt_clues(
            first["clue_id"], [second["clue_id"]], edge=edge_payload(),
            cond_type="construction", name="重试", start_time=time.time() + 10,
            end_time=time.time() + 1800, action="closure", affected_modes=["walk"],
            review_note="两个事件关联时不能猜哪一个才是本次采纳目标。",
        )


def test_deleted_event_restores_clue_and_remaining_events_keep_adoption():
    clue = reviewing()
    first, _, _ = clues.adopt_clues(
        clue["clue_id"], [], edge=edge_payload(), cond_type="construction",
        name="第一个事件", start_time=time.time() + 10, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
        review_note="核对后采纳为第一个路况事件。",
    )
    second = rc.add_condition(
        cond_type="accident", name="第二个事件", edge=edge_payload(),
        start_time=time.time() + 10, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
    )
    rows = clues._load()
    rows[0]["condition_ids"].append(second["id"])
    clues._save(rows)
    clues.detach_condition(first["id"])
    assert clues.get_clue(clue["clue_id"])["status"] == "adopted"
    clues.detach_condition(second["id"])
    restored = clues.get_clue(clue["clue_id"])
    assert restored["status"] == "reviewing"
    assert restored["condition_ids"] == []


def test_reconcile_restores_adopted_clue_when_event_disappears():
    clue = reviewing()
    condition, _, _ = clues.adopt_clues(
        clue["clue_id"], [], edge=edge_payload(), cond_type="construction",
        name="将被删除", start_time=time.time() + 10, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"], review_note="核对后采纳，随后删除事件。",
    )
    rc.remove_condition(condition["id"])
    restored = clues.get_clue(clue["clue_id"])
    assert restored["status"] == "reviewing"
    assert restored["condition_ids"] == []


def test_near_query_excludes_unbound_candidates():
    bound = clues.create_clue(sample(edge_candidates=[{
        "u": 1, "v": 2, "key": 0, "road_name": "自强大道", "distance_m": 2,
        "geometry_gcj": [[114.3600, 30.5300], [114.3604, 30.5300]],
    }]))
    unbound = clues.create_clue(sample(edge_candidates=[{
        "u": 3, "v": 4, "key": 0, "road_name": "无位置道路", "distance_m": 3,
    }]))
    matched = clues.clues_near(clues.list_clues(), 114.3602, 30.5300, 50)
    ids = {row["clue_id"] for row in matched}
    assert bound["clue_id"] in ids
    assert unbound["clue_id"] not in ids


def test_concurrent_creation_keeps_unique_rows():
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _index: clues.create_clue(sample()), range(24)))
    assert len({row["clue_id"] for row in rows}) == 24
    assert len(clues.list_clues()) == 24


def test_cross_process_rate_limit_allows_only_one_claim(tmp_path):
    rate_file = tmp_path / "rate.json"
    code = (
        "import os, sys;"
        f"os.environ['ROAD_CLUE_RATE_FILE']={str(rate_file)!r};"
        "from spatial import road_clues as c;"
        "allowed, retry = c.claim_collect('桂园');"
        "print('1' if allowed else '0')"
    )
    processes = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    outputs = [process.communicate(timeout=60)[0].strip() for process in processes]
    assert sorted(outputs) == ["0", "1"]
