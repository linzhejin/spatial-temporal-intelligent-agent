import json
import subprocess
import sys
from pathlib import Path

import pytest

from vision.campus_water_data import (
    build_campus_flight_plan,
    extract_campus_annotation_pack,
)


def _flight_manifest(root):
    flights = []
    split_groups = {"train": [], "validation": [], "test": []}
    for flight_id, split in (
        ("flight-01", "train"),
        ("flight-02", "validation"),
        ("flight-03", "test"),
    ):
        video = f"raw/{flight_id}.mp4"
        path = root / video
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video placeholder {flight_id}".encode("ascii"))
        split_groups[split].append(flight_id)
        sidecar = f"raw/{flight_id}.srt"
        (root / sidecar).write_text("1\n00:00:00,000 --> 00:00:01,000\nflight\n", encoding="utf-8")
        flights.append({
            "flight_id": flight_id,
            "split": split,
            "site_id": f"site-{flight_id[-2:]}",
            "location": f"武大试点 {flight_id[-2:]}",
            "captured_at": f"2026-10-{10 + int(flight_id[-2:]):02d}T09:30:00+08:00",
            "video": video,
            "sidecars": [sidecar],
        })
    return {
        "schema_version": 1,
        "task": "campus_visible_water_frame_extraction",
        "dataset": "WHU UAV pilot",
        "split_unit": "flight_id",
        "split_groups": split_groups,
        "flights": flights,
    }


def test_flight_manifest_requires_exact_disjoint_flight_split_groups(tmp_path):
    manifest = _flight_manifest(tmp_path)

    plan = build_campus_flight_plan(manifest, tmp_path)

    assert plan["split_groups"] == {
        "train": ["flight-01"],
        "validation": ["flight-02"],
        "test": ["flight-03"],
    }
    assert len(plan["flights"]) == 3
    assert plan["flights"][0]["location"] == "武大试点 01"
    assert len(plan["flights"][0]["source_sha256"]) == 64
    assert plan["flights"][0]["source_bytes"] == len(b"video placeholder flight-01")
    assert len(plan["flights"][0]["sidecars"][0]["sha256"]) == 64

    leaked = json.loads(json.dumps(manifest))
    leaked["split_groups"]["validation"].append("flight-01")
    with pytest.raises(ValueError, match="flight groups must not overlap"):
        build_campus_flight_plan(leaked, tmp_path)


def test_flight_manifest_rejects_path_escape_duplicate_video_and_unknown_flight(tmp_path):
    manifest = _flight_manifest(tmp_path)
    manifest["flights"][0]["video"] = "../outside.mp4"
    with pytest.raises(ValueError, match="must stay inside the dataset"):
        build_campus_flight_plan(manifest, tmp_path)

    manifest = _flight_manifest(tmp_path)
    manifest["flights"][1]["video"] = manifest["flights"][0]["video"]
    with pytest.raises(ValueError, match="video file is assigned to multiple flights"):
        build_campus_flight_plan(manifest, tmp_path)

    manifest = _flight_manifest(tmp_path)
    first = tmp_path / manifest["flights"][0]["video"]
    second = tmp_path / manifest["flights"][1]["video"]
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="identical source video"):
        build_campus_flight_plan(manifest, tmp_path)

    manifest = _flight_manifest(tmp_path)
    manifest["flights"].pop()
    with pytest.raises(ValueError, match="must match declared flight groups"):
        build_campus_flight_plan(manifest, tmp_path)


def test_flight_manifest_requires_timezone_and_keeps_each_site_in_one_split(tmp_path):
    manifest = _flight_manifest(tmp_path)
    manifest["flights"][0]["captured_at"] = "2026-10-11T09:30:00"
    with pytest.raises(ValueError, match="timezone"):
        build_campus_flight_plan(manifest, tmp_path)

    manifest = _flight_manifest(tmp_path)
    manifest["flights"][1]["site_id"] = manifest["flights"][0]["site_id"]
    with pytest.raises(ValueError, match="site must not cross splits"):
        build_campus_flight_plan(manifest, tmp_path)

    manifest = _flight_manifest(tmp_path)
    manifest["flights"][2]["location"] = ""
    with pytest.raises(ValueError, match="location is required"):
        build_campus_flight_plan(manifest, tmp_path)


class _FakeCapture:
    def __init__(self, frame_count=5, fps=2.0):
        self._frame_count = frame_count
        self._fps = fps
        self._index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            _FakeCV2.CAP_PROP_FPS: self._fps,
            _FakeCV2.CAP_PROP_FRAME_COUNT: self._frame_count,
            _FakeCV2.CAP_PROP_FRAME_WIDTH: 32,
            _FakeCV2.CAP_PROP_FRAME_HEIGHT: 24,
        }[prop]

    def read(self):
        if self._index >= self._frame_count:
            return False, None
        value = self._index
        self._index += 1
        return True, value

    def release(self):
        self.released = True


class _FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    IMWRITE_JPEG_QUALITY = 5

    def __init__(self):
        self.captures = []

    def VideoCapture(self, _path):
        capture = _FakeCapture()
        self.captures.append(capture)
        return capture

    @staticmethod
    def imwrite(path, frame, _params):
        path = __import__("pathlib").Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(frame).encode("ascii"))
        return True


class _TruncatedCapture(_FakeCapture):
    def get(self, prop):
        if prop == _FakeCV2.CAP_PROP_FRAME_COUNT:
            return self._frame_count + 4
        return super().get(prop)


class _TruncatedCV2(_FakeCV2):
    def VideoCapture(self, _path):
        capture = _TruncatedCapture()
        self.captures.append(capture)
        return capture


def test_extraction_builds_frame_worklist_and_train_test_templates(tmp_path):
    manifest = _flight_manifest(tmp_path / "source")
    plan = build_campus_flight_plan(manifest, tmp_path / "source")
    cv2 = _FakeCV2()

    result = extract_campus_annotation_pack(
        plan,
        tmp_path / "annotation-pack",
        sample_fps=1.0,
        jpeg_quality=95,
        cv2_api=cv2,
    )

    assert result["sample_counts"] == {"train": 3, "validation": 3, "test": 3}
    assert len(cv2.captures) == 3
    assert all(capture.released for capture in cv2.captures)
    worklist = json.loads(
        (tmp_path / "annotation-pack" / "annotation-worklist.json").read_text(encoding="utf-8")
    )
    assert [sample["source_frame_index"] for sample in worklist["samples"][:3]] == [0, 2, 4]
    assert worklist["samples"][0]["timestamp_seconds"] == 0.0
    assert worklist["samples"][1]["timestamp_seconds"] == 1.0
    assert worklist["samples"][0]["road_surface_polygon"] is None
    expected_location = "武大试点 01"
    assert worklist["flights"][0]["location"] == expected_location
    assert worklist["flights"][0]["captured_at"].endswith("+08:00")
    assert len(worklist["flights"][0]["source_sha256"]) == 64
    assert (tmp_path / "annotation-pack" / "frames" / "train" / "flight-01"
            / "frame_00000000.jpg").is_file()

    training = json.loads(
        (tmp_path / "annotation-pack" / "training-template.json").read_text(encoding="utf-8")
    )
    testing = json.loads(
        (tmp_path / "annotation-pack" / "test-template.json").read_text(encoding="utf-8")
    )
    detection = json.loads(
        (tmp_path / "annotation-pack" / "detection-test-template.json").read_text(
            encoding="utf-8"
        )
    )
    events = json.loads(
        (tmp_path / "annotation-pack" / "event-test-template.json").read_text(
            encoding="utf-8"
        )
    )
    assert {sample["split"] for sample in training["samples"]} == {"train", "validation"}
    assert {flight["split"] for flight in training["flights"]} == {"train", "validation"}
    assert all(len(flight["source_sha256"]) == 64 for flight in training["flights"])
    assert len(testing["samples"]) == 3
    assert {sample["group_id"] for sample in testing["samples"]} == {"flight-03"}
    assert {flight["flight_id"] for flight in testing["flights"]} == {"flight-03"}
    assert detection["task"] == "campus_object_detection"
    assert detection["label_source"] == "manual"
    assert len(detection["samples"]) == 3
    assert detection["samples"][0]["width"] == 32
    assert detection["samples"][0]["height"] == 24
    assert detection["samples"][0]["objects"] == []
    assert events["task"] == "campus_event_detection"
    assert events["event_kinds"] == ["congestion", "crowding", "accident", "flooding"]
    assert events["annotation_pending"] is True
    assert events["samples"] == [{
        "sample_id": "flight-03",
        "video": "raw/flight-03.mp4",
        "group_id": "flight-03",
        "site_id": "site-03",
        "location": "武大试点 03",
        "captured_at": "2026-10-13T09:30:00+08:00",
        "source_sha256": worklist["flights"][2]["source_sha256"],
        "camera_stabilized": None,
        "event_kinds": [],
        "observation_regions": [],
        "events": [],
    }]
    instructions = (tmp_path / "annotation-pack" / "ANNOTATION.md").read_text(encoding="utf-8")
    assert '\n"' not in instructions
    assert instructions.startswith("# 校园道路可见水体人工标注\n\n1.")


def test_extraction_refuses_to_overwrite_existing_output(tmp_path):
    source = tmp_path / "source"
    plan = build_campus_flight_plan(_flight_manifest(source), source)
    output = tmp_path / "annotation-pack"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        extract_campus_annotation_pack(plan, output, cv2_api=_FakeCV2())


def test_extraction_rejects_video_that_ends_before_reported_frame_count(tmp_path):
    source = tmp_path / "source"
    plan = build_campus_flight_plan(_flight_manifest(source), source)
    output = tmp_path / "annotation-pack"

    with pytest.raises(ValueError, match="ended before reported frame count"):
        extract_campus_annotation_pack(plan, output, cv2_api=_TruncatedCV2())

    assert not output.exists()
    assert (tmp_path / ".annotation-pack.partial").is_dir()


def test_prepare_cli_dry_run_validates_manifest_without_writing_output(tmp_path):
    source = tmp_path / "source"
    manifest = _flight_manifest(source)
    manifest_path = source / "campus-flights.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "annotation-pack"
    script = Path(__file__).parents[1] / "scripts" / "vision" / "prepare_campus_water_data.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root", str(source),
            "--manifest", str(manifest_path),
            "--output-dir", str(output),
            "--dry-run",
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["flights_by_split"] == {"train": 1, "validation": 1, "test": 1}
    assert payload["writes_files"] is False
    assert payload["source_bytes"] == sum(
        len(f"video placeholder flight-{number:02d}".encode("ascii"))
        for number in range(1, 4)
    )
    assert not output.exists()
