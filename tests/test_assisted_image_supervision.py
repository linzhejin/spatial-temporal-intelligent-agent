import json
import subprocess
import sys
import time

import pytest

from vision import assisted_image_supervisor
from vision.engine import VisionAnalysisCancelled


def _child_command(source):
    return [sys.executable, "-c", source]


def test_assisted_image_deadline_terminates_a_stuck_analysis(monkeypatch, tmp_path):
    processes = []
    real_popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(assisted_image_supervisor.subprocess, "Popen", capture_process)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="120|时限"):
        assisted_image_supervisor.run_supervised_assisted_image(
            {}, deadline=started + 0.5, cancel_check=lambda: False,
            command=_child_command("import time; time.sleep(60)"), temp_root=tmp_path,
        )
    assert time.monotonic() - started < 5
    assert processes and all(process.poll() is not None for process in processes)
    assert not list(tmp_path.iterdir())


def test_assisted_image_cancellation_terminates_a_running_analysis(monkeypatch, tmp_path):
    marker = tmp_path / "started.json"
    processes = []
    real_popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(assisted_image_supervisor.subprocess, "Popen", capture_process)
    source = (
        "import json, pathlib, sys, time; "
        "request=json.loads(pathlib.Path(sys.argv[-1]).read_text(encoding='utf-8')); "
        "pathlib.Path(request['marker']).write_text('started'); time.sleep(60)"
    )
    started = time.monotonic()
    with pytest.raises(VisionAnalysisCancelled):
        assisted_image_supervisor.run_supervised_assisted_image(
            {"marker": str(marker)}, deadline=started + 30,
            cancel_check=marker.exists, command=_child_command(source), temp_root=tmp_path,
        )
    assert time.monotonic() - started < 5
    assert processes and all(process.poll() is not None for process in processes)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["started.json"]


def test_assisted_image_cancellation_terminates_the_process_tree(tmp_path):
    ready = tmp_path / "ready"
    survivor = tmp_path / "survivor"
    grandchild = (
        "import pathlib, sys, time; time.sleep(2); "
        "pathlib.Path(sys.argv[1]).write_text('not terminated')"
    )
    parent = (
        "import json, pathlib, subprocess, sys, time; "
        "request=json.loads(pathlib.Path(sys.argv[-1]).read_text(encoding='utf-8')); "
        "subprocess.Popen([sys.executable,'-c',request['grandchild'],request['survivor']]); "
        "pathlib.Path(request['ready']).write_text('started'); time.sleep(60)"
    )
    with pytest.raises(VisionAnalysisCancelled):
        assisted_image_supervisor.run_supervised_assisted_image(
            {"ready": str(ready), "survivor": str(survivor), "grandchild": grandchild},
            deadline=time.monotonic() + 10, cancel_check=ready.exists,
            command=_child_command(parent), temp_root=tmp_path,
        )
    time.sleep(2.5)
    assert not survivor.exists()


def test_assisted_image_returns_completed_child_result(tmp_path):
    source = (
        "import json, pathlib, sys; "
        "request=json.loads(pathlib.Path(sys.argv[-1]).read_text(encoding='utf-8')); "
        "pathlib.Path(request['response_path']).write_text(json.dumps("
        "{'status':'ok','result':{'candidates':[{'kind':'possible_flooding'}]}}), encoding='utf-8')"
    )
    result = assisted_image_supervisor.run_supervised_assisted_image(
        {}, deadline=time.monotonic() + 5, cancel_check=lambda: False,
        command=_child_command(source), temp_root=tmp_path,
    )
    assert result == {"candidates": [{"kind": "possible_flooding"}]}
    assert not list(tmp_path.iterdir())


def test_assisted_image_releases_traffic_process_before_starting_flood(tmp_path):
    timeline = tmp_path / "stages.jsonl"
    source = (
        "import json, pathlib, sys; "
        "request=json.loads(pathlib.Path(sys.argv[-1]).read_text(encoding='utf-8')); "
        "stage=request['stage']; "
        "f=pathlib.Path(request['timeline']).open('a',encoding='utf-8'); "
        "f.write(json.dumps({'stage':stage})+'\\n'); f.close(); "
        "result={'candidates':[{'kind':stage}], 'metrics':{}, 'safety':{}, 'model':{}}; "
        "pathlib.Path(request['response_path']).write_text(json.dumps("
        "{'status':'ok','result':result}),encoding='utf-8')"
    )
    result = assisted_image_supervisor.run_supervised_assisted_image(
        {"traffic_required": True, "timeline": str(timeline)},
        deadline=time.monotonic() + 5, cancel_check=lambda: False,
        command=_child_command(source), temp_root=tmp_path,
    )
    assert [json.loads(line)["stage"] for line in timeline.read_text().splitlines()] == ["traffic", "flood"]
    assert result["candidates"] == [{"kind": "traffic"}, {"kind": "flood"}]
