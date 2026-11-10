"""Supervise one image-analysis process using a single wall-clock budget."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from vision.assisted_image import _merge_assisted_flood_results
from vision.engine import VisionAnalysisCancelled


def _terminate_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=2, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=2)


def run_supervised_assisted_image(request, *, deadline, cancel_check, command=None, temp_root=None):
    """Run the complete image analysis in a child that can be stopped promptly."""
    if cancel_check and cancel_check():
        raise VisionAnalysisCancelled("管理员已取消该影像任务")
    if time.monotonic() >= deadline:
        raise TimeoutError("积水图片分析超过 120 秒时限")

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="whu-assisted-image-", dir=temp_root) as temporary:
        root = Path(temporary)
        environment = {
            key: value for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "WINDIR", "HOME", "TMPDIR", "TMP", "TEMP",
                       "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
            or key.startswith("VISION_")
        }
        environment["PYTHONPATH"] = str(project_root)
        python = request.get("runtime_python") or sys.executable
        args = list(command) if command is not None else [python, "-m", "vision.assisted_image_child"]
        results = {}
        stages = ["traffic", "flood"] if request.get("traffic_required") else ["flood"]
        for stage in stages:
            request_path = root / f"{stage}-request.json"
            response_path = root / f"{stage}-response.json"
            stderr_path = root / f"{stage}-stderr.log"
            payload = dict(request, stage=stage, response_path=str(response_path))
            request_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            process = None
            try:
                if cancel_check and cancel_check():
                    raise VisionAnalysisCancelled("管理员已取消该影像任务")
                if time.monotonic() >= deadline:
                    raise TimeoutError("积水图片分析超过 120 秒时限")
                with stderr_path.open("wb") as stderr:
                    process = subprocess.Popen(
                        [*args, str(request_path)], cwd=str(project_root), env=environment,
                        stdout=subprocess.DEVNULL, stderr=stderr,
                        start_new_session=os.name != "nt",
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                    )
                    while process.poll() is None:
                        if cancel_check and cancel_check():
                            raise VisionAnalysisCancelled("管理员已取消该影像任务")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("积水图片分析超过 120 秒时限")
                        try:
                            process.wait(timeout=min(0.1, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                if cancel_check and cancel_check():
                    raise VisionAnalysisCancelled("管理员已取消该影像任务")
                if time.monotonic() >= deadline:
                    raise TimeoutError("积水图片分析超过 120 秒时限")
                if process.returncode != 0:
                    error = stderr_path.read_text(encoding="utf-8", errors="replace")[-1000:]
                    raise RuntimeError("积水图片分析进程失败：" + (error.strip() or str(process.returncode)))
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("status") != "ok" or not isinstance(response.get("result"), dict):
                    raise RuntimeError("积水图片分析进程未返回有效结果")
                results[stage] = response["result"]
            finally:
                if process is not None and process.poll() is None:
                    _terminate_process_tree(process)
        if time.monotonic() >= deadline:
            raise TimeoutError("积水图片分析超过 120 秒时限")
        if "traffic" in results:
            return _merge_assisted_flood_results(results["traffic"], results["flood"])
        return results["flood"]
