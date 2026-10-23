"""Prepare pinned BFCL V4 O1 IDs or score existing official model results.

Preparation is offline and does not call a model. ``score`` calls the pinned
upstream evaluator only after checking that its result files contain exactly
the registered 40 IDs. The original BFCL prompt and scoring stay upstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.bfcl_official import (
    build_run_ids,
    validate_cached_sources,
    validate_official_scores,
    validate_result_subset,
)


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace existing BFCL artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def verify_upstream(upstream: Path, expected_commit: str) -> Path:
    package_root = upstream / "berkeley-function-call-leaderboard"
    if not (package_root / "bfcl_eval" / "eval_checker" / "eval_runner.py").is_file():
        raise FileNotFoundError("pinned BFCL evaluator is absent")
    observed = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != expected_commit:
        raise ValueError("BFCL evaluator commit differs from selection manifest")
    modified = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if modified:
        raise ValueError("BFCL upstream checkout has tracked modifications")
    return package_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "experiments/datasets/bfcl_v4_official_selection.json"))
    parser.add_argument("--cache", type=Path,
                        default=Path("output/experiments/official/bfcl_v4"))
    parser.add_argument("--upstream", type=Path,
                        default=Path("output/experiments/official/bfcl_upstream"))
    parser.add_argument("--run-root", type=Path,
                        default=Path("output/experiments/official/bfcl_o1"))
    parser.add_argument("--model", default="DeepSeek-V3.2-Exp-FC")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python interpreter with the pinned BFCL package dependencies")
    parser.add_argument("--synthetic-smoke", action="store_true",
                        help="Score synthetic fixtures only; receipt is never a model result")
    args = parser.parse_args(argv)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    source_hashes = validate_cached_sources(manifest, args.cache)
    package_root = verify_upstream(args.upstream, manifest["source_commit"])
    for category, digest in source_hashes.items():
        # Compare Git's pinned blob: Windows may check out text as CRLF while
        # the raw GitHub cache and repository blob use LF bytes.
        relative = ("berkeley-function-call-leaderboard/bfcl_eval/data/"
                    f"BFCL_v4_{category}.json")
        official_blob = subprocess.run(
            ["git", "-C", str(args.upstream), "show", f"HEAD:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(official_blob).hexdigest() != digest:
            raise ValueError(f"BFCL package source differs from pinned cache: {category}")
    root = args.run_root.resolve()
    selected = build_run_ids(manifest)

    if args.action == "prepare":
        _write_new(root / "test_case_ids_to_generate.json", selected)
        report = {
            "status": "prepared_only_no_model_calls",
            "source_commit": manifest["source_commit"],
            "selection_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": source_hashes,
            "selected_count": sum(map(len, selected.values())),
            "model_registry_name": args.model,
            "official_generation_command": [str(args.python), "-m",
                "bfcl_eval.openfunctions_evaluation", "--model", args.model,
                "--run-ids", "--num-threads", "1", "--result-dir", "result"],
            "generation_requires": "BFCL_PROJECT_ROOT, PYTHONPATH, DEEPSEEK_API_KEY; incurs API charges",
            "evaluation": "Run this script with action=score after official generation; scores remain O1 only.",
        }
        _write_new(root / "o1_preparation.json", report)
        print(json.dumps({"status": report["status"], "selected": report["selected_count"],
                          "run_root": str(root)}, ensure_ascii=False))
        return 0

    if not (root / "test_case_ids_to_generate.json").is_file():
        raise FileNotFoundError("run prepare before official scoring")
    if json.loads((root / "test_case_ids_to_generate.json").read_text(encoding="utf-8")) != selected:
        raise ValueError("prepared BFCL IDs differ from frozen selection")
    result_report = validate_result_subset(
        manifest, root / "result", args.model,
        require_generation_metadata=not args.synthetic_smoke,
    )
    if list((root / "score" / args.model).rglob("BFCL_v4_*_score.json")):
        raise FileExistsError("official BFCL scores already exist; use a new run root")
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(root)
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
    # Upstream prints emoji; Windows GBK console encoding raises before scoring.
    env["PYTHONIOENCODING"] = "utf-8"
    # The pinned upstream reads UTF-8 JSONL without an explicit encoding.
    env["PYTHONUTF8"] = "1"
    # The upstream scorer instantiates the model handler even though evaluation
    # only decodes saved responses. Keep credentials out of this offline process.
    if args.model.startswith("DeepSeek-"):
        env["OPENAI_API_KEY"] = "bfcl-offline-scoring-placeholder"
        env["DEEPSEEK_API_KEY"] = "bfcl-offline-scoring-placeholder"
    command = [str(args.python), "-m", "bfcl_eval.eval_checker.eval_runner",
               "--model", args.model, "--test-category", "multi_turn",
               "--result-dir", "result", "--score-dir", "score", "--partial-eval"]
    subprocess.run(command, check=True, cwd=package_root, env=env)
    score_report = validate_official_scores(manifest, root / "score", args.model)
    receipt = {
        "status": ("synthetic_smoke_official_scorer" if args.synthetic_smoke
                   else "official_partial_score"),
        "source_commit": manifest["source_commit"],
        "selection_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_registry_name": args.model,
        "result": result_report, "score": score_report,
        "scope": ("SYNTHETIC FIXTURES; NO MODEL WAS CALLED BY THIS SCRIPT; NOT RESEARCH RESULTS"
                  if args.synthetic_smoke else
                  "BFCL O1 only; not a campus T1-T5 or official full-leaderboard score"),
    }
    _write_new(root / "o1_score_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "selected": result_report["total"],
                      "run_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
