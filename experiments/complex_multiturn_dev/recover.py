"""Reproject immutable run results from the isolated test DB after collector fixes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import psycopg
from psycopg.rows import dict_row

from experiments.complex_multiturn_dev.contracts import assess_turn, read_dataset
from experiments.complex_multiturn_dev.observation import project_observation
from experiments.scenario_live_adapter import build_poi_lookup


ROOT = Path(__file__).resolve().parents[2]


def recover(run_dir: Path, dataset: Path):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != hashlib.sha256(Path(dataset).read_bytes()).hexdigest():
        raise ValueError("dataset differs from the one used by the run")
    dbname = manifest.get("database_name", "")
    if not re.fullmatch(r"luojia_multiturn_test_[a-f0-9]{16}", dbname):
        raise ValueError("run does not identify a dedicated local test database")
    rows = [json.loads(line) for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = {c["id"]: c for c in read_dataset(dataset)["cases"]}
    lookup = build_poi_lookup(json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"])
    output = run_dir / "restored_observations.jsonl"
    if output.exists():
        raise FileExistsError(output)
    last = {}
    counts = Counter()
    with psycopg.connect(host="127.0.0.1", port=55439, dbname=dbname, user="postgres",
                         password="", row_factory=dict_row, connect_timeout=5) as conn:
        identity = conn.execute("SELECT current_database() AS name, current_setting('data_directory') AS data_dir").fetchone()
        if identity["name"] != dbname or Path(identity["data_dir"]).resolve() != Path(
                r"C:\Users\HUAWEI\.codex\test-postgres-data").resolve():
            raise ValueError("unexpected database identity")
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            for old in rows:
                if not old.get("run_id"):
                    stream.write(json.dumps(old, ensure_ascii=False) + "\n")
                    continue
                raw = conn.execute("SELECT run_id,status,task_id,task_revision,result FROM run WHERE run_id=%s",
                                   (old["run_id"],)).fetchone()
                if not raw:
                    raise ValueError("saved run was not found")
                case_id = old["case"]
                prior = last.get(case_id, {})
                bound_id = raw["task_id"] or prior.get("task_id")
                revision = old.get("task_revision") if raw["task_id"] else prior.get("revision")
                task = None
                if bound_id and type(revision) is int:
                    row = conn.execute("SELECT revision,state FROM task_history WHERE task_id=%s AND revision<=%s "
                                       "ORDER BY revision DESC LIMIT 1", (bound_id, revision)).fetchone()
                    if row:
                        task = {"task_id": bound_id, "revision": row["revision"], "state": row["state"]}
                previous_candidates = prior.get("candidates") or []
                turn = cases[case_id]["turns"][old["turn"] - 1]
                selected = previous_candidates[0] if turn.get("action") == "select_first_candidate" and previous_candidates else None
                observed = project_observation(dict(raw), task, lookup, selected_first=selected)
                proof = old.get("event_proof")
                consumed = old.get("consumed_event_ids") or []
                if proof and consumed and all(item == proof["active_ids"] for item in consumed):
                    observed["event_state"] = proof["stage"]
                score = assess_turn(turn, observed)
                if score["verdict"] == "pass":
                    score["verdict"] = "unknown"
                    score["unknowns"].append("delivery_independent_verification_pending")
                record = {**old, "result": raw["result"], "observed": observed, "score": score,
                          "observed_task_id": bound_id, "observation_source": "isolated_postgresql_original"}
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                last[case_id] = {"task_id": bound_id, "revision": revision,
                                 "candidates": observed.get("candidate_ids") or []}
                counts[score["verdict"]] += 1
    return {"output": str(output), "counts": dict(counts), "source": "original isolated database"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.run_dir, args.dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
