"""Recover a worker result missed by an interrupted single-case browser bridge."""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest.get("case_ids") or []) != 1:
        raise ValueError("only a single-case run can be unambiguously recovered")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != manifest.get("dataset_sha256"):
        raise ValueError("dataset differs from the run")
    dbname = manifest.get("database_name", "")
    if not re.fullmatch(r"luojia_multiturn_test_[a-f0-9]{16}", dbname):
        raise ValueError("not a dedicated test database")
    case = next(row for row in read_dataset(args.dataset)["cases"] if row["id"] == manifest["case_ids"][0])
    original = args.run_dir / "turns.jsonl"
    existing = [json.loads(line) for line in original.read_text(encoding="utf-8").splitlines()] if original.exists() else []
    recorded = {row.get("run_id") for row in existing}
    lookup = build_poi_lookup(json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"])
    recovered = []
    with psycopg.connect(host="127.0.0.1", port=55439, dbname=dbname, user="postgres",
                         password="", row_factory=dict_row, connect_timeout=5) as conn:
        identity = conn.execute("SELECT current_database() AS name, current_setting('data_directory') AS data_dir").fetchone()
        if identity["name"] != dbname or Path(identity["data_dir"]).resolve() != Path(
                r"C:\Users\HUAWEI\.codex\test-postgres-data").resolve():
            raise ValueError("unexpected database identity")
        rows = conn.execute("SELECT r.run_id,r.status,r.task_id,r.task_revision,r.result,m.content AS user_input "
                            "FROM run r JOIN conversation_message m ON m.conversation_id=r.conversation_id "
                            "AND m.seq=r.input_message_seq ORDER BY r.created_at").fetchall()
        if len(rows) > len(case["turns"]):
            raise ValueError("database contains more runs than expected turns")
        for index, raw in enumerate(rows, 1):
            turn = case["turns"][index - 1]
            if raw["user_input"] != turn["user"]:
                raise ValueError("saved input does not match the question; cannot infer the turn")
            if str(raw["run_id"]) in recorded:
                continue
            task = None
            if raw["task_id"] and type(raw["task_revision"]) is int:
                saved = conn.execute("SELECT revision,state FROM task_history WHERE task_id=%s AND revision<=%s "
                                     "ORDER BY revision DESC LIMIT 1", (raw["task_id"], raw["task_revision"])).fetchone()
                if saved:
                    task = {"state": saved["state"], "revision": saved["revision"]}
            observed = project_observation(dict(raw), task, lookup)
            recovered.append({"case": case["id"], "turn": index, "user": turn["user"],
                              "run_id": str(raw["run_id"]), "task_id": str(raw["task_id"]) if raw["task_id"] else None,
                              "task_revision": raw["task_revision"], "observed": observed,
                              "score": assess_turn(turn, observed), "result": raw["result"],
                              "browser_evidence": None, "observation_source": "isolated_postgresql_uncollected"})
    output = args.run_dir / "recovered_uncollected.jsonl"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in recovered:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"output": str(output), "recovered": len(recovered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
