"""Export a privacy-limited comparison trace from an actual local run JSON.

Input is an ignored local API-run export or a development trace report. The
header JSON supplies the frozen comparison settings; output is never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.baselines.luojia_run_evidence import project_luojia_run_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--header-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.run_json.read_text(encoding="utf-8"))
    if isinstance(source, dict) and isinstance(source.get("data"), dict):
        source = source["data"]
    if isinstance(source, dict) and "events" not in source and isinstance(source.get("records"), list):
        source = {"events": [
            {"event_type": "agent_tool_trace_started", "payload": {}},
            *({"event_type": record.get("kind"), "payload": record.get("payload")}
              for record in source["records"]),
        ]}
    if not isinstance(source, dict):
        raise ValueError("run JSON must be an object")
    header = json.loads(args.header_json.read_text(encoding="utf-8"))
    records, usage = project_luojia_run_evidence(source, header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(records, target, ensure_ascii=False, indent=2)
        target.write("\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest,
                      "measured_usage": usage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
