"""Pin an official BFCL V4 multi-turn subset without repackaging its gold.

The output is a selection manifest, not a scored benchmark. Actual execution
must use the upstream BFCL evaluator at the same source commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request


COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
REPO = "https://github.com/ShishirPatil/gorilla"
RAW_ROOT = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/" + COMMIT
            + "/berkeley-function-call-leaderboard/bfcl_eval/data/")
CATEGORIES = ("multi_turn_base", "multi_turn_miss_func",
              "multi_turn_miss_param", "multi_turn_long_context")
SEED = 20261023


def parse_jsonl(data: bytes, category: str) -> list[str]:
    ids = []
    for line in data.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        case_id = item.get("id")
        if (not isinstance(case_id, str) or not case_id.startswith(category + "_")
                or not isinstance(item.get("question"), list)):
            raise ValueError(f"invalid official {category} case")
        ids.append(case_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate official IDs in {category}")
    return ids


def select_subset(raw_by_category: dict[str, bytes], *, per_category: int = 10) -> dict:
    if set(raw_by_category) != set(CATEGORIES) or per_category < 1:
        raise ValueError("four official categories and a positive sample size are required")
    selected = {}
    for category in CATEGORIES:
        content = raw_by_category[category]
        ids = parse_jsonl(content, category)
        if len(ids) < per_category:
            raise ValueError(f"insufficient official {category} cases")
        rng = random.Random(f"{SEED}:{category}")
        selected[category] = {
            "source_url": RAW_ROOT + f"BFCL_v4_{category}.json",
            "sha256": hashlib.sha256(content).hexdigest(),
            "available_count": len(ids),
            "selected_ids": sorted(rng.sample(ids, per_category)),
        }
    return {
        "format": "bfcl-v4-official-selection-v1", "source_repo": REPO,
        "source_commit": COMMIT, "seed": SEED,
        "scope": "O1 separate official tool-use benchmark; no campus T1-T5 score mixing",
        "scoring": "Use upstream BFCL evaluation code and gold at source_commit; IDs alone are not scores.",
        "categories": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/datasets/bfcl_v4_official_selection.json"))
    parser.add_argument("--cache", type=Path, default=Path("output/experiments/official/bfcl_v4"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"selection manifest already exists: {args.output}")
    args.cache.mkdir(parents=True, exist_ok=True)
    raw = {}
    for category in CATEGORIES:
        path = args.cache / f"BFCL_v4_{category}.json"
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(
                RAW_ROOT + path.name, headers={"User-Agent": "WHU-Walker-Research/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                content = response.read()
            path.write_bytes(content)
        raw[category] = content
    manifest = select_subset(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": sum(
        len(item["selected_ids"]) for item in manifest["categories"].values())},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
