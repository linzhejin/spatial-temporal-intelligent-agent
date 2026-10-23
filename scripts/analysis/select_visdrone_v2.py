"""Pin 100 labeled VisDrone validation images across scene-prefix groups."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments/datasets/visdrone_v2_selection.json"
SEED = 20261023


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_selection(archive_path: Path, *, count: int = 100, seed: int = SEED) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    images = sorted(name for name in names
                    if "/images/" in f"/{name}"
                    and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg"})
    annotations = {PurePosixPath(name).stem for name in names
                   if "/annotations/" in f"/{name}"
                   and PurePosixPath(name).suffix.lower() == ".txt"}
    if count < 1 or count > len(images):
        raise ValueError("sample size exceeds available images")
    if any(PurePosixPath(name).stem not in annotations for name in images):
        raise ValueError("validation archive has an image without annotation")
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in images:
        grouped[PurePosixPath(name).stem.split("_")[0]].append(name)
    rng = random.Random(seed)
    groups = sorted(grouped)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(grouped[group])
    selected = []
    while len(selected) < count:
        for group in groups:
            if grouped[group]:
                selected.append(grouped[group].pop())
                if len(selected) == count:
                    break
    return {
        "format": "visdrone-v2-selection", "dataset": "VisDrone2019-DET-val",
        "split": "validation", "archive_sha256": file_sha256(archive_path),
        "seed": seed, "selection_rule": "seeded round-robin by filename scene prefix",
        "scene_prefix_count": len({PurePosixPath(name).stem.split("_")[0] for name in selected}),
        "images": selected,
        "interpretation": "Pinned public validation sample for detector audit; not Wuhan campus event accuracy.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = build_selection(args.archive, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(selection["images"]),
                      "scene_prefixes": selection["scene_prefix_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
