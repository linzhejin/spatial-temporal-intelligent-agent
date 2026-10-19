"""Rebuild auditable activity tags without changing existing POI IDs or geometry."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spatial.poi_taxonomy import audit_poi_activities  # noqa: E402


def rebuild(poi_document: dict) -> dict:
    rows = poi_document.get("pois")
    if not isinstance(rows, list):
        raise ValueError("POI document must contain a pois list")
    by_id = {poi.get("id"): poi for poi in rows}
    if len(by_id) != len(rows) or None in by_id:
        raise ValueError("POI IDs must be present and unique")

    # These short names were attached to a dormitory and an area in the legacy
    # catalog. Keep the known sports-ground aliases on their actual grounds.
    for poi_id, alias in (("poi_110", "桂操"), ("poi_151", "梅操")):
        if poi_id not in by_id:
            raise ValueError(f"expected reviewed POI is missing: {poi_id}")
        by_id[poi_id]["aliases"] = [value for value in by_id[poi_id].get("aliases", [])
                                     if value != alias]
    expected_aliases = {
        "poi_085": ("桂操",),
        "poi_082": ("梅操",),
        "poi_260": ("信操", "竹园操场"),
    }
    for poi_id, aliases_to_add in expected_aliases.items():
        if poi_id not in by_id:
            raise ValueError(f"expected sports ground is missing: {poi_id}")
        aliases = by_id[poi_id].setdefault("aliases", [])
        for alias in aliases_to_add:
            if alias not in aliases:
                aliases.append(alias)

    reviewed = {item["poi_id"]: item for item in audit_poi_activities(rows)}
    if len(reviewed) != len(rows):
        raise ValueError("activity audit did not cover every POI")
    for poi in rows:
        activity_record = dict(reviewed[poi["id"]])
        activity_record.pop("poi_id", None)
        poi.update(activity_record)
    poi_document["activity_taxonomy_version"] = 1
    return poi_document


def main() -> int:
    path = ROOT / "data" / "pois.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = rebuild(document)
    # Keep the checked-in master’s established one-space indentation so this
    # enrichment remains a row-level data diff instead of a whole-file reformat.
    path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Audited activity tags for {len(rebuilt['pois'])} POIs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
