"""Keep confirmed engineering market POI labels aligned with source geometry."""

import json
from pathlib import Path

from scripts.fetch.campus_scope import campus_for_gcj, load_whu_polygons


ROOT = Path(__file__).resolve().parents[1]


def test_engineering_market_and_contained_shops_have_consistent_campus():
    pois = {
        item["id"]: item
        for item in json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]
    }
    boundaries = load_whu_polygons()
    for poi_id in ("poi_387", "poi_388", "poi_415"):
        poi = pois[poi_id]
        assert campus_for_gcj(**poi["coordinates"], polygons=boundaries) == "工学部"
        assert poi["campus"] == "工学部"
        assert poi["description"].startswith("工学部校内地点。")
