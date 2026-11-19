# -*- coding: utf-8 -*-
"""对主线路网的每条有向边采样绿量与坡度，输出覆盖统计与增强属性文件。

绿量：Sentinel-2 NDVI，边两侧 15 m 缓冲取中位数，按全图五分位映射 1–5 级。
坡度：DEM 端点高程差与边长之比，按项目现有阈值分级（<2%、<5%、<8%、<15%、≥15%）。

用法：
  python scripts/network/derive_greenery_from_ndvi.py
  python scripts/network/derive_greenery_from_ndvi.py --dry-run
  python scripts/network/derive_greenery_from_ndvi.py --ndvi input.ndvi.tif --dem input.dem.tif
"""
import argparse
import io
import json
import warnings
import math
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.mask import mask as rio_mask
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "data" / "whu_road_network.graphml"
NDVI = ROOT / "非项目材料" / "OGE东方航天港杯" / "local" / "track1-runs" / \
       "20261017-sentinel-autumn-v3-20251114" / "whu_s2_autumn_20251114_ndvi.tif"
DEM = ROOT / "非项目材料" / "OGE东方航天港杯" / "local" / "track1-runs" / \
      "20261015-platform-v4" / "exports" / "whu_terrain_20261015_v4_dem.tif"
OUT_STATS = ROOT / "实验补充" / "属性覆盖_20261119.json"
OUT_ATTR = ROOT / "data" / "edge_environment_attributes_20261119.json"

BUFFER_M = 15.0
SLOPE_BINS = [(0.02, 1), (0.05, 2), (0.08, 3), (0.15, 4), (math.inf, 5)]


def slope_level(g):
    for th, lv in SLOPE_BINS:
        if g < th:
            return lv
    return 5


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=GRAPH, help="输入路网 GraphML")
    parser.add_argument("--ndvi", type=Path, default=NDVI, help="输入 NDVI 栅格")
    parser.add_argument("--dem", type=Path, default=DEM, help="输入 DEM 栅格")
    parser.add_argument("--stats-output", type=Path, default=OUT_STATS,
                        help="覆盖统计 JSON 输出路径")
    parser.add_argument("--attributes-output", type=Path, default=OUT_ATTR,
                        help="边属性 JSON 输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写输出文件")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for label, path in (("路网", args.graph), ("NDVI", args.ndvi), ("DEM", args.dem)):
        if not path.is_file():
            parser.error(f"{label}输入文件不存在：{path}")

    warnings.filterwarnings('ignore')
    g = nx.read_graphml(str(args.graph))
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

    ndvi = rasterio.open(args.ndvi)
    dem = rasterio.open(args.dem)
    nd_no = ndvi.nodata if ndvi.nodata is not None else np.nan
    dem_no = dem.nodata if dem.nodata is not None else np.nan

    rows = []
    skipped = 0
    seen = set()
    for u, v, d in g.edges(data=True):
        key = tuple(sorted((u, v)))
        if key in seen:          # 有向边成对，同一几何只采样一次
            continue
        seen.add(key)
        geom = d.get("geometry")
        if geom:
            line = wkt.loads(geom)
        else:
            nu, nv = g.nodes[u], g.nodes[v]
            line = LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])])
            skipped += 1
        line_m = shp_transform(tr.transform, line)
        length = float(d.get("length") or line_m.length)

        buf = line_m.buffer(BUFFER_M)
        try:
            arr, _ = rio_mask(ndvi, [buf], crop=True, filled=True, nodata=np.nan)
            vals = np.asarray(arr[0], dtype="float64").ravel()
            vals = vals[np.isfinite(vals)]
            if ndvi.nodata is not None and not np.isnan(ndvi.nodata):
                vals = vals[vals != nd_no]
            nd = float(np.median(vals)) if vals.size else None
        except Exception:
            nd = None
        if nd is not None and not (-1.0 <= nd <= 1.0):
            nd = None

        hs = []
        for x, y in (line_m.coords[0], line_m.coords[-1]):
            r, c = dem.index(x, y)
            if 0 <= r < dem.height and 0 <= c < dem.width:
                h = float(dem.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
                if not (dem.nodata is not None and h == dem_no) and not np.isnan(h):
                    hs.append(h)
        grade = abs(hs[0] - hs[1]) / length if len(hs) == 2 and length > 0 else None

        rows.append({
            "u": u, "v": v,
            "length_m": round(length, 2),
            "greenery_ndvi": None if nd is None else round(nd, 4),
            "slope_grade_abs_pct": None if grade is None else round(grade * 100, 3),
        })

    # 五分位分级
    nds = sorted(r["greenery_ndvi"] for r in rows if r["greenery_ndvi"] is not None)
    cuts = [np.percentile(nds, p) for p in (20, 40, 60, 80)] if nds else []
    for r in rows:
        nd = r["greenery_ndvi"]
        r["greenery_level"] = None if nd is None else int(1 + sum(nd >= c for c in cuts))
        gr = r["slope_grade_abs_pct"]
        r["slope_level"] = None if gr is None else int(slope_level(gr / 100.0))

    covered_keys = {tuple(sorted((r["u"], r["v"]))) for r in rows}
    directed_covered = sum(1 for u, v in g.edges() if tuple(sorted((u, v))) in covered_keys)
    stats = {
        "graph_edges_directed": g.number_of_edges(),
        "directed_edges_covered": directed_covered,
        "sampled_undirected_pairs": len(rows),
        "fallback_no_geometry": skipped,
        "buffer_m": BUFFER_M,
        "ndvi_source": args.ndvi.name,
        "dem_source": args.dem.name,
        "greenery_covered": sum(1 for r in rows if r["greenery_level"] is not None),
        "slope_covered": sum(1 for r in rows if r["slope_level"] is not None),
        "greenery_hist": {str(k): int(v) for k, v in sorted(Counter(r["greenery_level"] for r in rows).items(), key=lambda x: str(x[0]))},
        "slope_hist": {str(k): int(v) for k, v in sorted(Counter(r["slope_level"] for r in rows).items(), key=lambda x: str(x[0]))},
        "ndvi_quantile_cuts": [round(float(c), 4) for c in cuts],
        "ndvi_median": round(float(np.median(nds)), 4) if nds else None,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.attributes_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    args.attributes_output.write_text(json.dumps({"records": rows}, ensure_ascii=False), encoding="utf-8")
    print("written:", args.stats_output, args.attributes_output)


if __name__ == "__main__":
    main()
