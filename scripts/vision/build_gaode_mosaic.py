"""拼接武大校园范围的高德卫星参考底图（仅内部校准/可视化）。

用法：
  python scripts/vision/build_gaode_mosaic.py --zoom 18

输出到 research/local/reference/（不入 Git）：
  whu_satellite_gcj_z18.png
  whu_satellite_gcj_z18.json   # bbox、瓦片范围、来源、获取日期

边界：瓦片为 GCJ-02；只用于内部核对 footprint 与道路，不再分发，不能作为
对外地图数据包。管理台仍使用高德在线图层，本脚本不改变生产依赖。
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

_TILE_SIZE = 256
_TILE_URL = "https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
_MAX_TILES = 600
_UA = "Mozilla/5.0 (compatible; WHU-Walker-internal-reference/1.0)"


def deg_to_num(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def num_to_deg(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return math.degrees(lat_rad), lon


def fetch_tile(x: int, y: int, zoom: int, retries: int = 3) -> Image.Image:
    last_error = None
    for attempt in range(retries):
        server = (x + y) % 4 + 1
        url = _TILE_URL.format(s=server, x=x, y=y, z=zoom)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read()
            from io import BytesIO
            image = Image.open(BytesIO(data)).convert("RGB")
            if image.size != (_TILE_SIZE, _TILE_SIZE):
                image = image.resize((_TILE_SIZE, _TILE_SIZE))
            return image
        except Exception as error:
            last_error = error
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"瓦片下载失败 z={zoom} x={x} y={y}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--west", type=float, default=114.345)
    parser.add_argument("--south", type=float, default=30.525)
    parser.add_argument("--east", type=float, default=114.385)
    parser.add_argument("--north", type=float, default=30.550)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research/local/reference"))
    args = parser.parse_args()
    if not (args.west < args.east and args.south < args.north):
        raise ValueError("bbox 必须满足 west<east 且 south<north")
    x_min_f, y_min_f = deg_to_num(args.north, args.west, args.zoom)
    x_max_f, y_max_f = deg_to_num(args.south, args.east, args.zoom)
    x_min, x_max = int(math.floor(x_min_f)), int(math.floor(x_max_f))
    y_min, y_max = int(math.floor(y_min_f)), int(math.floor(y_max_f))
    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if tile_count > _MAX_TILES:
        raise ValueError(f"瓦片数量 {tile_count} 超过上限 {_MAX_TILES}；降低 zoom 或缩小 bbox")

    canvas = Image.new("RGB", ((x_max - x_min + 1) * _TILE_SIZE,
                               (y_max - y_min + 1) * _TILE_SIZE), (32, 32, 32))
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile = fetch_tile(x, y, args.zoom)
            canvas.paste(tile, ((x - x_min) * _TILE_SIZE, (y - y_min) * _TILE_SIZE))

    north, west = num_to_deg(x_min, y_min, args.zoom)
    south, east = num_to_deg(x_max + 1, y_max + 1, args.zoom)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"whu_satellite_gcj_z{args.zoom}"
    image_path = args.output_dir / f"{name}.png"
    canvas.save(image_path, format="PNG")
    metadata = {
        "name": name,
        "source": "AutoNavi (Gaode) satellite tiles, style=6",
        "coordinate_system": "GCJ-02",
        "zoom": args.zoom,
        "tile_range": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "tile_count": tile_count,
        "bbox_gcj": {"west": round(west, 7), "south": round(south, 7),
                     "east": round(east, 7), "north": round(north, 7)},
        "pixel_size": [canvas.width, canvas.height],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "license_note": "仅内部校准/可视化使用，不再分发；不得作为对外地图数据包。",
    }
    metadata_path = args.output_dir / f"{name}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(image_path), "metadata": str(metadata_path),
                      "pixels": [canvas.width, canvas.height], "tiles": tile_count},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
