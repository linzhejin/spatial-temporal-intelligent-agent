"""批量为 data/pois.json 打 subcategory / is_minor 标签。

规则按名称关键词正则匹配（按顺序命中即停），打标结果回写 pois.json，
同时输出 data/pois_subcategory_review.csv 供人工复核。

设计原则：宁可漏标（subcategory 留空，检索时退化为 type 级）不可错标。
is_minor=True 表示小店铺（连锁奶茶/咖啡/快餐档口等），仅在泛推荐时降权，
用户明确按类别检索时仍可命中。

用法：
    python scripts/tag_subcategory.py            # 打标并回写
    python scripts/tag_subcategory.py --dry-run  # 只输出 CSV 不回写
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POIS_PATH = ROOT / "data" / "pois.json"
REVIEW_CSV = ROOT / "data" / "pois_subcategory_review.csv"

# 每个 type 下的有序规则：(正则, subcategory, is_minor)
_RULES = {
    "dining": [
        (r"咖啡|瑞幸|库迪|[Mm]anner|星巴克|Café|café", "coffee", True),
        (r"蜜雪冰城|古茗|茶颜悦色|喜茶|霸王茶姬|书亦烧仙草|茶百道|柠季|柠檬茶"
         r"|鲜茶|饮品|酸奶|冰激淋|冰淇淋|糖水|水果捞|烧仙草|不泡茶|豆花|东方茶",
         "tea_drink", True),
        (r"餐厅|湘菜馆|川味坊|家常菜|农家小炒|宝岛小厨|小炒|孃孃出川|周麻婆"
         r"|火锅|餐吧|烤鸭|小厨", "restaurant", False),
        (r"食堂.{0,3}(?:店|摊|档口)|(?:店|摊|档口).{0,3}食堂", "fastfood", True),
        (r"食堂", "canteen", False),
        (r".", "fastfood", True),  # 其余餐饮一律按快餐小食
    ],
    "study": [
        (r"图书馆|阅览室", "library", False),
        (r"博物馆|音乐厅", "culture", False),
        (r"教学楼|教\d+楼|教[一二三四五六七八九十]|综合楼|实验大楼", "classroom", False),
        (r"学院|研究院|研究中心|实验室|研究所|中心|训练", "college", False),
        (r".", "study_other", False),
    ],
    "sports": [
        (r"游泳池", "pool", False),
        (r"体育馆|风雨操场", "gym", False),
        (r"操场|运动场|球场", "field", False),
        (r".", "sports_other", False),
    ],
    "scenery": [
        (r"樱花大道|樱园|老斋舍|日字斋", "sakura", False),
        (r"湖|栈桥|码头", "lake", False),
        (r"珞珈山|狮子山|火石山", "hill", False),
        (r"公园|梅园|枫园|广场", "park", False),
        (r".", "landmark", False),
    ],
    "service": [
        (r"集贸市场|农贸市场|菜市场|菜场", "market", False),
        (r"便利店|便利商店", "convenience_store", False),
        (r"超市", "supermarket", False),
        (r"银行", "bank", False),
        (r"邮政", "post", False),
        (r"医院", "hospital", False),
        (r"活动中心|明珠园", "activity_center", False),
        (r".", "service_other", False),
    ],
    "dorm": [(r".", "dormitory", False)],
    "gate": [(r".", "gate", False)],
    "area": [(r".", "area", False)],
}


def tag_poi(poi: dict) -> tuple[str, bool, bool]:
    """返回 (subcategory, is_minor, matched)。matched=False 表示该 type 无规则。"""
    name = poi.get("name", "")
    rules = _RULES.get(poi.get("type", ""))
    if not rules:
        return "", False, False
    for pattern, subcategory, is_minor in rules:
        if re.search(pattern, name):
            return subcategory, is_minor, True
    return "", False, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只输出复核 CSV，不回写 pois.json")
    args = parser.parse_args()

    data = json.loads(POIS_PATH.read_text(encoding="utf-8"))
    pois = data["pois"] if isinstance(data, dict) and "pois" in data else data

    unmatched, changed = [], 0
    rows = []
    for poi in pois:
        subcategory, is_minor, matched = tag_poi(poi)
        if not matched:
            unmatched.append(poi.get("name", "?"))
        if poi.get("subcategory") != subcategory or bool(poi.get("is_minor", False)) != is_minor:
            changed += 1
        poi["subcategory"] = subcategory
        poi["is_minor"] = is_minor
        rows.append({
            "id": poi.get("id", ""),
            "name": poi.get("name", ""),
            "type": poi.get("type", ""),
            "subcategory": subcategory,
            "is_minor": int(is_minor),
            "campus": poi.get("campus", ""),
        })

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "type", "subcategory", "is_minor", "campus"])
        writer.writeheader()
        writer.writerows(rows)

    if not args.dry_run:
        if isinstance(data, dict) and "pois" in data:
            data["pois"] = pois
        else:
            data = pois
        POIS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    print(f"POI 总数: {len(pois)}，本次变更: {changed}，复核表: {REVIEW_CSV}")
    if unmatched:
        print(f"未命中规则（需人工标注）: {len(unmatched)} 条")
        for n in unmatched:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
