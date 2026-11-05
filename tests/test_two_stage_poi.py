# -*- coding: utf-8 -*-
"""两阶段 POI 识别测试（V2 数据扩展配套）

覆盖：
  1. POI 数据完整性（pois.json 合并结果：数量、字段、坐标范围）
  2. 后端归一化 _normalize_poi_refs（别名 / 模糊 / 未匹配保留 / 类型不动）
  3. prompt 瘦身（不含 POI 列表，两阶段架构生效）
  4. 规则兜底在新 320+ POI 库下工作正常
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parser import (
    _build_poi_name_index,
    _load_system_prompt,
    _normalize_poi_refs,
    _rule_based_classify,
    Constraints,
    PoiRef,
    TaskIntent,
)
from spatial.poi import find_poi, find_poi_candidates, load_pois

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POIS_PATH = os.path.join(PROJECT_ROOT, "data", "pois.json")

# 校园 bbox（config.WHU_BBOX 同值，硬编码避免 env 依赖）
BBOX = {"north": 30.5510, "south": 30.5170, "east": 114.3880, "west": 114.3460}


# ===== 1. POI 数据完整性 =====

class TestPoiDataIntegrity:
    @pytest.fixture(scope="class")
    def pois_data(self):
        with open(POIS_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_poi_count_expanded(self, pois_data):
        """V2 合并后 POI 数量应显著超过原 24 条。"""
        assert pois_data["count"] >= 290
        assert len(pois_data["pois"]) == pois_data["count"]

    def test_poi_required_fields(self, pois_data):
        for poi in pois_data["pois"]:
            assert poi.get("name"), f"POI 缺 name: {poi}"
            coords = poi.get("coordinates") or {}
            assert "lng" in coords and "lat" in coords, f"POI 缺坐标: {poi['name']}"

    def test_coords_in_campus_bbox(self, pois_data):
        """所有 POI 坐标应落在扩展后的三学部 bbox 内（南缘食堂等允许少量越界，
        实测最近路网节点吸附距离 ≤ 200m，路由可用）。"""
        tol = 0.005
        for poi in pois_data["pois"]:
            lng = poi["coordinates"]["lng"]
            lat = poi["coordinates"]["lat"]
            assert BBOX["west"] - tol <= lng <= BBOX["east"] + tol, \
                f"{poi['name']} 经度越界: {lng}"
            assert BBOX["south"] - tol <= lat <= BBOX["north"] + tol, \
                f"{poi['name']} 纬度越界: {lat}"

    def test_no_duplicate_names(self, pois_data):
        names = [p["name"] for p in pois_data["pois"]]
        assert len(names) == len(set(names)), "存在重复 POI 名称"

    def test_load_pois_matches_json(self, pois_data):
        held = [poi for poi in pois_data["pois"]
                if poi.get("recommendation_status") == "reported_absent_pending_review"]
        active = load_pois()
        assert len(pois_data["pois"]) == pois_data["count"]
        assert len(held) == 3
        assert len(active) == pois_data["count"] - len(held)
        assert not {poi["id"] for poi in active} & {poi["id"] for poi in held}


# ===== 2. 后端地名归一化（两阶段·第二阶段） =====

class TestNormalizePoiRefs:
    def _intent(self, start=None, end=None, task_type="path_planning"):
        return TaskIntent(
            task_type=task_type,
            start=PoiRef(name=start, type="poi") if start else None,
            end=PoiRef(name=end, type="poi") if end else None,
            constraints=Constraints(distance="medium", slope="normal", scenery="normal"),
        )

    def test_alias_normalized(self):
        """「图书馆」应归一化为总馆规范名。"""
        intent = self._intent(start="教五", end="图书馆")
        result = _normalize_poi_refs(intent)
        assert result.end.name == "武汉大学图书馆(总馆)"

    def test_fuzzy_prefix_normalized(self):
        """「武大牌坊」应归一化为「珞珈门」（牌坊为别名）。"""
        result = _normalize_poi_refs(self._intent(start="武大牌坊"))
        assert result.start.name == "珞珈门"

    def test_exact_name_unchanged(self):
        """别名命中时归一化为规范名「珞珈门」。"""
        result = _normalize_poi_refs(self._intent(start="牌坊"))
        assert result.start.name == "珞珈门"

    def test_unmatched_keeps_raw(self):
        """匹配不到的地名保留原样（由 route 层引导用户）。"""
        raw = "不存在的地方xyz"
        result = _normalize_poi_refs(self._intent(start=raw))
        assert result.start.name == raw

    def test_none_refs_untouched(self):
        result = _normalize_poi_refs(self._intent())
        assert result.start is None and result.end is None

    def test_chat_intent_skipped(self):
        """非路径/查询意图不做归一化。"""
        intent = TaskIntent(
            task_type="chat",
            start=None,
            end=None,
            constraints=Constraints(distance="medium", slope="normal", scenery="normal"),
        )
        assert _normalize_poi_refs(intent).task_type == "chat"

    def test_both_ends_normalized(self):
        result = _normalize_poi_refs(self._intent(start="图书馆", end="樱顶"))
        assert result.start.name == "武汉大学图书馆(总馆)"
        assert result.end.name == "武汉大学老斋舍"


# ===== 3. prompt 瘦身（两阶段·第一阶段） =====

class TestPromptSlim:
    def test_prompt_has_no_poi_list(self):
        prompt = _load_system_prompt()
        assert "poi_list_json" not in prompt
        assert "{poi_count}" not in prompt

    def test_prompt_size_bounded(self):
        """不含 POI 列表的 prompt 应远小于注入 320 POI 时的体量。"""
        assert len(_load_system_prompt()) < 10000

    def test_prompt_has_verbatim_rule(self):
        """prompt 应包含「原样提取」指令。"""
        prompt = _load_system_prompt()
        assert "原样提取" in prompt


# ===== 4. 规则兜底 + POI 索引 =====

class TestRuleFallbackWithExpandedPois:
    def test_name_index_built(self):
        names_set, names_lower = _build_poi_name_index()
        assert len(names_set) >= 290
        assert "总图书馆" in names_set

    def test_index_contains_aliases(self):
        _, names_lower = _build_poi_name_index()
        # 「图书馆」可能是别名或模糊可达，索引至少应能查到规范名
        assert any("图书馆" in k for k in names_lower)

    def test_classify_path_query(self):
        r = _rule_based_classify("从牌坊到樱顶")
        assert r["task_type"] == "path_planning"
        assert r["start_name"] == "珞珈门"
        assert r["end_name"] == "武汉大学老斋舍"

    def test_classify_poi_query(self):
        """规则层返回规范名（两阶段：归一化由 fuzzy 统一映射完成）。"""
        r = _rule_based_classify("图书馆怎么走")
        assert r["task_type"] == "poi_query"
        assert "图书馆" in r["start_name"]

    def test_pipeline_rule_then_normalize(self):
        """管线级：规则提取原文名 → 归一化得规范名。"""
        r = _rule_based_classify("图书馆怎么走")
        intent = TaskIntent(
            task_type=r["task_type"],
            start=PoiRef(name=r["start_name"], type="poi"),
            end=None,
            constraints=Constraints(distance="medium", slope="normal", scenery="normal"),
        )
        intent = _normalize_poi_refs(intent)
        assert intent.start.name == "武汉大学图书馆(总馆)"

    def test_classify_new_campus_poi(self):
        """三学部扩展后的新 POI 应能被规则兜底匹配。"""
        r = _rule_based_classify("信息学部第一教学楼在哪")
        assert r["task_type"] == "poi_query"


# ===== 6. 校园边界（V2.1：修复校外 POI/穿城路线）=====

class TestCampusBoundary:
    def test_all_pois_inside_campus_polys(self):
        """所有 POI 必须落在三学部校园多边形内。"""
        import networkx as nx  # noqa: F401
        from config import CAMPUS_POLYS_GCJ

        def pip(lng, lat, poly):
            inside = False
            j = len(poly) - 1
            for i in range(len(poly)):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > lat) != (yj > lat)) and \
                        (lng < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
                    inside = not inside
                j = i
            return inside

        pois = load_pois()
        assert len(pois) >= 290
        for p in pois:
            lng, lat = p["coordinates"]["lng"], p["coordinates"]["lat"]
            assert any(pip(lng, lat, poly) for poly in CAMPUS_POLYS_GCJ.values()), \
                f"校外 POI 未清理: {p['name']} ({lng}, {lat})"

    def test_blacklist_pois_absent(self):
        from config import POI_NAME_BLACKLIST
        names = [p["name"] for p in load_pois()]
        for bad in POI_NAME_BLACKLIST:
            assert not any(bad in n for n in names), f"黑名单 POI 仍存在: {bad}"

    def test_known_offcampus_pois_removed(self):
        """已确认的校外点必须不存在。"""
        names = {p["name"] for p in load_pois()}
        for bad in ("珞珈山剧院", "武汉工学院", "东湖食堂", "国家网络安全学院大楼"):
            assert bad not in names, f"校外 POI 未删除: {bad}"

    def test_network_clipped_to_campus(self):
        """路网节点应全部在校园多边形（WGS-84）缓冲带内，且三校区连通。"""
        import networkx as nx
        from shapely.geometry import Point, Polygon
        from shapely.ops import unary_union
        from config import CAMPUS_POLYS_GCJ
        from spatial.coord_transform import gcj02_to_wgs84

        G = nx.read_graphml(
            os.path.join(PROJECT_ROOT, "data", "whu_road_network.graphml"),
            node_type=int,
        )
        polys = [
            Polygon([gcj02_to_wgs84(lng, lat) for lng, lat in poly])
            for poly in CAMPUS_POLYS_GCJ.values()
        ]
        clip = unary_union(polys).buffer(0.0011)
        outside = [
            n for n, d in G.nodes(data=True)
            if not clip.covers(Point(float(d["x"]), float(d["y"])))
        ]
        assert len(outside) == 0, f"{len(outside)} 个路网节点在校园缓冲带外"

        # 三校区同属一个大连通分量（跨学部过街连接未被裁断）
        largest = max(nx.weakly_connected_components(G), key=len)
        assert len(largest) / G.number_of_nodes() > 0.95


# ===== 5. 匹配质量（别名/消歧） =====

class TestMatchingQuality:
    def test_find_poi_candidates_returns_scored(self):
        cands = find_poi_candidates("图书馆", limit=5)
        assert cands, "「图书馆」应有候选"
        assert cands[0][0]["name"] == "武汉大学图书馆(总馆)"
        scores = [s for _, s in cands]
        assert scores == sorted(scores, reverse=True)

    def test_garbage_returns_none(self):
        assert find_poi("zzz不存在的地点qqq") is None


# ===== 6. 校门坐标与别名契约（2026-09 高德 API + 卫星图逐门核对） =====

class TestGateCoordinates:
    GATE_NAMES = {
        "凌波门", "珞南门", "武汉大学科技门", "珞珈门", "武汉大学茶港门",
        "洪波门", "武汉大学文澜门", "珞瑜二门", "珞瑜门", "弘毅门",
        "北门", "扬波门", "澄波门", "珞东二门", "珞南二门", "珞南三门",
        "玉兰2门",
    }

    @pytest.fixture(scope="class")
    def gates(self):
        with open(POIS_PATH, encoding="utf-8") as f:
            pois = json.load(f)["pois"]
        return {p["name"]: p for p in pois if p.get("subcategory") == "gate"}

    def test_required_gates_present(self, gates):
        missing = self.GATE_NAMES - set(gates)
        assert not missing, f"缺少校门 POI: {missing}"
        assert len(gates) >= len(self.GATE_NAMES)

    def test_no_phantom_xinan_gate(self, gates):
        """武大批图上曾误标「西南门」（珞珈山路附近的近似点），
        经熟悉校园者核实该门不存在，禁止以任何名称/别名再加回。"""
        with open(POIS_PATH, encoding="utf-8") as f:
            all_pois = json.load(f)["pois"]
        for term in ("西南门", "武汉大学西南门", "武大西南门"):
            assert term not in gates
            assert not any(term in p.get("aliases", []) for p in all_pois), term

    def test_luoyu_gate_on_luoyu_road(self, gates):
        """珞瑜门（信息学部正门）必须压在珞瑜路一侧：纬度 < 30.525。
        历史上该点曾错放在门内友谊广场中央（30.5254，偏北约 87m）。"""
        p = gates["珞瑜门"]
        assert p["coordinates"]["lat"] < 30.525, p["coordinates"]
        assert abs(p["coordinates"]["lng"] - 114.36116) < 0.0003

    def test_lingbo_and_yangbo_distinct(self, gates):
        """凌波门（东湖南路栈桥地标）与扬波门（东门）相距 1km 左右，
        不得混为一点。"""
        a = gates["凌波门"]["coordinates"]
        b = gates["扬波门"]["coordinates"]
        dlat = (a["lat"] - b["lat"]) * 111000
        dlng = (a["lng"] - b["lng"]) * 96000
        assert (dlat ** 2 + dlng ** 2) ** 0.5 > 800

    def test_gate_colloquial_aliases_resolve(self):
        """学生/游客常用俗称必须解析到正确的校门。"""
        assert find_poi("广埠屯校门")["name"] == "珞瑜门"
        assert find_poi("附中门")["name"] == "珞南门"
        assert find_poi("武大东门")["name"] == "扬波门"
        assert find_poi("牌坊")["name"] == "珞珈门"
        assert find_poi("玉兰二门")["id"] == "poi_449"
        assert find_poi("武汉大学玉兰2门")["name"] == "玉兰2门"

    def test_chagang_gate_belongs_engineering_campus(self, gates):
        """校保卫部将茶港门列在工学部：https://www.whu.edu.cn/info/5231/240334.htm"""
        p = gates["武汉大学茶港门"]
        assert p["campus"] == "工学部"
        assert "工学部茶港门" in p["aliases"]
        assert "文理学部茶港门" not in p["aliases"]

    def test_shuisheng_alias_points_to_wenlan(self, gates):
        """「水生所旁校门」是文澜门，曾误挂洪波门。"""
        assert "水生所旁校门" in gates["武汉大学文澜门"]["aliases"]
        assert "水生所旁校门" not in gates["洪波门"]["aliases"]
        assert find_poi("水生所旁校门")["name"] == "武汉大学文澜门"

    def test_nan_damen_is_luoyu_gate(self):
        """「南大门」＝信息学部珞瑜门；而「南门」仍是珞南门，两者不可混。"""
        assert find_poi("南大门")["name"] == "珞瑜门"
        assert find_poi("南门")["name"] == "珞南门"


# ===== 7. 校园文化 prompt 边界（直答授权 + 易变事实护栏） =====

class TestCulturePrompt:
    PROMPT_PATH = os.path.join(
        PROJECT_ROOT, "agents", "prompts", "agent_system.txt"
    )

    @pytest.fixture(scope="class")
    def prompt(self):
        with open(self.PROMPT_PATH, encoding="utf-8") as f:
            return f.read()

    def test_culture_qa_authorized(self, prompt):
        assert "校园文化常识" in prompt

    def test_sixteen_dorms_anchor_present(self, prompt):
        """十六斋序列是高错率锚点，缺失会让模型乱排。"""
        for seq in ("天、地、玄、黄", "宇、宙、洪、荒", "日、月、盈、昃", "辰、宿、列、张"):
            assert seq in prompt

    def test_gate_disambiguation_present(self, prompt):
        assert "扬波门" in prompt and "凌波门" in prompt
        assert "珞瑜门" in prompt and "珞珈门" in prompt

    def test_mutable_rules_guardrail_present(self, prompt):
        """易变规则不许凭记忆答，必须导向"看最新通知"。"""
        assert "最新通知" in prompt
        assert "禁止凭记忆作答" in prompt
