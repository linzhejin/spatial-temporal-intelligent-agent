import json
from pathlib import Path

from spatial.poi_taxonomy import (
    ACTIVITY_SUBCATEGORIES,
    audit_poi_activities,
    matches_activities,
    resolve_activity_query,
    resolve_activity_sequence,
)
from spatial.poi import _exact_master_poi_matches, load_pois


def _by_id():
    return {poi["id"]: poi for poi in load_pois()}


def test_explicit_ball_sports_map_to_their_own_facilities():
    assert resolve_activity_query("我想打篮球") == {
        "status": "resolved", "activities": ["basketball"],
        "subcategories": ["field", "sports_field", "court"],
    }
    assert resolve_activity_query("去打羽毛球") == {
        "status": "resolved", "activities": ["badminton"],
        "subcategories": ["field", "sports_field", "court", "gym"],
    }
    assert resolve_activity_query("想打乒乓球") == {
        "status": "resolved", "activities": ["table_tennis"],
        "subcategories": ["court", "sports_centre", "gym"],
    }
    assert resolve_activity_query("去踢足球") == {
        "status": "resolved", "activities": ["football"],
        "subcategories": ["field", "sports_field"],
    }


def test_unspecified_ball_sport_requires_the_compact_choices():
    assert resolve_activity_query("我想去打球") == {
        "status": "needs_clarification",
        "question": "想打什么球？",
        "options": ["篮球", "羽毛球", "乒乓球", "足球", "其他球类"],
    }


def test_general_exercise_requires_an_activity_but_running_is_specific():
    assert resolve_activity_query("我想去运动")["status"] == "needs_clarification"
    assert resolve_activity_query("找个地方跑步") == {
        "status": "resolved", "activities": ["running"],
        "subcategories": ["field", "sports_field"],
    }


def test_common_activity_phrases_cover_the_controlled_activity_vocabulary():
    phrases = {
        "找篮球场": "basketball",
        "去打羽毛球": "badminton",
        "去打乒乓球": "table_tennis",
        "去打乒乓": "table_tennis",
        "找足球场": "football",
        "想打排球": "volleyball",
        "去打网球": "tennis",
        "去跑步": "running",
        "想去跑操": "running",
        "找健身房": "fitness",
        "去游泳": "swimming",
        "我想吃饭": "meal",
        "找食堂": "canteen_meal",
        "找餐厅": "restaurant_meal",
        "找快餐": "fast_food",
        "想喝咖啡": "coffee",
        "想喝奶茶": "tea_drink",
        "找学习场所": "study",
        "附近超市": "shopping",
        "去医院看病": "medical",
        "哪里取快递": "package_pickup",
        "找邮局": "postal_service",
        "去看湖景": "view_lake",
        "想去赏樱": "view_sakura",
    }
    assert set(phrases.values()) == set(ACTIVITY_SUBCATEGORIES)
    for phrase, activity in phrases.items():
        resolved = resolve_activity_query(phrase)
        assert resolved == {
            "status": "resolved",
            "activities": [activity],
            "subcategories": list(ACTIVITY_SUBCATEGORIES[activity]),
        }, phrase


def test_courier_point_synonyms_resolve_to_package_pickup():
    assert resolve_activity_query("武汉大学有哪些快递点") == {
        "status": "resolved", "activities": ["package_pickup"],
        "subcategories": ["service"],
    }
    assert resolve_activity_query("哪里可以取件") == {
        "status": "resolved", "activities": ["package_pickup"],
        "subcategories": ["service"],
    }


def test_general_study_place_phrases_resolve_to_evidenced_study_activity():
    assert resolve_activity_query("武汉大学有哪些学习场所") == {
        "status": "resolved", "activities": ["study"],
        "subcategories": ["library", "study_other"],
    }
    assert resolve_activity_query("找个安静的地方") == {
        "status": "resolved", "activities": ["study"],
        "subcategories": ["library", "study_other"],
    }


def test_buying_bottled_water_resolves_to_location_ranked_shopping():
    for phrase in ("我想买水", "去买瓶水", "附近买饮料"):
        assert resolve_activity_query(phrase) == {
            "status": "resolved", "activities": ["shopping"],
            "subcategories": ["supermarket"],
        }
    assert resolve_activity_query("我想喝水") is None


def test_composite_activity_requests_keep_order_and_unresolved_details():
    sequence = resolve_activity_sequence("我想先打球再吃饭")

    assert [item["query"] for item in sequence] == ["我想先打球", "吃饭"]
    assert sequence[0]["intent"]["status"] == "needs_clarification"
    assert sequence[1]["intent"]["activities"] == ["meal"]

    explicit = resolve_activity_sequence("先打篮球，然后去吃饭")
    assert [item["intent"]["activities"] for item in explicit] == [
        ["basketball"], ["meal"],
    ]
    assert resolve_activity_query("我想去健身") == {
        "status": "resolved", "activities": ["fitness"],
        "subcategories": ["gym", "sports_centre", "sports_other"],
    }


def test_composite_request_does_not_drop_a_final_unspecified_dorm():
    sequence = resolve_activity_sequence("先打篮球，再吃饭，最后回宿舍，不走台阶")

    assert len(sequence) == 3
    assert [item["intent"]["activities"] for item in sequence] == [
        ["basketball"], ["meal"], [],
    ]
    assert "宿舍" in sequence[2]["query"]
    assert sequence[2]["intent"]["status"] == "needs_destination"


def test_curated_pois_carry_audited_activity_provenance():
    pois = _by_id()
    # Keep the established coarse category field stable; use the new activity
    # field for the specific supported sport.
    assert {pois[poi_id]["subcategory"] for poi_id in ("poi_083", "poi_206", "poi_262")} == {"field"}
    assert all("basketball" in pois[poi_id]["activities"]
               for poi_id in ("poi_083", "poi_206", "poi_262"))
    assert all(poi.get("activity_review_status") in {
        "source_supported", "category_supported", "reviewed_no_supported_activity",
    } for poi in pois.values())
    assert len(pois) == len(json.loads(Path("data/pois.json").read_text(encoding="utf-8"))["pois"])


def test_every_master_poi_matches_its_reproducible_activity_audit():
    pois = load_pois()
    audited = audit_poi_activities(pois)
    by_id = {poi["id"]: poi for poi in pois}
    assert len(audited) == len(pois) == 443
    for decision in audited:
        poi = by_id[decision["poi_id"]]
        assert poi.get("activities", []) == decision["activities"], poi["id"]
        assert poi.get("activity_basis", "") == decision["activity_basis"], poi["id"]
        assert poi.get("activity_review_status", "") == decision["activity_review_status"], poi["id"]


def test_common_sports_nicknames_resolve_to_the_correct_grounds_only():
    pois = load_pois()
    guicao = _exact_master_poi_matches("桂操", pois)
    assert [poi["id"] for poi in guicao] == ["poi_085"]
    assert guicao[0]["name"] == "武汉大学桂园操场"
    assert [poi["id"] for poi in _exact_master_poi_matches("武汉大学足球场", pois)] == ["poi_085"]
    assert [poi["id"] for poi in _exact_master_poi_matches("梅操", pois)] == ["poi_082"]
    assert [poi["id"] for poi in _exact_master_poi_matches("信操", pois)] == ["poi_260"]


def test_osm_cafe_is_not_an_alias_of_the_museum_and_is_searchable_as_coffee():
    pois = _by_id()
    museum = pois["poi_004"]
    cafe = pois["poi_452"]

    assert "珞珈咖啡" not in museum["aliases"]
    assert [poi["id"] for poi in _exact_master_poi_matches(
        "珞珈咖啡", list(pois.values()))] == ["poi_452"]
    assert cafe["subcategory"] == "coffee"
    assert "coffee" in cafe["activities"]
    assert any(source.get("id") == "node/13246222238"
               for source in cafe["source_refs"])


def test_manual_renames_keep_specific_former_names_on_the_same_poi():
    pois = list(_by_id().values())
    for query, poi_id in [
        ("湖滨十三舍", "poi_135"),
        ("工学部清真食堂", "poi_202"),
        ("武汉大学医院一门诊部", "poi_426"),
        ("泥沙实验楼", "poi_437"),
        ("高压大厅", "poi_439"),
        ("珞珈咖啡", "poi_452"),
    ]:
        assert [poi["id"] for poi in _exact_master_poi_matches(query, pois)] == [poi_id]


def test_manual_canteen_name_has_one_owner_without_removing_other_facilities():
    pois = list(_by_id().values())
    for query in ("工学部一食堂", "工学部1食堂"):
        assert [poi["id"] for poi in _exact_master_poi_matches(query, pois)] == ["poi_202"]
    assert [poi["id"] for poi in _exact_master_poi_matches(
        "武汉大学工学部学生一食堂", pois)] == ["poi_203"]


def test_named_table_tennis_and_badminton_osm_candidates_have_audited_decisions():
    pois = _by_id()
    by_source = {
        source.get("id"): poi
        for poi in pois.values()
        for source in poi.get("source_refs", [])
        if source.get("source") == "OpenStreetMap"
    }
    assert by_source["node/2177869899"]["activities"] == ["table_tennis"]
    assert by_source["node/10742578168"]["activities"] == ["badminton"]
    review = json.loads(Path("data/campus_review_decisions.json").read_text(encoding="utf-8"))
    decisions = {row.get("source_id"): row for row in review["poi_decisions"]}
    assert decisions["node/2177869899"]["decision"] != "pending_review"
    assert decisions["node/10742578168"]["decision"] != "pending_review"


def test_unverified_sports_facilities_are_not_misrepresented_as_specific_activities():
    pois = _by_id()

    for poi_id in ("poi_258", "poi_445", "poi_448"):
        assert pois[poi_id]["activities"] == []
        assert pois[poi_id]["activity_review_status"] == "reviewed_no_supported_activity"
        assert pois[poi_id]["activity_basis"].startswith("reviewed:")


def test_official_informatics_track_source_supports_its_alias_and_field_uses():
    pois = _by_id()
    track = pois["poi_260"]

    assert track["activities"] == ["running", "football", "volleyball"]
    assert "竹园操场" in track["aliases"]
    assert "https://zzgl.whu.edu.cn/info/1017/1030.htm" in track["activity_basis"]
    assert matches_activities(track, ["football"])
    assert matches_activities(track, ["volleyball"])
    assert not matches_activities(track, ["basketball"])


def test_official_star_lake_gym_source_supports_badminton():
    gym = _by_id()["poi_257"]

    assert gym["activities"] == ["fitness", "badminton"]
    assert "https://zzgl.whu.edu.cn/info/1017/1030.htm" in gym["activity_basis"]
    assert "gym" in ACTIVITY_SUBCATEGORIES["badminton"]
    assert matches_activities(gym, ["badminton"])


def test_official_fengyu_and_engineering_gym_sources_support_indoor_courts():
    pois = _by_id()
    fengyu = pois["poi_086"]
    engineering = pois["poi_204"]

    assert fengyu["activities"] == ["fitness", "badminton", "table_tennis"]
    assert "https://zzgl.whu.edu.cn/info/1017/1028.htm" in fengyu["activity_basis"]
    assert engineering["activities"] == [
        "fitness", "badminton", "table_tennis", "volleyball",
    ]
    assert "https://zzgl.whu.edu.cn/info/1017/1029.htm" in engineering["activity_basis"]
    assert "gym" in ACTIVITY_SUBCATEGORIES["table_tennis"]
    assert "gym" in ACTIVITY_SUBCATEGORIES["volleyball"]
    assert matches_activities(fengyu, ["badminton", "table_tennis"])
    assert matches_activities(engineering, ["volleyball"])
