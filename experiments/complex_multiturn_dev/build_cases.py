"""Build the human-authored, exposed development task set, never a holdout."""
from __future__ import annotations

import json
from pathlib import Path


def check(field, value, op="eq"):
    return {"field": field, "op": op, "value": value}


def turn(user, *checks, event_stage=None, action=None):
    result = {"user": user, "checks": list(checks)}
    if event_stage is not None:
        result["event_stage"] = event_stage
    if action is not None:
        result["action"] = action
    return result


def case(group, number, title, turns, *, phase="core", event_scenario=None):
    result = {"id": f"{group}_{number:02d}", "group": group, "phase": phase,
              "title": title, "turns": turns,
              "basis": "固定校园地点目录与图内证据；地点营业及现场通行未核实"}
    if event_scenario:
        result["event_scenario"] = event_scenario
        result["basis"] += "；道路事件为隔离模拟，不代表校园真实路况"
    return result


O = lambda x: check("origin", x)
A = lambda x: check("search_anchor", x)
M = lambda x: check("mode", x)
S = lambda x: check("avoid_steps", x)
R = lambda x: check("radius_m", x)
T = lambda *x: check("stops", list(x), "ordered_contains")
E = lambda x: check("event_state", x)
Q = lambda x: check("clarification_slot", x)


def build():
    cases = [
        case("search", 1, "更换地点用途并扩大搜索半径", [
            turn("我在卓尔体育馆，想找万林艺术博物馆附近的咖啡店，之后步行过去，避开台阶。先推荐几个。", O("poi_418"), A("poi_004"), M("walk"), S(True)),
            turn("先不喝咖啡了，改成附近吃饭的地方。", O("poi_418"), A("poi_004"), M("walk"), S(True)),
            turn("搜索范围扩大到两公里。", O("poi_418"), A("poi_004"), R(2000), S(True))]),
        case("search", 2, "更换球类并保留起点", [
            turn("我从信息学部快递服务中心步行出发，找附近的篮球场，别走台阶。", O("poi_424"), M("walk"), S(True)),
            turn("不打篮球了，帮我找羽毛球场。", O("poi_424"), M("walk"), S(True)),
            turn("范围设为一公里。", O("poi_424"), R(1000), S(True))]),
        case("search", 3, "缩小范围后的空候选", [
            turn("从工学部社区卫生服务中心出发，找附近羽毛球场，先按一公里范围。", O("poi_426"), R(1000)),
            turn("范围缩到100米。", O("poi_426"), R(100)),
            turn("如果没有符合范围的，就直接告诉我。", O("poi_426"), R(100))]),
        case("search", 4, "条件不变时换一批", [
            turn("我在万林艺术博物馆，找两公里内的篮球场，步行。", O("poi_004"), R(2000), M("walk")),
            turn("换一批看看。", O("poi_004"), R(2000), M("walk")),
            turn("不要重复刚才展示过的。", O("poi_004"), R(2000), M("walk"))]),
        case("search", 5, "修改搜索中心但不修改起点", [
            turn("我在珞珈门，帮我找万林艺术博物馆附近的咖啡店，走路过去。", O("poi_140"), A("poi_004"), M("walk")),
            turn("改成在星湖体育馆附近找。", O("poi_140"), A("poi_257"), M("walk")),
            turn("范围扩大到两公里。", O("poi_140"), A("poi_257"), R(2000))]),
        case("search", 6, "选择已展示候选并续接", [
            turn("我在信息学部快递服务中心，推荐附近可以打篮球的地方，我步行去。", O("poi_424"), M("walk")),
            turn("范围扩大到两公里。", O("poi_424"), R(2000)),
            turn("去第一个。", O("poi_424"), check("selected_matches_first", True), action="select_first_candidate")]),
        case("itinerary", 1, "替换中间一站", [
            turn("从珞珈门步行出发，先去信息学部快递服务中心，再去星湖体育馆，最后去星湖园食堂，避开台阶。", O("poi_140"), M("walk"), S(True), T("poi_424", "poi_257", "poi_307")),
            turn("体育馆换成信息学部篮球场。", O("poi_140"), S(True), T("poi_424", "poi_262", "poi_307")),
            turn("就按这个顺序走。", O("poi_140"), S(True), T("poi_424", "poi_262", "poi_307"))]),
        case("itinerary", 2, "删除首个活动", [
            turn("我从卓尔体育馆步行出发，先去梅园快递服务中心，再去万林艺术博物馆，最后到凌波门。", O("poi_418"), M("walk"), T("poi_423", "poi_004", "poi_137")),
            turn("快递先不取了。", O("poi_418"), T("poi_004", "poi_137"), check("stops", "poi_423", "not_contains")),
            turn("给我更新后的路线。", O("poi_418"), T("poi_004", "poi_137"))]),
        case("itinerary", 3, "交换目的地顺序", [
            turn("从工学部社区卫生服务中心走路出发，先去工学部网球场，再去凌波门。", O("poi_426"), M("walk"), T("poi_084", "poi_137")),
            turn("顺序反过来，先去凌波门。", O("poi_426"), T("poi_137", "poi_084")),
            turn("继续步行。", O("poi_426"), M("walk"), T("poi_137", "poi_084"))]),
        case("itinerary", 4, "插入途中站点", [
            turn("我从珞珈门步行去凌波门。", O("poi_140"), M("walk"), T("poi_137")),
            turn("中途先去万林艺术博物馆。", O("poi_140"), T("poi_004", "poi_137")),
            turn("再在博物馆后面加上卓尔体育馆。", O("poi_140"), T("poi_004", "poi_418", "poi_137"))]),
        case("itinerary", 5, "更换出行方式但保留站点", [
            turn("从信息学部快递服务中心出发，步行先去星湖体育馆，再去星湖园食堂。", O("poi_424"), M("walk"), T("poi_257", "poi_307")),
            turn("改成骑车。", O("poi_424"), M("bike"), T("poi_257", "poi_307")),
            turn("先后顺序就这样。", O("poi_424"), M("bike"), T("poi_257", "poi_307"))]),
        case("itinerary", 6, "已完成的活动不再重复", [
            turn("从珞珈门步行，先去梅园快递服务中心，再去万林艺术博物馆。", O("poi_140"), M("walk"), T("poi_423", "poi_004")),
            turn("快递已经取完了，我现在就在梅园快递服务中心。", O("poi_423"), M("walk"), T("poi_004")),
            turn("接下来去博物馆。", O("poi_423"), T("poi_004"))]),
        case("clarification", 1, "缺少实际起点", [
            turn("我想先去工学部网球场，再去凌波门，帮我安排步行路线。", Q("start")),
            turn("我现在在工学部社区卫生服务中心。", O("poi_426"), M("walk"), T("poi_084", "poi_137")),
            turn("顺序反过来，先去凌波门。", O("poi_426"), T("poi_137", "poi_084"))]),
        case("clarification", 2, "补充模糊运动类型", [
            turn("从星湖园食堂出发，我想找地方打球，步行过去。", O("poi_307"), M("walk")),
            turn("打羽毛球。", O("poi_307"), M("walk")),
            turn("找一公里范围内的。", O("poi_307"), R(1000))]),
        case("clarification", 3, "补充场馆校区", [
            turn("我想去体育馆，从珞珈门步行出发。", O("poi_140"), M("walk")),
            turn("我指的是信息学部的星湖体育馆。", O("poi_140"), T("poi_257")),
            turn("尽量走最短的路线。", O("poi_140"), T("poi_257"))]),
        case("clarification", 4, "明确要求独立景点间路线", [
            turn("只看万林艺术博物馆到凌波门的步行参考路线，不用考虑我从哪里出发。", M("walk")),
            turn("中间加上卓尔体育馆。", M("walk"), T("poi_418", "poi_137")),
            turn("按最短距离。", M("walk"), T("poi_418", "poi_137"))]),
        case("clarification", 5, "只撤销避台阶限制", [
            turn("从珞珈门步行到万林艺术博物馆，避开台阶。", O("poi_140"), M("walk"), S(True)),
            turn("台阶现在可以走了。", O("poi_140"), M("walk"), S(False)),
            turn("还是去博物馆，按最短距离。", O("poi_140"), M("walk"), S(False))]),
        case("clarification", 6, "切换任务后恢复原行程", [
            turn("从卓尔体育馆步行去万林艺术博物馆，避开台阶。", O("poi_418"), M("walk"), S(True)),
            turn("先不改这个行程，告诉我珞珈门在哪里。", check("status", "completed")),
            turn("回到刚才去博物馆的路线。", O("poi_418"), M("walk"), S(True))]),
        case("events", 1, "原路线封闭后绕行", [
            turn("从卓尔体育馆步行去万林艺术博物馆，走最短路。", O("poi_418"), M("walk"), event_stage="before"),
            turn("按当前路况更新这条路。", O("poi_418"), M("walk"), E("active"), event_stage="active"),
            turn("再核对一次当前路线。", O("poi_418"), M("walk"), E("active"), event_stage="active")], event_scenario="detour"),
        case("events", 2, "封闭不可达后撤销", [
            turn("从梅园综合楼步行去梅园快递服务中心。", O("poi_001"), M("walk"), event_stage="before"),
            turn("当前路况下还能走通吗？", O("poi_001"), M("walk"), E("active"), event_stage="active"),
            turn("根据最新路况再规划。", O("poi_001"), M("walk"), E("revoked"), event_stage="revoked")], event_scenario="blocked"),
        case("events", 3, "改骑行并遇到骑行限制", [
            turn("从枫园食堂步行去万林艺术博物馆，按最短距离。", O("poi_072"), M("walk"), event_stage="before"),
            turn("改成骑车，按当前路况走。", O("poi_072"), M("bike"), E("active"), event_stage="active"),
            turn("现在的路况再更新一下。", O("poi_072"), M("bike"), E("expired"), event_stage="expired")], event_scenario="bike_only"),
        case("events", 4, "单方向限制", [
            turn("从珞珈门步行去卓尔体育馆。", O("poi_140"), M("walk"), event_stage="before"),
            turn("结合当前管制更新路线。", O("poi_140"), M("walk"), E("active"), event_stage="active"),
            turn("现在反过来，从卓尔体育馆去珞珈门。", O("poi_418"), M("walk"), E("active"), event_stage="active")], event_scenario="direction"),
        case("events", 5, "无关道路事件", [
            turn("从信息学部快递服务中心步行去星湖体育馆。", O("poi_424"), M("walk"), event_stage="before"),
            turn("有新路况，看看我的路线需不需要变。", O("poi_424"), M("walk"), E("active"), event_stage="active"),
            turn("重新核对这条步行路线。", O("poi_424"), M("walk"), E("active"), event_stage="active")], event_scenario="off_path"),
        case("events", 6, "事件生效前后及到期", [
            turn("从卓尔体育馆步行到万林艺术博物馆。", O("poi_418"), M("walk"), E("before"), event_stage="before"),
            turn("现在路况有变化吗？", O("poi_418"), M("walk"), E("active"), event_stage="active"),
            turn("过一会儿再按当前路况算。", O("poi_418"), M("walk"), E("expired"), event_stage="expired")], event_scenario="lifecycle"),
        case("search", 7, "变体：起点与远处搜索中心", [
            turn("我人还在工学部社区卫生服务中心，想看看星湖体育馆附近有什么篮球场，步行。", O("poi_426"), A("poi_257"), M("walk")),
            turn("改找羽毛球场。", O("poi_426"), A("poi_257"), M("walk")),
            turn("以体育馆为中心，半径两公里。", O("poi_426"), A("poi_257"), R(2000))], phase="variant"),
        case("search", 8, "变体：选点后修改路线偏好", [
            turn("我在梅园快递服务中心，推荐附近能喝咖啡的地方，走路去。", O("poi_423"), M("walk")),
            turn("范围放大到两公里。", O("poi_423"), R(2000)),
            turn("就去第一个。", O("poi_423"), check("selected_matches_first", True), action="select_first_candidate"),
            turn("这条路改走最短距离。", O("poi_423"), M("walk"))], phase="variant"),
        case("itinerary", 7, "变体：替换后再删除", [
            turn("从凌波门走路先到工学部网球场，再到工学部篮球场，最后去珞珈门。", O("poi_137"), M("walk"), T("poi_084", "poi_206", "poi_140")),
            turn("篮球场换成工学部排球场。", O("poi_137"), T("poi_084", "poi_432", "poi_140")),
            turn("网球场不去了。", O("poi_137"), T("poi_432", "poi_140"), check("stops", "poi_084", "not_contains"))], phase="variant"),
        case("itinerary", 8, "变体：插入站点并调整方式", [
            turn("从信息学部快递服务中心到星湖园食堂，步行。", O("poi_424"), M("walk"), T("poi_307")),
            turn("先到星湖体育馆。", O("poi_424"), T("poi_257", "poi_307")),
            turn("骑车去。", O("poi_424"), M("bike"), T("poi_257", "poi_307"))], phase="variant"),
        case("clarification", 7, "变体：先补起点后撤销约束", [
            turn("带我步行去万林艺术博物馆，再去凌波门，别走台阶。", Q("start")),
            turn("从卓尔体育馆出发。", O("poi_418"), M("walk"), S(True), T("poi_004", "poi_137")),
            turn("台阶没关系了。", O("poi_418"), M("walk"), S(False), T("poi_004", "poi_137"))], phase="variant"),
        case("clarification", 8, "变体：无起点的独立参考路线", [
            turn("不考虑我的位置，只给珞珈门到万林艺术博物馆的步行参考路线。", M("walk")),
            turn("途经卓尔体育馆。", M("walk"), T("poi_418", "poi_004")),
            turn("再按最短距离看看。", M("walk"), T("poi_418", "poi_004"))], phase="variant"),
        case("events", 7, "变体：不适用方式的事件", [
            turn("从卓尔体育馆步行去万林艺术博物馆。", O("poi_418"), M("walk"), event_stage="before"),
            turn("按当前路况确认步行路线。", O("poi_418"), M("walk"), E("active"), event_stage="active"),
            turn("现在改成骑车。", O("poi_418"), M("bike"), E("active"), event_stage="active")], phase="variant", event_scenario="bike_only"),
        case("events", 8, "变体：封闭恢复后继续原需求", [
            turn("我从珞珈门步行到卓尔体育馆，避开台阶。", O("poi_140"), M("walk"), S(True), event_stage="before"),
            turn("路况变了，帮我重新规划。", O("poi_140"), M("walk"), S(True), E("active"), event_stage="active"),
            turn("按现在的路况再算一遍。", O("poi_140"), M("walk"), S(True), E("revoked"), event_stage="revoked")], phase="variant", event_scenario="detour"),
    ]
    return {"version": "complex_multiturn_dev_v1", "scope": "development_only",
            "formal_test": False, "variant_exposure": "run_after_core_fixes",
            "cases": cases}


def main():
    import argparse
    from experiments.complex_multiturn_dev.contracts import validate_dataset

    default = Path(__file__).resolve().parents[1] / "datasets/complex_multiturn_dev_v1/cases.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default)
    target = parser.parse_args().output
    data = build()
    validate_dataset(data)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise FileExistsError("dataset already exists with different content")
    if not target.exists():
        target.write_text(content, encoding="utf-8")
    print(f"{len(data['cases'])} development cases: {target}")


if __name__ == "__main__":
    main()
