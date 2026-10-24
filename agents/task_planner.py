"""Small deterministic intent planner for dispatching independent task tools.

This planner only identifies high-confidence, bounded requirements. It does not
replace the language agent that resolves ambiguous place names or route details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re


@dataclass(frozen=True)
class TaskRequirement:
    requirement_id: str
    description: str


@dataclass(frozen=True)
class PlannedNode:
    node_id: str
    tool: str
    requirement_ids: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlan:
    kind: Literal["route", "weather", "composite", "agent"]
    requirements: tuple[TaskRequirement, ...]
    nodes: tuple[PlannedNode, ...]


_WEATHER_INTENT = re.compile(r"天气|气温|下雨|降雨|带伞|刮风|风大|适合出门|能见度")
_ROUTE_INTENT = re.compile(
    r"(?:从.{1,40}(?:到|去)|(?:途经|经过|经由).{1,40}(?:到|去)|^经.{1,40}(?:到|去)|"
    r"路线|怎么走|如何去|带我去|带我过去|导航到|去.{1,30}(?:门|楼|馆|食堂|图书馆|宿舍|樱顶|珞珈))"
)
_OTHER_TOOL_INTENT = re.compile(
    r"推荐|找(?:一下|一个|家|到)?|搜(?:一下|索)|哪里有|哪家|"
    r"查询.{0,15}(?:地点|餐厅|食堂|咖啡|奶茶|厕所|图书馆|体育馆|路况)|"
    r"附近.{0,15}(?:食堂|餐厅|咖啡|奶茶|吃|喝|洗手间|厕所|图书馆|体育馆|运动场|景点)|"
    r"路况|封路|施工|积水|事故|管制|能不能走|可不可以走"
)


def build_task_plan(query: str, *, has_route_context: bool = False) -> TaskPlan:
    """Dispatch explicit weather and route asks without losing either request."""
    text = str(query or "").strip()
    wants_weather = bool(_WEATHER_INTENT.search(text))
    wants_route = has_route_context or bool(_ROUTE_INTENT.search(text))
    wants_other_tool = bool(_OTHER_TOOL_INTENT.search(text))
    requirements: list[TaskRequirement] = []
    nodes: list[PlannedNode] = []
    # Parallelize only the well-defined route + weather pair. When a message
    # combines weather with another tool domain, hand the complete request to
    # the tool-calling agent so a weather shortcut cannot drop the other ask.
    if wants_route and wants_weather and not wants_other_tool:
        requirements.append(TaskRequirement("route", "完成用户请求的路线规划"))
        nodes.append(PlannedNode("route-agent", "route_agent", ("route",)))
        requirements.append(TaskRequirement("weather", "查询用户请求的当前天气"))
        nodes.append(PlannedNode("weather", "get_weather", ("weather",)))
        kind: Literal["route", "weather", "composite", "agent"] = "composite"
    elif wants_weather and not (wants_route or wants_other_tool):
        kind = "weather"
        requirements.append(TaskRequirement("weather", "查询用户请求的当前天气"))
        nodes.append(PlannedNode("weather", "get_weather", ("weather",)))
    elif wants_route or wants_other_tool:
        kind = "route" if wants_route else "agent"
        requirement_id = "route" if wants_route else "answer"
        description = "完成用户请求的路线规划" if wants_route else "回答用户的全部信息与工具请求"
        requirements.append(TaskRequirement(requirement_id, description))
        nodes.append(PlannedNode("route-agent", "route_agent", (requirement_id,)))
    else:
        kind = "agent"
        requirements.append(TaskRequirement("answer", "回答用户请求"))
        nodes.append(PlannedNode("route-agent", "route_agent", ("answer",)))
    return TaskPlan(kind, tuple(requirements), tuple(nodes))
