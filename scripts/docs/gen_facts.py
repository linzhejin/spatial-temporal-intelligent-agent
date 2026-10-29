#!/usr/bin/env python3
"""生成/校验项目当前事实（docs/当前事实.md 与 README/HANDOFF/CLAUDE 标记块）。

用法（建议用项目 venv 运行）：
    python scripts/docs/gen_facts.py            # 重新生成事实文档并刷新标记块
    python scripts/docs/gen_facts.py --check    # 仅校验稳定字段；不一致时退出码 1

设计：
- 稳定字段（工具/POI/校门/路网/边覆盖/SW/APK/迁移）参与 --check；
- git 信息（分支、提交、远程差距）与生成时间仅作信息性展示，不参与校验，
  否则每次 commit 都会导致校验变红。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "当前事实.md"
MARKER_TARGETS = (ROOT / "README.md", ROOT / "HANDOFF.md", ROOT / "CLAUDE.md")
FACTS_BEGIN = "<!-- FACTS:BEGIN -->"
FACTS_END = "<!-- FACTS:END -->"
SHANGHAI = timezone(timedelta(hours=8))

# (key, 表格事实名, 来源说明)；key 同时用于标记块正则的比对
STABLE_FIELDS = (
    ("tools", "Agent 工具数", "agents/tools.py 的 TOOL_SCHEMAS"),
    ("pois", "POI 总数", "data/pois.json 的 pois 数组"),
    ("gates", "校门数", 'data/pois.json 中 type=="gate" 的条目'),
    ("nodes", "路网节点数", "data/whu_road_network.graphml"),
    ("edges", "路网有向边数", "data/whu_road_network.graphml"),
    ("overrides", "人工边覆盖条数", "data/edge_overrides.json 的 edges"),
    ("sw", "前端缓存名", "static/sw.js 的 CACHE_NAME"),
    ("ver_name", "Android 版本名", "android-app/build.ps1 的 $verName"),
    ("ver_code", "Android versionCode", "android-app/build.ps1 的 $verCode"),
    ("migrations", "数据库迁移数", "storage/migrations/ 的 *.sql 数量"),
    ("latest_migration", "最新迁移编号", "storage/migrations/ 最新文件名前缀"),
)

# 标记块内的提取规则：正则 -> 与 group 依次对应的稳定字段 key
MARKER_PATTERNS = (
    (re.compile(r"Agent 工具 (\d+)"), ("tools",)),
    (re.compile(r"POI ([\d,]+)（含校门 ([\d,]+)）"), ("pois", "gates")),
    (re.compile(r"路网 ([\d,]+) 节点 / ([\d,]+) 有向边"), ("nodes", "edges")),
    (re.compile(r"人工边覆盖 ([\d,]+)"), ("overrides",)),
    (re.compile(r"SW ([\w.-]+)"), ("sw",)),
    (re.compile(r"Android ([\d.]+) \(versionCode (\d+)\)"), ("ver_name", "ver_code")),
    (re.compile(r"数据库迁移 (\d+)（最新 (\d+)）"), ("migrations", "latest_migration")),
)


def fmt_int(value: int) -> str:
    return f"{value:,}"


def run_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _count_tool_schemas() -> int:
    """用 AST 静态解析 TOOL_SCHEMAS 列表长度，避免 import agents.tools 的副作用。"""
    source = (ROOT / "agents" / "tools.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "TOOL_SCHEMAS" for target in node.targets
        ):
            if isinstance(node.value, ast.List):
                return len(node.value.elts)
            raise SystemExit("agents/tools.py: TOOL_SCHEMAS 不是列表字面量，无法静态解析")
    raise SystemExit("agents/tools.py: 未找到 TOOL_SCHEMAS")


def _count_network() -> tuple[int, int]:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise SystemExit(
            "需要 networkx：请使用项目 venv 运行"
            "（venv\\Scripts\\python.exe scripts/docs/gen_facts.py）"
        ) from exc
    graph = nx.read_graphml(ROOT / "data" / "whu_road_network.graphml")
    return graph.number_of_nodes(), graph.number_of_edges()


def collect_stable() -> dict[str, str]:
    """从代码/数据/前端读取全部稳定字段（不访问 git）。"""
    tools = _count_tool_schemas()

    pois_data = json.loads((ROOT / "data" / "pois.json").read_text(encoding="utf-8"))
    pois = pois_data["pois"] if isinstance(pois_data, dict) else pois_data
    gates = sum(1 for item in pois if item.get("type") == "gate")

    nodes, edges = _count_network()

    overrides = json.loads((ROOT / "data" / "edge_overrides.json").read_text(encoding="utf-8"))
    override_count = len(overrides["edges"])

    sw_text = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
    sw_match = re.search(r"CACHE_NAME\s*=\s*['\"]([^'\"]+)['\"]", sw_text)
    if not sw_match:
        raise SystemExit("static/sw.js: 未找到 CACHE_NAME")
    sw_name = sw_match.group(1)

    build_text = (ROOT / "android-app" / "build.ps1").read_text(encoding="utf-8")
    ver_name = re.search(r'\$verName\s*=\s*"([^"]+)"', build_text)
    ver_code = re.search(r"\$verCode\s*=\s*(\d+)", build_text)
    if not ver_name or not ver_code:
        raise SystemExit("android-app/build.ps1: 未找到 $verName / $verCode")

    migrations = sorted((ROOT / "storage" / "migrations").glob("*.sql"))
    if not migrations:
        raise SystemExit("storage/migrations/: 未找到迁移文件")
    latest = max(int(item.name.split("_", 1)[0]) for item in migrations)

    return {
        "tools": str(tools),
        "pois": fmt_int(len(pois)),
        "gates": fmt_int(gates),
        "nodes": fmt_int(nodes),
        "edges": fmt_int(edges),
        "overrides": fmt_int(override_count),
        "sw": sw_name,
        "ver_name": ver_name.group(1),
        "ver_code": ver_code.group(1),
        "migrations": str(len(migrations)),
        "latest_migration": f"{latest:03d}",
    }


def collect_git_facts() -> dict:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "未知"
    commit = run_git(["rev-parse", "--short", "HEAD"]) or "未知"
    gaps: dict[str, str] = {}
    for remote in ("origin", "gitee"):
        ref = f"{remote}/main"
        if run_git(["rev-parse", "--verify", "--quiet", ref]) is None:
            gaps[remote] = "未 fetch"
            continue
        counts = run_git(["rev-list", "--left-right", "--count", f"{ref}...HEAD"])
        if counts is None:
            gaps[remote] = "未 fetch"
            continue
        # 左侧仅存在于远程引用，右侧仅存在于本地 HEAD；两个方向都要检查。
        behind, ahead = map(int, counts.split())
        if behind and ahead:
            gaps[remote] = f"本地落后 {behind}，领先 {ahead}（已分叉）"
        elif behind:
            gaps[remote] = f"本地落后 {behind}"
        elif ahead:
            gaps[remote] = f"本地领先 {ahead}"
        else:
            gaps[remote] = "同步"
    return {"branch": branch, "commit": commit, "gaps": gaps}


def render_facts_doc(stable: dict[str, str], git_facts: dict, generated_at: str) -> str:
    rows = [
        ("分支", git_facts["branch"], "git rev-parse --abbrev-ref HEAD"),
        ("提交", git_facts["commit"], "git rev-parse --short HEAD"),
        ("origin/main 差距", git_facts["gaps"]["origin"],
         "git rev-list --left-right --count origin/main...HEAD"),
        ("gitee/main 差距", git_facts["gaps"]["gitee"],
         "git rev-list --left-right --count gitee/main...HEAD"),
    ]
    rows += [(name, stable[key], source) for key, name, source in STABLE_FIELDS]
    lines = [
        "# 当前事实（自动生成）",
        "",
        "由 `scripts/docs/gen_facts.py` 自动生成，勿手改；git 与生成时间为信息性字段，不参与校验。",
        "",
        f"生成时间：{generated_at}（Asia/Shanghai）",
        "更新命令：`python scripts/docs/gen_facts.py`；校验：`python scripts/docs/gen_facts.py --check`",
        "",
        "| 事实 | 当前值 | 来源 |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {name} | {value} | {source} |" for name, value, source in rows]
    lines.append("")
    return "\n".join(lines)


def render_marker_block(stable: dict[str, str], git_facts: dict, generated_at: str) -> str:
    gaps = git_facts["gaps"]
    lines = [
        FACTS_BEGIN,
        "（以下为自动生成区，勿手改；更新：python scripts/docs/gen_facts.py）",
        f"- 版本：{git_facts['branch']} @ {git_facts['commit']} ｜ origin/main {gaps['origin']}"
        f" ｜ gitee/main {gaps['gitee']} ｜ 生成于 {generated_at}",
        f"- Agent 工具 {stable['tools']} ｜ POI {stable['pois']}（含校门 {stable['gates']}）"
        f" ｜ 路网 {stable['nodes']} 节点 / {stable['edges']} 有向边"
        f" ｜ 人工边覆盖 {stable['overrides']}",
        f"- 前端 SW {stable['sw']} ｜ Android {stable['ver_name']} "
        f"(versionCode {stable['ver_code']}) ｜ 数据库迁移 {stable['migrations']}"
        f"（最新 {stable['latest_migration']}）",
        FACTS_END,
    ]
    return "\n".join(lines)


def update_marker_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if FACTS_BEGIN not in text or FACTS_END not in text:
        raise SystemExit(
            f"{path.name}: 缺少 {FACTS_BEGIN} / {FACTS_END} 标记，请先插入标记块"
        )
    pattern = re.compile(re.escape(FACTS_BEGIN) + r".*?" + re.escape(FACTS_END), re.S)
    path.write_text(pattern.sub(block, text, count=1), encoding="utf-8", newline="\n")


def read_marker_block(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    begin = text.find(FACTS_BEGIN)
    end = text.find(FACTS_END)
    if begin == -1 or end == -1 or end < begin:
        return None
    return text[begin:end + len(FACTS_END)]


def parse_doc_values(text: str) -> dict[str, str]:
    names = {name for _, name, _ in STABLE_FIELDS}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] in names:
            values[cells[0]] = cells[1]
    return values


def check_all(stable: dict[str, str]) -> list[str]:
    failures: list[str] = []
    name_of = {key: name for key, name, _ in STABLE_FIELDS}

    if not DOC_PATH.exists():
        failures.append("docs/当前事实.md: 文件不存在（运行 python scripts/docs/gen_facts.py 生成）")
    else:
        doc_values = parse_doc_values(DOC_PATH.read_text(encoding="utf-8"))
        for key, name, _ in STABLE_FIELDS:
            current = doc_values.get(name)
            expect = stable[key]
            if current is None:
                failures.append(f"docs/当前事实.md -> {name}: 缺失（期望 {expect}）")
            elif current != expect:
                failures.append(f"docs/当前事实.md -> {name}: 现有 {current} / 期望 {expect}")

    for path in MARKER_TARGETS:
        block = read_marker_block(path)
        if block is None:
            failures.append(f"{path.name}: 缺少标记块 {FACTS_BEGIN} / {FACTS_END}")
            continue
        for pattern, keys in MARKER_PATTERNS:
            match = pattern.search(block)
            if not match:
                failures.append(f"{path.name}: 标记块内未匹配到 {pattern.pattern}")
                continue
            for key, current in zip(keys, match.groups()):
                expect = stable[key]
                if current != expect:
                    failures.append(f"{path.name} -> {name_of[key]}: 现有 {current} / 期望 {expect}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="生成/校验项目当前事实")
    parser.add_argument("--check", action="store_true", help="只校验稳定字段，不一致时退出码 1")
    args = parser.parse_args()

    stable = collect_stable()

    if args.check:
        failures = check_all(stable)
        if failures:
            print("文档事实校验未通过：")
            for item in failures:
                print(f"  [不一致] {item}")
            print("请运行 python scripts/docs/gen_facts.py 更新后重试。")
            return 1
        total = len(STABLE_FIELDS) * (len(MARKER_TARGETS) + 1)
        print(
            f"文档事实校验通过：{len(STABLE_FIELDS)} 个稳定字段"
            f" x {len(MARKER_TARGETS) + 1} 个文件，共 {total} 项一致。"
        )
        return 0

    git_facts = collect_git_facts()
    generated_at = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M")
    block = render_marker_block(stable, git_facts, generated_at)
    for path in MARKER_TARGETS:
        update_marker_block(path, block)
    DOC_PATH.write_text(
        render_facts_doc(stable, git_facts, generated_at), encoding="utf-8", newline="\n"
    )
    print(f"已生成 {DOC_PATH.relative_to(ROOT)}")
    for path in MARKER_TARGETS:
        print(f"已更新标记块 {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
