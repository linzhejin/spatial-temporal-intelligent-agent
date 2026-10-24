# 2026-10-25 T2 场地轮廓与末端接驳诊断

本记录推进 P1 的候选地点来源审查，不产生正式 Agent 成功率。输入为已见的 24 道 T2 编题、现有 443 个 POI、静态路网及 2026-09-24 命名的湖北 OSM 快照。图内排序报告仍是开发参考，实地入口没有独立验明。

## 复核方法与来源

从既有 `hubei-260924.osm.pbf` 的 libosmium 完整面要素提取 4 个与 T2 相关的场地轮廓，保存为 `experiments/source/osm_t2_area_evidence_20260924.geojson`（含 OSM ID、来源哈希及 ODbL 署名）。用 WGS-84 面轮廓与可步行路网节点投影到 UTM 50N 后计算距离；POI 原坐标只在计算边界由 GCJ-02 转换一次。脚本 `scripts/analysis/audit_t2_access_geometry.py` 严格核对 T2 不确定性报告、POI 和路网的来源哈希。

轮廓内有路网节点，**只能证明两份数据的几何重叠**。它不能证明围栏开口、开放时间、公共通行权或入口到节点的实际路径。4 个轮廓和路网都可能来自 OSM，同源数据也不能冒充独立现场真值。

| POI | 名称 | 点到可步行节点 | 源面到可步行节点 | 结果 |
|---|---|---:|---:|---|
| `poi_080` | 武汉大学九一二运动场 | 51.36 m | 本次来源匹配未命中完整面 | 保留未知 |
| `poi_194` | 武汉大学法学图书馆 | 30.87 m | 本次来源匹配未命中完整面 | 保留未知 |
| `poi_205` | 工学部操场 | 70.99 m | 0 m；面内 3 节点 | 中心点偏移可能夸大接驳；入口仍未知 |
| `poi_206` | 工学部篮球场 | 38.08 m | 0 m；面内 4 节点 | 同上 |
| `poi_260` | 武汉大学信息学部操场 | 45.94 m | 0 m；面内 5 节点 | OSM 面带 `barrier=fence`，尤其不能据几何重叠推断入口 |
| `poi_387` | 小食坊粉丝煲（工学部集贸市场） | 34.17 m | 所在市场面到节点 3.79 m | 店面入口未知；学部字段冲突 |
| `poi_388` | 桂圆馄饨（工学部集贸市场） | 31.75 m | 所在市场面到节点 3.79 m | 店面入口未知；学部字段冲突 |

最后两家店的 GCJ-02 坐标转换后落在 OSM `way/153741407` 工学部集贸市场面内，而主库把 `campus` 标为“文理学部”；相应名称、市场 POI 与 OSM 工学部边界也指向工学部。这是**待修正的主数据冲突**，不是本轮入口证据。变更主库前需对同一边界附近的其他地点作一致审查，并在变更后重算所有依赖 POI 哈希的开发参考，不能让旧参考继续冒充当前版本。

## 判分边界与下一步

T2 原先 24 题中 15 题可按当前图内规则判候选前四名，9 题因末端接驳保持 `unknown`。本次新增轮廓证据只解释了 5 个地点的中心点偏移可能性，**9 题均未从未知改为通过**。后续优先用明确入口或独立资料补这 7 个地点；缺证据的题应在正式结果中单列未知分母和原因，不拿中心点距离或 OSM 面重叠制造“现场可达”。

运行：

```powershell
python -m scripts.analysis.audit_t2_access_geometry --areas experiments/source/osm_t2_area_evidence_20260924.geojson --uncertainty output/experiments/T2_uncertainty_9e5c1e2_20261025.json --output output/experiments/T2_access_geometry_20261025.json
python -m pytest tests/test_audit_t2_access_geometry.py tests/test_audit_t2_candidate_uncertainty.py tests/test_candidate_reference.py -q --junitxml=output/experiments/p1_t2_geometry_diagnostic_20261025.xml
```

诊断 JSON SHA-256 `a312443073197b21706da199c6db2371285f3fe102330323855fd2f551028369`；来源面 GeoJSON SHA-256 `3a8772abd6fd03ca649eb4b3fdb36d4303d83c875f99d1557ddf1c996aae2bb0`；7 项聚焦测试通过，JUnit SHA-256 `dc47baae334c764fe7eed9fd87c6660fa018e4e8ef43f727f8ad39a782346206`。测试证明诊断实现与现有候选参考回归通过，不证明入口、路况或 Agent 实际推荐正确。
