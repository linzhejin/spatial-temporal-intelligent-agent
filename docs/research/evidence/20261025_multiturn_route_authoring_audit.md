# 2026-10-25 T3–T5 多轮路线编题审计

## 本次处理

本轮延续《论文实验推进方案》的 P1 样本与评分工作。目标是给**已见的可见编题草稿**补齐独立图内参考，逐轮检查起点、途经点和终点顺序；没有启动正式对照或写入论文成绩。

首次对 T3–T5 的 72 个路线轮次计算参考时，发现 `v2h_T4_change_start_2` 第二轮被编成 `poi_016 → poi_016`，图内距离为零。根因是工学部空间锚点中的第五个终点与该家族固定图书馆起点相同。生成器将这个终点改为 `poi_202`，重建可见草稿；新增回归检查拒绝起终点相同的路线，并使审计器在遇到这种编题时明确报错。受影响的 T1、T2、T3、T4、T5 草稿来源字段和相关路线已随生成器一并更新。旧草稿哈希下的参考报告只作历史记录，不能用于新版草稿判分。

新增 `scripts/analysis/audit_multiturn_route_references_v2.py`：只接受带来源哈希的可见多样化编题草稿，读取独立的当前 POI 和按方式过滤的路网，对 T3–T5 中 `kind=route` 的每一轮计算分段图内路线。输出保留题号、轮次、有序地点、方式、约束、每段距离和来源哈希；来源中途变化或输出文件已存在时拒绝写入。这个审计不调用产品 Agent 或产品算路器。

## 核对结果与文件

新草稿 120 题中，T3 有 3 个、T4 有 45 个、T5 有 24 个路线轮次，共 72 个。独立图内参考显示 72 个均连通，零长度 0 个，图内最短 93.97 米、最长 2319.13 米；地点到对应方式可导航节点的最大吸附距离为 28.64 米，按本次 30 米编题阈值没有超限。这些数字只检查草稿的图内可计算性。未审计普通文字、澄清或天气轮次的语义金标准。

| 本机文件（`output/experiments/` 下的报告未纳入 Git） | SHA-256 |
|---|---|
| `experiments/datasets/scenarios_test_v2_diverse_draft.json` | `1a66848fad5675169cc702e56ccfd96594482b992e31188aa33203e221dd4c46` |
| `T3_T5_route_reference_20261025_after_anchor_fix.json` | `cadf1c732c1c798483bff7cc6785029bf3bdf20e7a3c86497828cc0bf9e32977` |
| `split_audit_20261025_after_anchor_fix.json` | `fc3d3dac7c63bfd23a4d8c381597503c09347c4412221e588f030e95df232a7c` |
| `T1_route_reference_20261025_after_anchor_fix.json` | `46ec2560d4f7b282ce1caf7fcee79623ad0b66ab72ee86caaab1093d66b73665` |
| `T2_candidate_reference_20261025_after_anchor_fix.json` | `45155408c2e0893f2b893890826a9853d70e1cbc079f1713c4ca01f2823890d7` |
| `T2_uncertainty_20261025_after_anchor_fix.json` | `77b0e5c58d82339b6466f451c0022e3f09daea7478003713c07b55687990cf62c` |
| `T5_composite_reference_20261025_after_anchor_fix.json` | `01a00f0561ca9bd223ef732c1fd1adf8dc45c804cac96f8e3ce2488029af2a13` |

T1 参考在 24 道草稿上图内连通；T2 候选参考重建后，独立前四名审计仍为 15 道图内可判、9 道末端接驳未知；T5 的 9 条组合草稿在选点后均有图内完整路线。拆分报告对指定字段给出 `leakage_free=true`，但 `formal_ready=false`，仍有 7 类阻碍，包括未裁决、未冻结和正式快照闸门未核验。已见草稿不能充作未见测试集。

## 复现与验收边界

在本研究工作树执行：

```powershell
python scripts/analysis/audit_multiturn_route_references_v2.py --output output/experiments/T3_T5_route_reference_20261025_after_anchor_fix.json
python -m pytest tests/test_build_t5_composite_references_v2.py tests/test_build_scenario_v2_development.py tests/test_build_scenario_v2_authoring.py tests/test_audit_t2_candidate_uncertainty.py tests/test_multiturn_route_reference_audit.py tests/test_route_reference.py tests/test_scenario_split_audit.py -q
python -m ruff check scripts/analysis/audit_multiturn_route_references_v2.py scripts/analysis/build_scenario_v2_authoring.py tests/test_multiturn_route_reference_audit.py tests/test_build_scenario_v2_authoring.py
```

报告输出使用独占创建；重新运行须指定新的输出文件名。此次聚焦测试 **24/24 通过**，Ruff 检查通过，`git diff --check` 未发现空白错误。路网参考仍依赖现有图、POI 点位和近邻吸附，不能证明现场可通行、建筑出入口可靠，也没有验证路线的风景或平坦最优性。下一步继续给 T3–T5 建立逐轮语义判分规则，并把已见草稿与独立留出集分开。
