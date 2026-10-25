## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-25
- Verification Status: ANALYZED
- Version Label: p1_t5_visible_real_model_pilot_v1

# T5 两条已见题：真实模型开发试跑与独立路网审计

## 范围与方法

从 120 条**已见、未裁决**的编题草稿中抽取两条 T5 单轮题，分别复制为独立的开发输入，并将 `split` 标为 `development`。输入未改题目内容。通过 `scripts/analysis/probe_product_live_case.py` 调用本地 Flask API、LangGraph worker 和 DeepSeek 模型；数据库是本机隔离测试库，天气使用固定的合成开发夹具 `development_weather_clear_v1`。这两次是少量开发试跑，不是正式留出集测试或架构对照。

每个输入、观察及独立图审计均作为 JSON 留存。`scripts/analysis/audit_model_t5_pilots.py` 先核对观察报告的数据文件哈希，再用 `experiments/route_reference.py` 和 `experiments/scenario_v2.py` 独立检查**实际选中边 ID**的起终点、连续性、途经顺序和图内通行属性。观察报告只保留结构化结果、模型用量及 HTTP 状态；未归档模型凭证或原始提示词。

| 案例 | 实际顺序与结果 | 模型与传输 | 独立图审计 |
|---|---|---|---|
| `v2h_T5_route_weather_1` | `poi_139 → poi_070`，返回路线和天气，31 条实际边；8.635 秒 | `deepseek-flash` 1 次逻辑请求、1 次 HTTP 200；输入/输出 12,918/119 token | `pass`，无图内问题；就近路网接驳未触发 30 米预警 |
| `v2h_T5_two_vias_weather_1` | `poi_025 → poi_015 → poi_068 → poi_074`，返回路线和天气，87 条实际边；11.098 秒 | `deepseek-flash` 2 次逻辑请求、2 次 HTTP 200；两次输入/输出 12,872/164、13,548/175 token | `pass`，无图内问题；两个途经点按顺序到达原终点，就近路网接驳未触发 30 米预警 |

两个观察报告中的 `development_dataset_sha256` 均与归档的对应输入文件一致。原始材料 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `20261025_p1_t5_weather_pilot_input.json` | `d4fe92072eba25f3cb9e3a5a96a47587f3a9b1b7b9adf6acff17ee21e43c4561` |
| `20261025_p1_t5_weather_pilot_observation.json` | `c8d66891594544bcf62856b950cd97877aaa58cb45946c40810d31527cb22211` |
| `20261025_p1_t5_two_vias_pilot_input.json` | `14fe125b4720d56c1606dc6f78bd771311cb08ed140c80f78b42860869408511` |
| `20261025_p1_t5_two_vias_pilot_observation.json` | `2a6045525c6725cdfd04365d592d18c5cb213f6db063dd9c82dc8d485991a1e9` |
| `20261025_p1_t5_two_real_model_pilot_graph_audit.json` | `60b9a7954ef889bcccb7819e84cc299a73f30d787eab2eb87a578556dd5cba53` |

审计复跑命令：

```powershell
python -m scripts.analysis.audit_model_t5_pilots `
  --observation docs/research/evidence/20261025_p1_t5_weather_pilot_observation.json `
  --observation docs/research/evidence/20261025_p1_t5_two_vias_pilot_observation.json `
  --output output/experiments/t5_two_pilot_graph_audit_rerun.json
```

## 可得结论与限制

这两条题证明了当前数据版本下的真实模型调用链可完成指定输出，其中双途经题给出了按顺序连接原终点的全程边序列。样本各运行一次，不能据此估计 T5 成功率、延迟分布或优于其他架构。图内审计也不证明现场可通行；“接驳未触发 30 米预警”不等于入口已实测。其余 13 条模型依赖 T5 编题、三架构同题比较、正式留出与端侧画线仍待验收。没有使用或修改生产路况。
