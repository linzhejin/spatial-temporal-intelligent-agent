# 2026-10-25 P1 编题草稿与开发参考版本重建

起点是 `4eb3e22` 对 `data/pois.json` 的三处市场学部/描述修正，POI SHA-256 从 `db5f4177b596a1b1ae614dacafa39ce78ae04d5a7f8ef41ef7fa166a62958b0c` 变为 `8322f12d0787e6aff7bceb489a7b8607bd748e3c4bbde543011475b7528d3307`。原 120 题草稿仍声明旧哈希，严格参考审计会拒绝与当前 POI 混用。先用现有生成器写入忽略目录，与两个跟踪草稿逐项比较，才更新跟踪文件。

| 草稿 | 变更范围 | 当前 SHA-256 |
|---|---|---|
| `experiments/datasets/scenarios_test_v2_diverse_draft.json` | 120 题的 `cases` 解析后逐项相等；仅 `poi_sha256` 与 `source_poi_sha256` 更新 | `b7e0971bba4c21591cbe3832d0cb70c2758819ec87cb5f8fcced8265a0d28bcd` |
| `experiments/datasets/scenarios_test_v2_draft.json` | 另有 3 道 `T5_sport_then_food` 旧题按当前生成器改为“选篮球场→选附近吃饭地点→给完整路线”的三轮流程，并补两轮候选规格；仍为已见草稿 | `712c6dd266cb773eddf2e59692d5298ee994b84e77eda2d5fcf31786be890394` |

多样化草稿重新生成 T1/T2/T3–T5 参考及轮次台账，所有报告的草稿、POI、路网、覆盖规则哈希彼此一致；报告都在忽略目录中，保留原始结果不覆盖旧文件。摘要如下：

| 开发态报告 | 结果 | SHA-256 |
|---|---|---|
| `scenario_split_after_poi_fix_20261025.json` | 60 开发题、120 草稿；指定字段无直接交叉，`formal_ready=false`，7 项阻碍 | `1c6e18ac555700142ee899df04e4af386b5f01adcf65113fffea0d8b6319d760` |
| `T1_route_reference_rebased_20261025.json` | 24/24 图内连通，不等于现场通行 | `ea9240334ec0376af8b231fabf31959f5277164baa179f9eca94e35f23078fd8` |
| `T2_candidate_reference_rebased_20261025.json` 与 `T2_uncertainty_rebased_20261025.json` | 24 题；15 道图内可判、9 道末端接驳未知 | `0f8120a81a1143ddc25e362a2f6722c8b013d512e76ef76e6445d7abfac3f372`；`65cb63ecb41d610e5ac07b0ff0ec2305460962a5b0c03475106aecbdb38650bb` |
| `T3_T5_route_reference_rebased_20261025.json` | 72 个路线轮次图内连通；无逐题语义或实地真值 | `71c486c9a20566543f2bac2328cabbe69da614777a6968b17e8516f17b24b536` |
| `T5_composite_reference_rebased_20261025.json` | 9/9 选点在前四，9/9 图内连通；不证明真实模型完成复合任务 | `c92fe3cf1d521902336f0684a5cb78e6868330d7d187c496a86550da6bc0798b` |
| `T1_T5_turn_evidence_coverage_rebased_20261025.json` | 120 题、159 轮；96 路线、24 排序、12 选点成员参考，**27 轮仍缺独立逐轮参考**；正式裁决 0 题 | `2d28be4bcab7a2431b1d39d7c03c1e4f52fce522abff93fa24452745eb3141d7` |
| `T2_access_geometry_rebased_20261025.json` | 7 个受影响 POI、5 个来源面、5 个中心点表示预警；入口结论仍未知 | `fac3bfa4c780eb0d864ea46260b2a9fbf727ed8855bd1e0fc514dc89532ec992` |

验证：编题生成、T5 组合与多轮路线参考 11/11 项通过（JUnit `93929370330ac551a3397bc427b0889f769a9d1d57d6c5f9d79e86b121bffe68`）；审查卡、草稿、拆分及轮次覆盖 34/34 项通过（JUnit `55bc21c9decbd5b15c0459107024833c65508ae91800f4700a95a8f9c86ed225`）。后一次测试首次命令误写不存在的文件名，未收集测试；修正为 `tests/test_turn_evidence_coverage.py` 后完成 34 项。

旧版报告继续作为已见开发记录保存。正式实验还缺具体题独立裁决、真正未见且来源分离的测试集、冻结快照、真实模型同题回放与相应预算确认；不能把本轮图内参考称为论文成功率。
