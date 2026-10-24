# 2026-10-25 首轮阶段验收：实验隔离与 T1/T2 编题参考

本记录按 `docs/research/13_实验推进验收执行清单_20261025.md` 执行。对象是可见的**开发/编题材料**，不是留出集，也没有计算 Agent 成功率。研究工作区为 `codex/isolation-cleanup`；本次输入清单基于干净提交 `1baf9e4a3823c7580088e5c4e8fce9047a97a922`。新增的 T2 审计器与本记录提交后，应重新捕获新的开发输入清单；旧指纹不能代表新提交。

## P0：版本和隔离

- `python -m scripts.validate.check_agent_research_readiness --capture-inputs output/experiments/dev_input_inventory_1baf9e4.json`：诊断快照 `formal_eligible=false`，捕获时受跟踪工作树干净，有效输入 SHA-256 为 `aa6b1b9bfc758a0253d3140721d9fe453a6c32f928ab324d1ecfaf7175bcd993`。清单文件 SHA-256：`092dc0c5b55abf8b636b8c69372992828497f6f8fcd3baf912293098e8554d61`。
- 数据库安全、场景集成、管理端到 PostgreSQL 的浏览器流和 T2 审计器聚焦回归，均仅向本机 `whu_agent_experiment_test` 测试库运行：`11 passed in 27.19s`，JUnit 报告 `output/experiments/p0_p1_focused_20261025.xml`，SHA-256：`6fe896882d9b0c8978fee8e76aecc4cdcf75116029bd7cb9bdab461763d03ff62`。该检查未验证生产数据库事件数前后相等，因此不据此宣称生产状态已被独立核对。
- 正式快照尚未冻结；本轮没有调用 `--write-snapshot`，没有运行收费的正式模型矩阵。
- 审计脚本提交后，在干净的研究提交 `69715b51bf4179ac21333d03f947c023da02acac` 重新捕获 `output/experiments/dev_input_inventory_69715b5.json`：`formal_eligible=false`，有效输入 SHA-256 仍为 `aa6b1b9bfc758a0253d3140721d9fe453a6c32f928ab324d1ecfaf7175bcd993`；清单 SHA-256 为 `8a602fd0e6470d3af5de9191d8f78de0f86ad2465d919bdd11957c66c3ac3eae`。这表明新增研究审计器未改变该清单覆盖的产品有效输入，不表示正式就绪。
- 本轮只读核对远端：GitHub `origin/main`、Gitee `gitee/main` 与本地主分支均为 `6d78bc59ad96a1d9883c8e4a632df66c7a43b867`。研究提交没有推送或部署；线上运行版本未在本轮通过服务端构建指纹独立核验。

## P1：拆分和独立图内参考

| 开发态检查 | 本轮观察 | 原始报告及 SHA-256 |
|---|---|---|
| 60 条开发题与 120 条可见编题草稿拆分 | 指定字段无直接交叉，空间锚点多样性通过；`formal_ready=false`，120 条均为 `draft` | `output/experiments/scenario_split_20261025.json` · `ed4cb86dc1ceebb15e9bea5583e748833467362f451abc942ea9a39e6ecba14e` |
| T1 路线编题参考 | 24/24 在当前方式过滤图内连通，0 条触发图内复核；不证明现场通行 | `output/experiments/T1_diverse_reference_20261025.json` · `a33393967f35fb04a282efdd412491a371642e759f4c6142f05c710d7b388eec` |
| T2 完整候选参考 | 24 条；参考生成器标记 11 条边界题 | `output/experiments/T2_diverse_reference_20261025.json` · `0b12f7f974b60d95dfd3947dbc000ea90bc09885340767ca9d0af0b250b11648` |
| T2 边界逐题审计 | 独立提出的前四名中，15 条按图内距离和 30 米并列容差可判，9 条因末端接驳保持未知；11 个生成器标记中有 2 个不影响排序 | `output/experiments/T2_uncertainty_20261025.json` · `1ff6cc857fde0dbcd1b0c90c794598ff8fafcfddf446bfc5791518174010a469` |

报告文件不覆盖已有结果。T2 审计的是程序独立提出的名单，**不是 Agent 实际答复**。受接驳不确定影响的去重地点只有 7 个：`poi_080`、`poi_194`、`poi_205`、`poi_206`、`poi_260`、`poi_387`、`poi_388`。先用现有来源核对这 7 个地点的导航落点；没有可靠证据的仍为未知，无需用户逐路实测。

拆分门槛仍有 7 项阻碍：120 题缺来源支持的独立答案、缺冻结输入指纹、缺图 SHA-256，以及正式快照、Agent 就绪、提交匹配、干净工作树四项外部门槛未验证。`leakage_free=true` 只表示本次检查的指定字段无直接交叉，不等于题源独立或未见盲测。

独立评分器及路线参考的现有反例回归：`python -m pytest tests/test_scenario_v2.py tests/test_scenario_protocol.py tests/test_route_reference.py -q --junitxml=output/experiments/p1_scorer_regressions_20261025.xml`，**66/66 通过**；JUnit SHA-256 为 `ad928df8eb1924eeab10748602d832390f78680a4ec74de3a8330b9e0a007eeb`。其中包括候选错序、重复候选、途经顺序和台阶约束反例。该自动回归验证评分器实现，不补足草稿题的独立答案。

## 结论与下一步

P0 仅完成开发快照和隔离回归的一部分；P1 已把 T1/T2 可判范围和未知范围分开，**P0/P1 均未通过正式实验门槛**。下一步先重捕提交后的输入清单，完善 T3–T5 的来源与独立判分，另封真正未见的测试集；在此之前三架构同模型运行只用于开发诊断，不报告论文成功率。
