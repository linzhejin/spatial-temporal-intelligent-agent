# 2026-10-25 推进方案进度复核

依据桌面《珞珈智行_论文实验推进方案_最终执行版_20261023.docx》与 `13_实验推进验收执行清单_20261025.md`。本记录只陈述本次核到的开发与验收状态，不把编题草稿或模拟影像写成论文结果。

## 版本与实验隔离

- 研究工作区 `codex/isolation-cleanup` 的干净提交为 `3f5dc21984a906ddf1623117097967028d511873`。本地 `main`、GitHub `origin/main` 和 Gitee `gitee/main` 的已获取提交均为 `6d78bc59ad96a1d9883c8e4a632df66c7a43b867`；两线自 `014c4832c6b847e65f4deb66dc302f4a477cba86` 分叉。这里没有证明线上服务运行的提交，也没有完成两线合并。
- `python -m scripts.validate.check_agent_research_readiness --capture-inputs output/experiments/dev_input_inventory_3f5dc21_20261025.json`：受跟踪工作树干净，有效输入指纹 `7cd3b0a1074dd447ab7f05b6fc09551b328fc822982986fff78c597aebd77dc6`；清单 SHA-256 为 `ab78ec252b01e94a1fa9f8cd548e2337e0854dbd573569365e225d5e560355d3`。清单明确为开发快照，`formal_eligible=false`。
- 设置专用本机 `TEST_DATABASE_URL` 后运行 `python -m scripts.validate.check_agent_research_readiness`：Agent 主实验的**代码与结构就绪检查**为 `ready=true`，11 组检查均通过，包括 356 项 Python 聚焦用例和 157 项前端用例。原始报告 `scripts/audit_output/agent_research_readiness.json`，SHA-256 `8cd1adf52884f2f31bf6001f1217c8f7459019c9e24060a932a65a7c33129d00`。这不是样本、费用或论文主实验的就绪证明。
- `RUN_MANAGER_VISION_BACKEND_E2E=1` 且仅设置专用测试库时，`python -m pytest tests/browser/test_manager_vision_backend_e2e.py -q --junitxml=output/experiments/p0_manager_vision_backend_3f5dc21.xml` 为 **1/1 通过**；JUnit SHA-256 `61073dea97ed16ae5825c88b8c497dd70653b4ac146689e552e18fafcdc4a534`。该用例使用本地 Flask、合成候选、临时道路事件文件并拦截远程浏览器请求。前后只读查询线上 `active_event_version` 均为 `roads-4f53cda18c2baa0c`；这证明活动事件版本未变，不是生产数据库所有记录的完整审计。
- 代码就绪报告仍将代表性路线证据列为未通过：6 条缺现场或校方可通行性证据，1 项缺独立步行路线对照；坡度和景观主数据各有 14622 个属性缺口；五类校园视觉现场证据均缺。它们与 Agent 代码检查是不同的证据范围。

## 样本与对照状态

- `python -m experiments.scenario_split_audit --development experiments/datasets/scenarios_dev_v2.json --test experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/split_audit_3f5dc21_20261025.json`：60 条开发题、120 条可见编题草稿（T1–T5 各 24）。指定的 ID、模板、起终点组、原文指纹和来源组均无直接交叉，`leakage_free=true`；报告 SHA-256 `a35fd1469ce914f4b7ad03f27a7c4537625f0c186b60f3c40e79bb70a8105e5`。
- 同一报告 `formal_ready=false`，有 7 类阻碍：120 条尚未独立裁决、快照 ID 未冻结、正式数据缺图 SHA-256，以及快照清单、Agent 就绪、提交匹配、干净工作树四项外部门槛未作为正式运行条件完成核验。字段无直接交叉不等于真正未见；这些草稿及同源题已用于开发，不能直接改名为正式留出集。
- T1/T2 图内参考、T3/T4 上下文与天气、T5 选点后有序全程，以及 DeepSeek 下五类任务的同题三架构开发诊断已有单独证据。它们暴露并修复了真实功能缺陷，但均为已见题，未做三次重复或完整独立金标准，不报告架构优劣的论文成功率。Kimi、GLM、BFCL 正式模型结果和批量费用上限仍未完成。
- 公开影像已具备本地识别与模拟路况的部分流程，受控封闭/到期夹具与合成候选管理端闭环可以演示；六类 D1 事件、6 视频 × 12 映射的真实模型候选复核以及事故、积水校园精度均未完成。公开视频映射不能解释为武汉大学真实路况。

## 接着推进的顺序

1. 研究分支逐项对齐主线新修复，保留干净开发清单；不因代码就绪 `ready=true` 提前宣称正式实验就绪。
2. 先完善 T3–T5 的逐题来源与可判范围，核对 T2 的 7 个接驳不确定地点；无法独立判定的项目保持未知。将所有已见草稿归入开发材料，另封真正未见的测试集并核验拆分与独立评分。
3. 扩充 DeepSeek 三架构的分层小样本费用与失败分布，形成可审核预算；费用上限明确后再做正式批量调用。Kimi、GLM 缺访问条件时如实标为未运行。
4. 并行完善 D1 六类事件及公开视频 V1 的逐候选复核材料，不等待实地航拍；论文引言、方法和实验设置可先供师兄审阅，结果须待正式实验。

目前无需用户逐路、逐题或逐帧采集；只有审查包和费用预算准备好时，才集中请求少量判断。
