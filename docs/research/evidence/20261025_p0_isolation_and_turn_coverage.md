# 2026-10-25 P0 隔离复核与 P1 逐轮证据覆盖

本记录续接 `20261025_p0_p1_alignment_recheck.md`，基于研究分支 `9e5c1e2` 的产品代码和随后新增的研究审计脚本。所有场景仍是**已见编题草稿**，本记录不产生正式论文成绩。

## P0：隔离目标与路况写入边界

执行前，线上公开 `GET /api/road-conditions` 只读取到 **0 条生效事件**，`active_event_version=roads-4f53cda18c2baa0c`。随后显式指定本机 `whu_agent_experiment_test` 库，运行：

```powershell
$env:TEST_DATABASE_URL='postgresql://postgres@127.0.0.1:55439/whu_agent_experiment_test'
$env:RUN_MANAGER_VISION_BACKEND_E2E='1'
python -m pytest tests/test_db_safety.py tests/browser/test_manager_vision_backend_e2e.py -q --junitxml=output/experiments/p0_isolation_9e5c1e2_20261025.xml
```

5 项通过（数据库目标保护 4 项、浏览器到 PostgreSQL 的管理端影像事件链路 1 项）。测试的 `isolated_test_database` 夹具在写入前核对库名、实际连接身份，以及本机业务连接配置；浏览器用本地 Flask、合成候选和临时路况文件，外部浏览器请求被拦截。测试中逐项确认/排除候选、发布仅影响步行的事件、重复发布不新增、路线绕行、删除后恢复，并清理测试作业。JUnit SHA-256：`e2ac70024d5a31cbeb6dae5b93efbaaf5a8a0d237206a8d23ebb039905f5eac3`。

运行后再次读取线上公开接口，生效事件仍为 **0**，`active_event_version` 仍为 `roads-4f53cda18c2baa0c`。这只能证明该时间窗口内**用户可见的生效路况快照未变**；公开接口不返回管理员可见的未开始、已过期和历史记录，因此不能据此断言生产事件库全部记录前后相同。生产历史记录的独立审计仍留在 P0 未完成项。

## P1：159 个逐轮任务的开发参考覆盖

新增 `scripts/analysis/audit_v2_turn_evidence_coverage.py`，只接受可见的 `scenario-v2` 编题草稿，核对输入草稿与 T1、T2、T3–T5、T5 复合参考报告的来源哈希，并将每一轮映射到可用的**开发参考范围**。T2 不确定性报告还须指向同一份 T2 参考。脚本不调用 Agent，也不把图内参考升级为语义或现场金标准。

| 逐轮参考范围 | 数量 | 含义 |
|---|---:|---|
| 图内路线参考 | 96 | T1 24，T3–T5 72；可检查当前图中的连通与顺序，现场通行仍未知 |
| 图内候选排序参考 | 24 | T2 中 15 轮按当前图可判前四名、9 轮末端接驳保持未知 |
| 所选地点在前四名 | 12 | T5 的 9 个首轮及 3 个第二轮选点；没有全名单顺序和入口真值 |
| 无对应独立逐轮参考 | 27 | T3 澄清 24 轮、T4 只查询信息 3 轮；类别规则与开发试跑不能替代具体题的来源裁决 |

共 **120 题、159 轮**，正式裁决题数为 **0**。报告 `output/experiments/T1_T5_turn_evidence_coverage_sourcechecked_20261025.json`，SHA-256：`f93542644c3edf0b7dbac69ae597bb5268d5ee6363f80ed913c56c955ca1a7d7`。报告逐轮保留题号、轮次、参考类型、窄范围判断及尚缺的证据，不输出用户原文。

审计器先以“草稿参考不可当金标准”和“错误途经顺序必须被拒绝”两项反例测试复现缺口，再实现并运行；最终 2 项聚焦测试通过，JUnit SHA-256：`c3b9b9619ec0fd1a2e75ddc342341c5bb63b4da05b1a63b11922d14887e1d5ba`；`ruff check` 通过。没有运行完整测试套件。

## 下一步依据

1. **P0 仍未全部通过。**线上公开快照保持不变，但生产端全部历史事件未读回核对；正式输入快照也未冻结。
2. **P1 优先补具体题来源。**24 个 T3 澄清轮和 3 个 T4 信息轮需独立语义/输出证据；T2 的 9 个接驳未知保持未知，优先从现有地图或校方资料校核 7 个受影响 POI。T5 选点目前只有“所选地点在前四名”的编题参考。
3. 当前 120 条草稿都已用于开发，不能改名成未见测试。正式对照前还需另封测试集、冻结快照、完成同模型公平性和费用门槛。
