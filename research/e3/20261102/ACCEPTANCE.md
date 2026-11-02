# E3 全链路对照验收

范围：现有受限最短路复用与完整重算。验收完成时间以 `acceptance/acceptance_commands.json` 和各批摘要为准；提交与远端同步在发布回执中另记。

| 要求 | 核对结果 | 直接依据 |
|---|---|---|
| 同一产品入口与共享条件的两策略 | 两侧均实际调用API；完整重算仅去掉证明句柄 | `raw_batches.zip` 内协议、请求和完整响应；冻结执行源码 |
| 固定矩阵及重复次数 | 126条件，756计时＋252探针，键无缺失和重复 | [汇总](summary.json)、[1008条逐行来源审阅](acceptance/final_independent_audit.json) |
| 独立有效性与最优性 | 每侧357路线交付、21正确不可达；357组可达计时代价一致 | [逐次评分](executions.csv)、[配对表](paired_comparison.csv)；原始audit字段 |
| 事件与任务变化范围 | 21类变化，含时效／方式／恢复／联合变化／图版本／不可达 | [分变化汇总](variant_summary.csv)、原始protocol与expected文件 |
| 实际复用与应重算行为 | 126次实际复用；其余按条件重算或拒绝后新建 | 逐次execution_policy_audit、实际Dijkstra计数 |
| 维护代价与计时分离 | 756计时与252操作探针分开；总体Dijkstra减少35%，实际复用配对时延差中位数+0.961秒 | [总体](overall_summary.csv)、[阶段](phase_summary.csv)、[六组起终点](family_summary.csv) |
| 弱监听失效反例 | 3类反例，18计时＋6探针；旧路径可行但不再最优 | [反例逐次](counterexample_executions.csv)、原始weak_diagnostics |
| 真实证明与失效保护 | 主分析504份、反例12份初始证明对照数据库；伪造、过期、owner、版本与存储故障均有回归 | 原始stored_proofs；[238项Python回归](acceptance/regression.xml) |
| 交互及旧结果保护 | PC与移动尺寸浏览器续接通过；101项JS生命周期检查通过 | [Python日志](acceptance/python_regression.log)、[JS日志](acceptance/javascript_lifecycle.log) |
| 中断保留和预定续跑 | 保留910，补98，额外6单列；504策略配对同段；随机顺序一致 | [预定选择清单](acceptance/resume_selection.json)、[独立审阅](acceptance/final_independent_audit.json)、[系统电源事件](acceptance/power_events_during_batch.json) |
| 源码、原始记录及复制校验 | 原始ZIP逐成员SHA＋CRC通过；351项源码／输入逐字节回读；验收复制与原件相同 | [原始清单](archive_manifest.json)、[源码清单](source_inputs_manifest.json)、[复制清单](acceptance/copied_files_manifest.json) |
| 研究范围与交接 | 报告保留已见固定子集和时延未改善结论；E1规模与通用依赖工作另列 | [研究交接](../../../docs/research/19_E3路径复用全链路对照_20261102.md) |

两段实际保存1014条记录：762次计时、252次探针；主分析排除了预先指定的6条重复计时。另一次待机中断探针未取得响应，未计成功。其余诊断、冒烟和旧评分批次保留原状，不纳入主分母。

[重启后的6条重复计时](restart_diagnostic.csv)用于观察批段差异，未用于选择或替换主结果。[起终点名称](od_groups.csv)来自冻结参考样本，现场通行核验保持未知。

本次回归覆盖E3、证明生命周期及相关任务／交互保护，不是全项目或线上部署验收。源码补充包的使用范围见 [复现说明](SOURCE_REPRODUCTION.md)。

验收日志及清单另以 [原字节归档](acceptance_evidence.zip) 保存，[逐文件SHA清单](acceptance_archive_manifest.json) 用于回读核对；Git对文本换行的转换不影响ZIP中的原始证据。
