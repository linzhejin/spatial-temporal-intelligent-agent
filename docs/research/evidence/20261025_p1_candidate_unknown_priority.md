# P1 候选排序的可判错误与接驳未知

研究代码提交为 `89ad06a147b52b1262ba93772c78ef11fb6202b3`。提交后在干净受跟踪工作树捕获开发态输入清单 `output/experiments/dev_input_inventory_89ad06a_20261025.json`，SHA-256 `87cf8df3ed5f73cd8c97ee000c03cd7a4c64987c2f3dcd134faadd61b4e25c54`；有效产品输入指纹仍为 `83ad771013cd1d857675760a60863541f688fc0fad5b2111e0f5f40d4c0dca03`。清单明确 `snapshot_kind=diagnostic`、`formal_eligible=false`，所以这不是正式冻结快照。

当前 `scenario-v2` 独立评分器先前在名单含有末端接驳未核实地点时，直接返回 `unknown`。这会掩盖名单中**其他已知地点**之间的确定性错误：距离明显倒序，或漏掉比已选已知地点更近的合格地点。末端未知只影响与该地点有关的名次，不应豁免其余可判部分。

现先检查已知地点：超过容差的倒序判 `wrong_distance_order`；漏掉比已选最远已知地点更近的合格已知地点判 `closer_eligible_omitted`。其余受未核实入口影响的情况仍判 `last_meter_access_unverified`，不以估算距离强判对错。两项回归先各自复现 `unknown` 的错误结果，再修改评分顺序使其判 `fail`；包含未知地点但已知地点顺序正确的反例仍判 `unknown`。

运行 `python -m pytest tests/test_scenario_v2.py tests/test_candidate_reference.py tests/test_turn_evidence_coverage.py tests/test_development_graph_reference.py -q --junitxml=output/experiments/p1_candidate_unknown_priority_regression_20261025.xml`：43/43 通过；JUnit SHA-256 `45e8964e53836a2bd278166d611083e26af298d5786b5163ebe8918babbdafe9`。运行 `python -m ruff check experiments/scenario_v2.py tests/test_scenario_v2.py` 与 `git diff --check`：均通过。

按当前 POI 与路网重算全部 24 道 T2 **编题参考**，仍为图内可判 15、末端接驳未知 9；逐题结果与改动前一致。重算报告 `output/experiments/T2_uncertainty_after_priority_fix_20261025.json` 的 SHA-256 为 `65cb63ecb41d610e5ac07b0ff0ec2305460962a5b0c03475106aecbdb38650bb`，与前一版输出相同。这说明修正没有改变当前预设参考的分布，只增强了对未来错误 Agent 名单的判别；不等于 24 题已获独立现场真值，更不等于正式留出集就绪。
