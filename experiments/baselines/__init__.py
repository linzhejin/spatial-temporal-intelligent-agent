"""基线方法集合

- shortest_path_baseline: R1 最短路径基线 (networkx dijkstra length)
- fixed_weight_baseline: R2 固定多因素权重基线 (DEFAULT_WEIGHTS)
- rule_baseline:          B1 规则兜底解析基线 (复用 _rule_based_classify)
- generic_llm_baseline:   B2 通用 LLM 直答基线（无工具、无地名核验、无路网）
"""
