# 2026-10-25 工学部集贸市场地点标签修正

在 T2 末端接驳审查中，`poi_387` 与 `poi_388` 的 GCJ-02 坐标转为 WGS-84 后落在 OSM `way/153741407` 工学部集贸市场面内；两条名称均指向工学部集贸市场，市场自身 `poi_415` 的 `campus` 为工学部，三点也落在本地 OSM 工学部边界内。原两家店误标为文理学部，市场本身的描述也误写“文理学部校内地点”。本轮只修正这两家店的 `campus` 与描述，以及市场描述；不改变坐标、类别、活动、POI ID 或排序算法。证据是来源几何与现有主数据的一致性，**不代表店面入口已经核验**。

`poi_392`–`poi_394` 名称也写“工学部店”，本地 OSM 边界归工学部，而主数据记文理学部；它们在边界附近，不因本次市场包含关系自动改动。`poi_395` 则相反：主数据工学部，OSM 边界归文理学部。这些需要独立来源复核，保持待审状态。

## 重算结果和版本边界

- 修正后的 `data/pois.json` SHA-256：`8322f12d0787e6aff7bceb489a7b8607bd748e3c4bbde543011475b7528d3307`。旧版依赖 POI 哈希的开发报告保留为历史记录，不可与新版混用。
- 新版 T2 候选参考：`output/experiments/T2_candidate_reference_after_campus_fix_20261025.json`，SHA-256 `6a7bccdb3b614489ccb0fe7c14786e5ae6d5cadedd9debd0bf06efcc528b4546`。24 题的 `cases` 与修正前逐项相同，因本轮只改学部标签和描述；来源哈希已变。
- 新版逐题不确定性：`output/experiments/T2_uncertainty_after_campus_fix_20261025.json`，SHA-256 `75dfc600c1cfac94710167ae028e5cbeff74bfd8dd94d8b483939c7805b7664d`。仍为 15 道图内可判、9 道入口未知，涉及 7 个 POI。
- 新版场地几何诊断：`output/experiments/T2_access_geometry_after_campus_fix_20261025.json`，SHA-256 `4378840b70376cdc29e55e4d2eafef875dd198f55925dc5bf8158f80cbffd216`。两家店与市场面的学部冲突已消除；5 个中心点表示预警和 9 道未知均未变化。

聚焦回归 `tests/test_market_poi_campus_labels.py` 与 `tests/test_poi_activity_taxonomy.py` 为 21/21 通过。前者同时检查三个点在 OSM 工学部边界内、主数据学部与描述相符。这个检查依赖 OSM 边界，不是独立现场证据；附近边界争议仍待核。

重算命令：

```powershell
python -m scripts.analysis.build_candidate_references_v2 --draft experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/T2_candidate_reference_after_campus_fix_20261025.json
python -m scripts.analysis.audit_t2_candidate_uncertainty --reference output/experiments/T2_candidate_reference_after_campus_fix_20261025.json --output output/experiments/T2_uncertainty_after_campus_fix_20261025.json
python -m scripts.analysis.audit_t2_access_geometry --areas experiments/source/osm_t2_area_evidence_20260924.geojson --uncertainty output/experiments/T2_uncertainty_after_campus_fix_20261025.json --output output/experiments/T2_access_geometry_after_campus_fix_20261025.json
```

以上输出均为忽略目录中的本地开发记录，正式实验前还须冻结新输入、形成未见测试集和可独立裁决的答案。由于输出命令拒绝覆盖既有文件，复跑时应指定新文件名。
