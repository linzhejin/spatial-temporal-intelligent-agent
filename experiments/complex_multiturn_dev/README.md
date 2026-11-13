# 复杂多轮开发测试

这套工具用于当前系统的开发与回归，不是论文盲测，也不比较三种架构。题包共 32 题，每题 3—5 轮；四类各有 6 道核心题和 2 道变体。实际开发过程中，变体在所有核心问题修复之前已被运行，并参与了后续修复；这一顺序与最初计划有偏差，不能称为未见过的留出集。

题目及中文检查说明见 `experiments/datasets/complex_multiturn_dev_v1/题目说明.md`，机器题包是同目录的 `cases_event_endpoint_revised.json`。`cases.json` 保留最初题包：事件题 `events_03` 初版终点在骑行图内不可达，因此在任何事件题真实调用之前修订了终点并重建事件快照；两版均保留；旧版事件文件缺少题包散列绑定，仅作历史记录，当前运行使用带完整来源散列的 v2 快照，不可混用。所有事件均为合成数据，绝不表示真实道路状态。

## 使用前提

- Windows 本机已启动专用 PostgreSQL 测试集群，运行器会核对监听进程和数据目录，创建随机的新测试库；不得指向业务库。
- 指定的环境文件需提供 DeepSeek 密钥和地图检索凭据。不要复制或打印凭据。模型固定为用户授权的 `deepseek-flash`，并强制 `AGENT_DIALOGUE_MODE=model`；该测试运行器不更改线上配置。
- 本轮预算账本保存在 `output/experiments/complex_multiturn_dev_v1/budget.jsonl`，采用文件锁、逐次按全未命中峰值预留；返回的缓存用量完整且自洽时，预算读取按峰值命中/未命中费率核算，缺失数据仍全部按未命中计入，原记录不改写。用户追加授权后，同一账本的累计保守上限为 40 元；运行时需显式传 `--budget-limit-cny 40`，默认仍为 20 元。不得删除或另起账本绕过上限。供应商用量不明、费率无法确认或预算不足时停止新增付费调用。账本上界并非实际扣费，详见 `费用对账_20261113.md`。
- 一笔 HTTP 400 请求未返回用量，默认守卫因此中止；用户随后明确同意将其预留的 0.18553 元全额作为未知用量的保守上界入账后继续。账本追加了带原因和时间的记录，原请求记录未改写。此处理不代表实际扣费，也不自动授权未来未知用量。
- 测试记录中的 `开发测试逐题报告_20261113_v3.md` 是中间快照；最终批次逐题结果见 `开发测试逐题报告_20261114_v4.md`，问题及证据边界见 `问题处理记录_20261113.md`。
- Git 属性为冻结路网、POI、题包和锚点指定原始换行格式，为检查器和补充题包指定 LF；这样在不同 `core.autocrlf` 设置下全新检出仍能匹配已保存的 SHA-256。已有旧工作区执行 `git pull` 不会自动重写未变化的路网文件；散列不符时应新建工作区检出，不要直接覆盖现有数据。- 固定校园路网和 POI、60 米几何锚点以及模拟事件受 SHA-256 绑定。在线地图检索结果可能变化；POI 现场存在、营业、入口和末端接驳尚未核实。

## 执行

在本工作区的 PowerShell 中，将下方的路径替换为实际环境文件；该文件本身不写入仓库。

```powershell
python -m experiments.complex_multiturn_dev.run_dev --env-file 'C:\path\to\.env' --phase core --budget-limit-cny 40 --dataset 'experiments/datasets/complex_multiturn_dev_v1/cases_event_endpoint_revised.json'
python -m experiments.complex_multiturn_dev.run_dev --env-file 'C:\path\to\.env' --phase variant --case events_07 --browser-case events_07 --budget-limit-cny 40 --dataset 'experiments/datasets/complex_multiturn_dev_v1/cases_event_endpoint_revised.json'
```

可重复使用 `--case` 按题选择，用 `--browser-case` 让所选题通过 Chromium 实际聊天页面输入；应先确认 Playwright 与 Chromium 已安装。浏览器题与 API 题共享隔离数据库、正常会话 API、真实后台 worker、模型费用账本和事件回放。选点分支在页面点击上一轮首张卡片；记录同时保存题目原话和页面实际提交的点选话语。

每次执行生成独立目录，保留运行清单、代码差异及散列、逐轮任务和工具观察、路线边序列、事件证据、用量、费用和浏览器截图。先前版本采集器的历史结果可用 `python -m experiments.complex_multiturn_dev.recover --help` 查看恢复方式；恢复从隔离库读取并另存，不覆盖原记录。

```powershell
python -m experiments.complex_multiturn_dev.audit 'output/experiments/complex_multiturn_dev_v1/<运行目录>' --dataset 'experiments/datasets/complex_multiturn_dev_v1/cases_event_endpoint_revised.json'
python -m experiments.complex_multiturn_dev.candidate_audit 'output/experiments/complex_multiturn_dev_v1/<运行目录>' --case search_07
python -m experiments.complex_multiturn_dev.candidate_audit 'output/experiments/complex_multiturn_dev_v1/<运行目录>' --case supplement_time_radius_separate_center --time
python -m experiments.complex_multiturn_dev.report --help
```

独立路网检查只核验冻结图中的路线边、端点、出行方式、可确认的硬约束及合成封闭边。候选范围检查使用本地地点或与供应商原始记录的身份及坐标一致、可吸附至冻结路网的在线地点；缺少这些证据时保留为证据不足。逐轮先运行 `candidate_audit`，再运行 `audit`。候选用途、空结果和必要澄清根据目录或脱敏供应商记录及工具调用检查；证据不足时不得记为通过。任务字段一致或产品自报成功不自动构成通过。所有失败、超时、证据不足及未运行项目都应保留；不要把逐轮图核验数量当作完整题目通过率。

额外的离线构造测试发现并修复了时间与搜索半径同时存在时可能漏掉一项限制的问题；另行提供 `supplement_time_radius_v2.json`，用于两个独立的真实多轮复测场景，不替换冻结的 32 题，也不是盲测。初版 `supplement_time_radius.json` 保留首次失败记录所用的原始散列；v2 加查没有被用户提出的路线距离上限。`--time` 在冻结图上独立寻找满足时间与距离的见证路线；找不到见证时不会把未检查的其他路线判为失败，现场可通行性仍未知。详见同目录的 `问题处理记录_20261113.md`。
