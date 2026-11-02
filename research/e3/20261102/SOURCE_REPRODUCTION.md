# 源码与输入复现范围

基准提交：`c9aa0f3f90e6a89c1e0856bd6ac1e8d059de9c37`。先检出该提交，再将 `source_inputs.zip` 解压到仓库根目录。归档提供本批冻结的145项执行输入、汇总脚本及验收测试源码，应用入口、前端资源和其他依赖源码由基准提交提供。这份归档不是脱离仓库即可启动的独立应用包。

逐文件SHA-256见 `source_inputs_manifest.json`；结果派生脚本的哈希同时与`summary.json` 中的生成记录核对。软件版本见原始批次 `source_manifest.json` 及 `acceptance/environment.json`。数据库只使用本机专用测试库；运行命令见 `experiments/E3_PRODUCT_COMPARISON.md`，验收的实际命令见`acceptance/acceptance_commands.json`。
