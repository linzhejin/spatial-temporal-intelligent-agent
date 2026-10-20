# 启动/停止本机隔离 PostgreSQL（仅测试用途；不接触业务库、不读取 .env）
# 运行时: C:\Users\HUAWEI\.codex\test-postgres-runtime\Library\bin  (PostgreSQL 17)
# 数据目录: C:\Users\HUAWEI\.codex\test-postgres-data
# 端口 55439，仅监听 127.0.0.1，pg_hba 为 trust（无需口令）
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts\dev\start_test_postgres.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\dev\start_test_postgres.ps1 -Database whu_other_test
#   powershell -ExecutionPolicy Bypass -File scripts\dev\start_test_postgres.ps1 -Stop
param(
    [string]$Database = "whu_agent_local_test",
    [int]$Port = 55439,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$runtime = "C:\Users\HUAWEI\.codex\test-postgres-runtime\Library\bin"
$data    = "C:\Users\HUAWEI\.codex\test-postgres-data"
$log     = "C:\Users\HUAWEI\.codex\test-postgres.log"

$pgCtl    = Join-Path $runtime "pg_ctl.exe"
$pgReady  = Join-Path $runtime "pg_isready.exe"
$createdb = Join-Path $runtime "createdb.exe"
$psql     = Join-Path $runtime "psql.exe"

if (-not (Test-Path $pgCtl)) { throw "便携运行时缺失: $runtime（勿自行 initdb 到项目目录）" }
if (-not (Test-Path (Join-Path $data "PG_VERSION"))) { throw "集群数据目录缺失: $data" }

function Test-ServerUp {
    & $pgReady -h 127.0.0.1 -p $Port *> $null
    return $LASTEXITCODE -eq 0
}

if ($Stop) {
    if (Test-ServerUp) { & $pgCtl -D $data stop -m fast } else { Write-Host "服务未运行。" }
    return
}

if (-not (Test-ServerUp)) {
    Write-Host "启动隔离 PostgreSQL (port $Port)..."
    & $pgCtl -D $data -o "-p $Port -h 127.0.0.1" -l $log start
    for ($i = 0; $i -lt 30 -and -not (Test-ServerUp); $i++) { Start-Sleep -Milliseconds 500 }
}
if (-not (Test-ServerUp)) { throw "启动失败，查看日志: $log" }

# 本轮专用测试库（库名须满足 tests/db_safety.py 的 (?:^|_)test(?:_|$) 规则）
$exists = (& $psql -h 127.0.0.1 -p $Port -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$Database'" 2>$null | Out-String).Trim()
if ($exists -ne "1") {
    Write-Host "创建本轮专用测试库 $Database ..."
    & $createdb -h 127.0.0.1 -p $Port -U postgres $Database
    if ($LASTEXITCODE -ne 0) { throw "创建数据库失败: $Database" }
}

Write-Host ""
Write-Host "就绪。当前测试进程使用："
Write-Host ('  $env:TEST_DATABASE_URL="postgresql://postgres@127.0.0.1:' + $Port + '/' + $Database + '"')
Write-Host ""
Write-Host "集群现有库（模板库除外）："
& $psql -h 127.0.0.1 -p $Port -U postgres -tAc "SELECT datname FROM pg_database WHERE datistemplate=false ORDER BY 1"
