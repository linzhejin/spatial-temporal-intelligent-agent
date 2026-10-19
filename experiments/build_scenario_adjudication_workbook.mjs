import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = path.join(root, "experiments", "datasets", "scenarios_dev_v1.json");
const protocolPath = path.join(root, "experiments", "protocol_v1.json");
const outputPath = path.join(root, "output", "experiments", "scenario_adjudication_queue_20261018_intent-tools.xlsx");
const previewDir = path.join(root, "output", "experiments", "scenario_adjudication_queue_20261018_previews");

const sourceBytes = await fs.readFile(sourcePath);
const protocolBytes = await fs.readFile(protocolPath);
const sourceHash = createHash("sha256").update(sourceBytes).digest("hex");
const protocolHash = createHash("sha256").update(protocolBytes).digest("hex");
const dataset = JSON.parse(sourceBytes.toString("utf8"));
const protocol = JSON.parse(protocolBytes.toString("utf8"));
if (dataset.cases.length !== 60) throw new Error(`expected 60 draft cases, found ${dataset.cases.length}`);
const turnCount = dataset.cases.reduce((sum, item) => sum + item.messages.length, 0);
if (turnCount !== 72) throw new Error(`expected 72 user turns, found ${turnCount}`);
const replaceGenerated = process.argv.includes("--replace-generated");
if (await exists(outputPath)) {
  if (!replaceGenerated) throw new Error(`refusing to overwrite ${outputPath}`);
  await fs.rm(outputPath);
}

const intentLabels = [
  ["place_recommendation", "地点推荐：按类别找吃饭、学习等地点"],
  ["activity_recommendation", "活动推荐：按运动或活动找合适设施"],
  ["route_planning", "路线规划：新建路线"],
  ["route_revision", "路线修改：调整现有路线或约束"],
  ["route_comparison", "路线比较：比较不同策略或路线"],
  ["information_lookup", "信息查询：查天气、地点详情、路况等"],
  ["clarification", "澄清：本轮主要任务是询问缺失信息"],
  ["compound_request", "组合任务：同一轮有多个并列目标"],
  ["other", "其他：不属于以上类型，请在备注说明"],
];
const toolNames = protocol.gold_contract.available_tools;

const workbook = Workbook.create();
const notes = workbook.worksheets.add("审查说明");
const casesSheet = workbook.worksheets.add("场景审查");
const turnsSheet = workbook.worksheets.add("逐轮裁决");
for (const sheet of [notes, casesSheet, turnsSheet]) sheet.showGridLines = false;

notes.getRange("A1:B1").merge();
notes.getRange("A1").values = [["珞珈智行 Agent 需求理解与工具选择审查表"]];
notes.getRange("A1:B1").format = { fill: "#6E3532", font: { name: "Microsoft YaHei", size: 16, bold: true, color: "#FFFFFF" }, verticalAlignment: "center" };
notes.getRange("A1:B1").format.rowHeight = 32;
const noteRows = [
  ["用途", "对开发集草稿进行人工独立裁决；当前文件不含预设金标准答案。"],
  ["源开发集", `experiments/datasets/scenarios_dev_v1.json | SHA-256 ${sourceHash}`],
  ["源开发集 SHA-256", sourceHash],
  ["协议文件", `experiments/protocol_v1.json | SHA-256 ${protocolHash}`],
  ["协议 SHA-256", protocolHash],
  ["开发场景 / 用户轮次", `${dataset.cases.length} 场景 / ${turnCount} 轮`],
  ["重要限制", "当前是开发集，不是盲测集；填写结果不能作为模型效果结论。请根据用户原话和独立证据裁决，不要照抄草稿程序当前的 expected_turns。"],
  ["审查流程", "先在“场景审查”填写场景级结论；再在“逐轮裁决”逐轮填写预期响应、用途标签、可接受工具序列和证据。完成后由 import_scenario_adjudication.py 导入到新 JSON；不会覆盖源数据。"],
  ["工具序列填写", "JSON 二维数组；每个内层数组是一种可接受的工具调用顺序。示例：[[\"search_poi_candidates\"]]；没有工具调用应写 [[]]；允许多种等价方式可写 [[\"resolve_poi\",\"plan_route\"],[\"plan_route\"]]。不得把参数值或敏感信息写进工具序列。"],
  ["工具选择评分", "只比较本轮实际工具名称及顺序。没有启用工具追踪时记为未知；未完成人工裁决时记为未裁决。工具选择单独报告，不直接改变任务成功率。"],
  ["意图标签", "只选最贴合本轮用户目标的一项。组合任务仅在同一轮确有多个目标时使用；后续补充条件仍按其所服务的任务标注。"],
  ["来源与证据", "记录能够支持 POI、道路可行性、事件快照或工具选择判据的来源。没有独立证据时写明未知，不根据界面自述推定正确。"],
  ["审查人和日期", "场景和逐轮的审查人需一致；方法必须为 manual_independent；日期用 YYYY-MM-DD。"],
  ["意图标签选项", intentLabels.map(([id]) => id).join(", ")],
  ["注册工具", toolNames.join(", ")],
];
notes.getRange(`A2:B${noteRows.length + 1}`).values = noteRows;
notes.getRange(`A2:A${noteRows.length + 1}`).format = { fill: "#F1EEE7", font: { name: "Microsoft YaHei", bold: true, color: "#53312F" }, wrapText: true, verticalAlignment: "center" };
notes.getRange(`B2:B${noteRows.length + 1}`).format = { font: { name: "Microsoft YaHei", size: 10, color: "#333333" }, wrapText: true, verticalAlignment: "center" };
notes.getRange("A:A").format.columnWidth = 24;
notes.getRange("B:B").format.columnWidth = 108;
notes.getRange(`A2:B${noteRows.length + 1}`).format.rowHeight = 36;
notes.getRange("A8:B13").format.rowHeight = 58;
notes.freezePanes.freezeRows(1);

const caseHeaders = [
  "场景ID", "源快照标记", "用户轮次数", "源审查状态", "审查状态", "结论摘要",
  "审查人", "审查方法", "证据/参考", "理由与备注",
];
const caseRows = dataset.cases.map((item) => [
  item.id, item.snapshot_id, item.messages.length, item.adjudication?.status ?? "draft",
  "", "", "", "", "", "",
]);
casesSheet.getRangeByIndexes(0, 0, 1, caseHeaders.length).values = [caseHeaders];
casesSheet.getRangeByIndexes(1, 0, caseRows.length, caseHeaders.length).values = caseRows;
styleTable(casesSheet, `A1:J${caseRows.length + 1}`, `A1:J1`);
casesSheet.getRange(`A2:D${caseRows.length + 1}`).format = { fill: "#F7F5F0", font: { name: "Microsoft YaHei", size: 9, color: "#666666" }, wrapText: true, verticalAlignment: "center" };
casesSheet.getRange(`E2:J${caseRows.length + 1}`).format = { font: { name: "Microsoft YaHei", size: 10, color: "#272727" }, wrapText: true, verticalAlignment: "center" };
casesSheet.getRange("A:A").format.columnWidth = 22;
casesSheet.getRange("B:B").format.columnWidth = 28;
casesSheet.getRange("C:C").format.columnWidth = 12;
casesSheet.getRange("D:D").format.columnWidth = 14;
casesSheet.getRange("E:E").format.columnWidth = 12;
casesSheet.getRange("F:F").format.columnWidth = 42;
casesSheet.getRange("G:H").format.columnWidth = 20;
casesSheet.getRange("I:J").format.columnWidth = 42;
casesSheet.getRange(`A2:J${caseRows.length + 1}`).format.rowHeight = 32;
casesSheet.getRange(`E2:E${caseRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["待审", "已审", "暂缓"] } };
casesSheet.freezePanes.freezeRows(1);

const turnHeaders = [
  "场景ID", "轮次", "源快照标记", "此前对话", "本轮用户原话", "审查状态",
  "预期响应状态", "预期输出类型", "任务判读", "可行性", "起点 POI ID",
  "终点 POI ID", "途经 POI ID（有序）", "出行方式", "路线策略", "硬约束 JSON",
  "必需输出（JSON列表）", "候选 POI ID（JSON有序列表）", "澄清槽位",
  "允许未知项（JSON列表）", "意图标签", "可接受工具调用序列（JSON二维数组）",
  "事件快照/证据ID", "证据来源/参考", "审查人", "审查日期", "理由与备注",
];
const turnRows = [];
for (const item of dataset.cases) {
  item.messages.forEach((message, index) => {
    turnRows.push([
      item.id, index + 1, item.snapshot_id,
      item.messages.slice(0, index).map((prior) => prior.content).join("\n"),
      message.content, "", "", "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "", "", "",
    ]);
  });
}
turnsSheet.getRangeByIndexes(0, 0, 1, turnHeaders.length).values = [turnHeaders];
turnsSheet.getRangeByIndexes(1, 0, turnRows.length, turnHeaders.length).values = turnRows;
styleTable(turnsSheet, `A1:AA${turnRows.length + 1}`, "A1:AA1");
turnsSheet.getRange(`A2:E${turnRows.length + 1}`).format = { fill: "#F7F5F0", font: { name: "Microsoft YaHei", size: 9, color: "#555555" }, wrapText: true, verticalAlignment: "center" };
turnsSheet.getRange(`F2:AA${turnRows.length + 1}`).format = { font: { name: "Microsoft YaHei", size: 9, color: "#272727" }, wrapText: true, verticalAlignment: "center" };
const widths = [22, 8, 28, 42, 40, 12, 17, 16, 28, 14, 20, 20, 28, 12, 18, 26, 24, 30, 16, 24, 26, 42, 24, 42, 20, 14, 42];
widths.forEach((width, index) => {
  const col = columnName(index + 1);
  turnsSheet.getRange(`${col}:${col}`).format.columnWidth = width;
});
turnsSheet.getRange(`A2:AA${turnRows.length + 1}`).format.rowHeight = 46;
turnsSheet.getRange(`D2:E${turnRows.length + 1}`).format.rowHeight = 60;
turnsSheet.getRange(`F2:F${turnRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["待审", "已审", "暂缓"] } };
turnsSheet.getRange(`G2:G${turnRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["completed", "needs_input", "infeasible", "failed"] } };
turnsSheet.getRange(`H2:H${turnRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["route", "candidates", "clarify", "information", "no_route"] } };
turnsSheet.getRange(`J2:J${turnRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["feasible", "infeasible", "unknown", "not_applicable"] } };
turnsSheet.getRange(`N2:N${turnRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["walk", "bike", "drive"] } };
turnsSheet.getRange(`U2:U${turnRows.length + 1}`).dataValidation = { rule: { type: "list", values: intentLabels.map(([id]) => id) } };
turnsSheet.freezePanes.freezeRows(1);

workbook.recalculate();
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
for (const sheet of [notes, casesSheet, turnsSheet]) {
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 0.65, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheet.name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, cases: dataset.cases.length, turns: turnRows.length, sourceHash, protocolHash }));

async function exists(target) {
  try { await fs.access(target); return true; } catch { return false; }
}

function columnName(number) {
  let result = "";
  while (number > 0) {
    number -= 1;
    result = String.fromCharCode(65 + (number % 26)) + result;
    number = Math.floor(number / 26);
  }
  return result;
}

function styleTable(sheet, range, header) {
  sheet.getRange(range).format.font = { name: "Microsoft YaHei", size: 10, color: "#272727" };
  sheet.getRange(range).format.borders = { preset: "inside", style: "thin", color: "#E2DED5" };
  sheet.getRange(header).format = {
    fill: "#6E3532",
    font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(header).format.rowHeight = 34;
}
