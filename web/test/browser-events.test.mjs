import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(
  new URL("../lib/browser-events.ts", import.meta.url),
  "utf8",
)
const compiled = ts.transpile(source, {
  module: ts.ModuleKind.ES2022,
  target: ts.ScriptTarget.ES2022,
})
const browserEvents = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
)

assert.deepEqual(
  browserEvents.getBrowserOperationUpdate("tool_start", {
    tool_name: "browser_use",
  }),
  { kind: "running", label: "网页自动化" },
)
assert.deepEqual(
  browserEvents.getBrowserOperationUpdate("tool_progress", {
    tool: "browser_navigate",
    action: "navigate_completed",
    progress: 140,
  }),
  { kind: "running", label: "网页已打开", progress: 100 },
)
assert.equal(
  browserEvents.getBrowserOperationUpdate("tool_start", {
    tool_name: "python_executor",
  }),
  null,
)
assert.deepEqual(
  browserEvents.getBrowserOperationUpdate("tool_end", {
    tool_name: "browser_search",
  }),
  { kind: "finished" },
)
assert.deepEqual(
  browserEvents.getBrowserOperationUpdate("tool_error", {
    tool_name: "browser_use",
    error: "页面加载失败",
  }),
  {
    kind: "error",
    label: "网页自动化",
    message: "页面加载失败",
  },
)
assert.deepEqual(
  browserEvents.getBrowserOperationUpdate("cancelled", {}),
  { kind: "reset" },
)

console.log("browser-events.test.mjs passed")
