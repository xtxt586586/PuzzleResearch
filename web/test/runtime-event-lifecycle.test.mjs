import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(
  new URL("../lib/runtime-event-lifecycle.ts", import.meta.url),
  "utf8",
)
const compiled = ts.transpile(source, {
  module: ts.ModuleKind.ES2022,
  target: ts.ScriptTarget.ES2022,
})
const lifecycle = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
)

const pendingApproval = { kind: "tool_approval", message: "需要确认" }

for (const eventType of ["completion", "cancelled", "error"]) {
  assert.equal(lifecycle.isTerminalRuntimeEvent(eventType), true)
  assert.equal(
    lifecycle.pendingInterruptAfterRuntimeEvent(eventType, pendingApproval),
    null,
  )
}

for (const eventType of [
  "done",
  "interrupt",
  "tool_approval_required",
  "tool_start",
  "status",
  "token",
]) {
  assert.equal(lifecycle.isTerminalRuntimeEvent(eventType), false)
  assert.equal(
    lifecycle.pendingInterruptAfterRuntimeEvent(eventType, pendingApproval),
    pendingApproval,
  )
}

assert.equal(lifecycle.firstPendingInterrupt({ interrupts: [] }), null)
assert.equal(lifecycle.firstPendingInterrupt({ __interrupt__: [] }), null)
assert.deepEqual(
  lifecycle.firstPendingInterrupt({ interrupts: [{ value: { type: "human_review" } }] }),
  { value: { type: "human_review" } },
)
assert.deepEqual(
  lifecycle.firstPendingInterrupt({ type: "human_review" }),
  { type: "human_review" },
)

console.log("runtime-event-lifecycle.test.mjs passed")
