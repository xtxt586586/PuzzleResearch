import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(
  new URL("../lib/stream-text-batcher.ts", import.meta.url),
  "utf8",
)
const compiled = ts.transpile(source, {
  module: ts.ModuleKind.ES2022,
  target: ts.ScriptTarget.ES2022,
})
const { createStreamTextBatcher } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
)

let nextFrameId = 0
const frames = new Map()
const cancelledFrames = []
const flushed = []
const batcher = createStreamTextBatcher(
  (text) => flushed.push(text),
  (callback) => {
    const id = ++nextFrameId
    frames.set(id, callback)
    return id
  },
  (id) => {
    cancelledFrames.push(id)
    frames.delete(id)
  },
)

batcher.append("alpha")
batcher.append("beta")
assert.equal(frames.size, 1)
const [scheduledId, callback] = frames.entries().next().value
frames.delete(scheduledId)
callback()
assert.deepEqual(flushed, ["alphabeta"])

batcher.append("gamma")
const pendingId = frames.keys().next().value
batcher.flush()
assert.deepEqual(flushed, ["alphabeta", "gamma"])
assert.deepEqual(cancelledFrames, [pendingId])

batcher.append("discarded")
batcher.dispose()
assert.equal(frames.size, 0)
assert.deepEqual(flushed, ["alphabeta", "gamma"])

console.log("stream-text-batcher.test.mjs passed")
