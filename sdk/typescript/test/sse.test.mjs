import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

const responseBody = [
  "event: completion",
  "id: event-001",
  'data: {"text":"hello"}',
  "",
  "event: tool_result",
  'data: {"tool":"echo","result":"ok"}',
  "",
].join("\n")

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async () =>
    new Response(responseBody, {
      status: 200,
      headers: {
        "content-type": "text/event-stream",
      },
    }),
})

const events = []

for await (const event of client.chatSse({
  message: "test",
})) {
  events.push(event)
}

assert.equal(events.length, 2)
assert.equal(events[0].event, "completion")
assert.equal(events[0].id, "event-001")
assert.deepEqual(events[0].value, {
  text: "hello",
})
assert.equal(events[1].event, "tool_result")
assert.deepEqual(events[1].value, {
  tool: "echo",
  result: "ok",
})

console.log("sse.test.mjs passed")