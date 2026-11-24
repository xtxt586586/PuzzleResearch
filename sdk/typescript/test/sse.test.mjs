import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

const responseBody = [
  "event: completion",
  "id: 7",
  'data: {"type":"completion","data":{"text":"hello"}}',
  "",
  "event: tool_result",
  'data: {"type":"tool_result","data":{"result":"ok"}}',
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

assert.deepEqual(events[0], {
  type: "completion",
  data: {
    text: "hello",
  },
})

assert.deepEqual(events[1], {
  type: "tool_result",
  data: {
    result: "ok",
  },
})

console.log("sse.test.mjs passed")