import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

let receivedHeaders

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async (input, init = {}) => {
    const url = new URL(String(input))

    receivedHeaders = new Headers(init.headers)

    assert.equal(
      url.pathname,
      "/api/events/thread%2Fone",
    )

    return new Response(
      [
        "id: 4",
        'data: {"type":"done","data":{}}',
        "",
      ].join("\n"),
      {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
        },
      },
    )
  },
})

const events = []

for await (
  const event of client.streamResearchEvents(
    "thread/one",
    {
      lastEventId: 3,
    },
  )
) {
  events.push(event)
}

assert.deepEqual(events, [
  {
    type: "done",
    data: {},
  },
])

assert.equal(
  receivedHeaders.get("Last-Event-ID"),
  "3",
)

console.log("stream-parity.test.mjs passed")