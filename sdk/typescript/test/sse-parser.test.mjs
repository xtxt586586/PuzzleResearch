import assert from "node:assert/strict"
import {
  normalizeSseEvent,
  parseSseFrame,
  readDataStreamEvents,
  readSseEvents,
} from "../dist/sse.js"

const parsed = parseSseFrame(
  [
    "event: completion",
    "id: 7",
    'data: {"type":"completion","data":{"text":"ok"}}',
  ].join("\n"),
)

assert.deepEqual(parsed, {
  event: "completion",
  id: 7,
  data: {
    type: "completion",
    data: {
      text: "ok",
    },
  },
})

const normalized = normalizeSseEvent(parsed)

assert.deepEqual(normalized, {
  type: "completion",
  data: {
    text: "ok",
  },
})

const sseChunks = [
  "event: completion\r",
  '\nid: 7\r\ndata: {"type":"completion",\r\n',
  'data: "data":{"text":"ok"}}\r\n\r\n',
]

const sseResponse = new Response(
  new ReadableStream({
    start(controller) {
      for (const chunk of sseChunks) {
        controller.enqueue(new TextEncoder().encode(chunk))
      }

      controller.close()
    },
  }),
)

const sseEvents = []

for await (const event of readSseEvents(sseResponse)) {
  sseEvents.push(event)
}

assert.equal(sseEvents.length, 1)
assert.equal(sseEvents[0].event, "completion")
assert.equal(sseEvents[0].id, 7)
assert.deepEqual(sseEvents[0].data, {
  type: "completion",
  data: {
    text: "ok",
  },
})

const dataChunks = [
  '0: {"type":"token",',
  '"data":{"text":"hello"}}\n',
  "ignored-line\n",
]

const dataResponse = new Response(
  new ReadableStream({
    start(controller) {
      for (const chunk of dataChunks) {
        controller.enqueue(new TextEncoder().encode(chunk))
      }

      controller.close()
    },
  }),
)

const dataEvents = []

for await (const event of readDataStreamEvents(dataResponse)) {
  dataEvents.push(event)
}

assert.deepEqual(dataEvents, [
  {
    type: "token",
    data: {
      text: "hello",
    },
  },
])

console.log("sse-parser.test.mjs passed")