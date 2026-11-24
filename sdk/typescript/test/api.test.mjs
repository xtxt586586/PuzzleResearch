import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

const calls = []

function jsonResponse(payload) {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      "content-type": "application/json",
      "X-Thread-ID": "thread-001",
    },
  })
}

function sseResponse() {
  return new Response(
    [
      "event: message",
      'data: {"ok":true}',
      "",
    ].join("\n"),
    {
      status: 200,
      headers: {
        "content-type": "text/event-stream",
        "X-Thread-ID": "thread-001",
      },
    },
  )
}

function dataProtocolResponse() {
  return new Response(
    '0: {"type":"data","data":{"ok":true}}\n',
    {
      status: 200,
      headers: {
        "content-type": "text/plain",
        "X-Thread-ID": "thread-001",
      },
    },
  )
}

async function fakeFetch(input, init = {}) {
  const url = new URL(String(input))

  calls.push({
  method: init.method ?? "GET",
  path: url.pathname,
  search: url.search,
  init,
})

  if (url.pathname === "/api/chat/sse") {
    return sseResponse()
  }

  if (url.pathname === "/api/mcp/diagnose") {
    return new Response(
      JSON.stringify({
        enabled: true,
        status: "degraded",
        ready: false,
        server_count: 1,
        connected_count: 0,
        servers: [
          {
            server_id: "demo",
            transport: "sse",
            status: "not_loaded",
            loaded_tools: 0,
            message: "服务器尚未成功加载。",
          },
        ],
      }),
      {
        status: 200,
        headers: {
          "content-type": "application/json",
        },
      },
    )
  }

  if (url.pathname === "/api/research/sse") {
    return sseResponse()
  }

  if (url.pathname === "/api/events/thread%2Fone") {
    return sseResponse()
  }

  if (url.pathname === "/api/research") {
    return dataProtocolResponse()
  }

  if (url.pathname === "/api/export/thread%2Fone") {
    return new Response("export-data", {
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "X-Thread-ID": "thread-001",
      },
    })
  }

  return jsonResponse({
    ok: true,
    path: url.pathname,
  })
}

async function collect(iterable) {
  const result = []

  for await (const item of iterable) {
    result.push(item)
  }

  return result
}

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: fakeFetch,
})

await client.chat({
  message: "hello",
})

const chatEvents = await collect(
  client.chatSse({
    message: "hello",
  }),
)

assert.equal(chatEvents.length, 1)
assert.equal(chatEvents[0].type, "message")

const researchEvents = await collect(
  client.research({
    query: "test",
  }),
)

assert.equal(researchEvents.length, 1)
assert.deepEqual(researchEvents[0].data, {
  ok: true,
})

const researchSseEvents = await collect(
  client.researchSse({
    query: "test",
  }),
)

assert.equal(researchSseEvents.length, 1)

const runtimeEvents = await collect(
  client.events("thread/one"),
)

assert.equal(runtimeEvents.length, 1)

await client.cancelChat("thread/one")
await client.cancelAllChats()

await client.listSessions()
await client.getSessions()
await client.getSession("thread/one")
await client.getSessionState("thread/one")
await client.getSessionEvidence("thread/one")
await client.getResearchTree("thread/one")

await client.resumeSession("thread/one", {
  checkpoint_id: "checkpoint-001",
})

await client.resumeInterrupt({
  thread_id: "thread/one",
})

await client.getMcpConfig()
await client.updateMcpConfig({
  enabled: true,
})

const mcpDiagnosis = await client.diagnoseMcp()
assert.equal(mcpDiagnosis.status, "degraded")
assert.equal(mcpDiagnosis.servers[0].server_id, "demo")
await client.diagnoseBrowser()
await client.getExportTemplates()

const exported = await client.exportSession("thread/one")
assert.equal(await exported.text(), "export-data")

const expectedRoutes = [
  ["POST", "/api/chat"],
  ["POST", "/api/chat/sse"],
  ["POST", "/api/research"],
  ["POST", "/api/research/sse"],
  ["GET", "/api/events/thread%2Fone"],
  ["POST", "/api/chat/cancel/thread%2Fone"],
  ["POST", "/api/chat/cancel-all"],
  ["GET", "/api/sessions"],
  ["GET", "/api/sessions"],
  ["GET", "/api/sessions/thread%2Fone"],
  ["GET", "/api/sessions/thread%2Fone/state"],
  ["GET", "/api/sessions/thread%2Fone/evidence"],
  ["GET", "/api/sessions/thread%2Fone/research-tree"],
  ["POST", "/api/sessions/thread%2Fone/resume"],
  ["POST", "/api/interrupt/resume"],
  ["GET", "/api/mcp/config"],
  ["POST", "/api/mcp/config"],
  ["GET", "/api/mcp/diagnose"],
  ["GET", "/api/sandbox/browser/diagnose"],
  ["GET", "/api/export/templates"],
  ["GET", "/api/export/thread%2Fone"],
]

assert.deepEqual(
  calls.map((call) => [call.method, call.path]),
  expectedRoutes,
)

assert.equal(client.lastThreadId, "thread-001")

console.log("api.test.mjs passed")

assert.equal(
  calls[2].init.method,
  "POST",
)

assert.equal(
  calls[2].init.body,
  undefined,
)

assert.equal(
  calls[2].search,
  "?query=test",
)
