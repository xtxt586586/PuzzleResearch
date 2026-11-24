import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

const calls = []
const controller = new AbortController()

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async (input, init = {}) => {
    const url = new URL(String(input))

    calls.push({
      method: init.method ?? "GET",
      path: url.pathname,
      search: url.search,
      signal: init.signal,
      body: init.body,
    })

    if (url.pathname.startsWith("/api/export/")) {
      return new Response("export", {
        status: 200,
        headers: {
          "content-type": "text/html",
        },
      })
    }

    return new Response(
      JSON.stringify({
        ok: true,
      }),
      {
        status: 200,
        headers: {
          "content-type": "application/json",
        },
      },
    )
  },
})

await client.chat(
  {
    messages: [
      {
        role: "user",
        content: "hello",
      },
    ],
  },
  {
    signal: controller.signal,
  },
)

await client.cancelChat(
  "thread/one",
  {
    reason: "user_cancelled",
    signal: controller.signal,
  },
)

await client.cancelAllChats({
  signal: controller.signal,
})

await client.resumeInterrupt(
  {
    thread_id: "thread/one",
    payload: {
      approved: true,
    },
  },
  {
    signal: controller.signal,
  },
)

await client.resumeThreadInterrupt(
  "thread/one",
  {
    approved: true,
  },
  {
    signal: controller.signal,
  },
)

await client.getMcpConfig({
  signal: controller.signal,
})

await client.updateMcpConfig(
  {
    enable: true,
  },
  {
    signal: controller.signal,
  },
)

await client.getExportTemplates({
  signal: controller.signal,
})

const exported = await client.downloadExport(
  "thread/one",
  {
    format: "json",
    title: "Research",
    template: "default",
    signal: controller.signal,
  },
)

assert.equal(await exported.text(), "export")

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    ["POST", "/api/chat", ""],
    ["POST", "/api/chat/cancel/thread%2Fone", ""],
    ["POST", "/api/chat/cancel-all", ""],
    ["POST", "/api/interrupt/resume", ""],
    [
      "POST",
      "/api/interrupt/thread%2Fone/resume",
      "",
    ],
    ["GET", "/api/mcp/config", ""],
    ["POST", "/api/mcp/config", ""],
    ["GET", "/api/export/templates", ""],
    [
      "GET",
      "/api/export/thread%2Fone",
      "?format=json&title=Research&template=default",
    ],
  ],
)

for (const call of calls) {
  assert.equal(call.signal, controller.signal)
}

assert.deepEqual(
  JSON.parse(calls[1].body),
  {
    reason: "user_cancelled",
  },
)

assert.deepEqual(
  JSON.parse(calls[6].body),
  {
    enable: true,
  },
)

console.log("request-options.test.mjs passed")