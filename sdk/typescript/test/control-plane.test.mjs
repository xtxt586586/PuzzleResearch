import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

const calls = []

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async (input, init = {}) => {
    const url = new URL(String(input))

    calls.push({
      method: init.method ?? "GET",
      path: url.pathname,
    })

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

await client.getActiveTasks()
await client.deleteSession("thread/one")
await client.getInterruptStatus("thread/one")
await client.listRuns()
await client.getRunMetrics("thread/one")
await client.getTrace("thread/one")
await client.getTraceSummary("thread/one")
await client.getAllTraces("thread/one")

assert.deepEqual(calls, [
  {
    method: "GET",
    path: "/api/tasks/active",
  },
  {
    method: "DELETE",
    path: "/api/sessions/thread%2Fone",
  },
  {
    method: "GET",
    path: "/api/interrupt/thread%2Fone/status",
  },
  {
    method: "GET",
    path: "/api/runs",
  },
  {
    method: "GET",
    path: "/api/runs/thread%2Fone",
  },
  {
    method: "GET",
    path: "/api/traces/thread%2Fone",
  },
  {
    method: "GET",
    path: "/api/traces/thread%2Fone/summary",
  },
  {
    method: "GET",
    path: "/api/traces/thread%2Fone/all",
  },
])

console.log("control-plane.test.mjs passed")