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
      search: url.search,
      body: init.body,
    })

    if (url.pathname === "/metrics") {
      return new Response(
        "# HELP puzzleresearch_requests_total\n",
        {
          status: 200,
          headers: {
            "content-type": "text/plain; version=0.0.4",
          },
        },
      )
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

await client.getRoot()
await client.getHealth()
const metrics = await client.getMetrics()
assert.equal(metrics, "# HELP puzzleresearch_requests_total\n")

await client.getMemoryStatus()
await client.supportChat({
  message: "How does research work?",
  user_id: "user-001",
})

await client.getSearchCacheStats()
await client.clearSearchCache()

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    ["GET", "/", ""],
    ["GET", "/health", ""],
    ["GET", "/metrics", ""],
    ["GET", "/api/memory/status", ""],
    ["POST", "/api/support/chat", ""],
    ["GET", "/api/search/cache/stats", ""],
    ["POST", "/api/search/cache/clear", ""],
  ],
)

assert.deepEqual(
  JSON.parse(calls[4].body),
  {
    message: "How does research work?",
    user_id: "user-001",
  },
)

console.log("support.test.mjs passed")
