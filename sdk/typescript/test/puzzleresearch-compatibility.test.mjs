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
    })

    if (url.pathname === "/api/export/thread%2Fone") {
      return new Response("export-data", {
        status: 200,
        headers: { "content-type": "text/markdown" },
      })
    }

    if (url.pathname === "/api/health") {
      return new Response("healthy", { status: 200 })
    }

    return new Response(
      JSON.stringify({
        sources: [],
        claims: [],
        quality_summary: {},
        fetched_pages: [],
        passages: [],
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    )
  },
})

const evidence = await client.getEvidence("thread/one")
const raw = await client.requestRaw("/api/health")
const exported = await client.exportReport("thread/one", {
  format: "md",
})

assert.deepEqual(evidence.sources, [])
assert.equal(await raw.text(), "healthy")
assert.equal(exported.contentType, "text/markdown")
assert.equal(
  new TextDecoder().decode(exported.bytes),
  "export-data",
)

assert.deepEqual(
  calls,
  [
    {
      method: "GET",
      path: "/api/sessions/thread%2Fone/evidence",
      search: "",
    },
    {
      method: "GET",
      path: "/api/health",
      search: "",
    },
    {
      method: "GET",
      path: "/api/export/thread%2Fone",
      search: "?format=md",
    },
  ],
)

console.log("puzzleresearch-compatibility.test.mjs passed")
