import assert from "node:assert/strict"

import { PuzzleResearchClient } from "../dist/index.js"

const calls = []
const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async (input, init = {}) => {
    const url = new URL(String(input))
    calls.push([init.method ?? "GET", url.pathname])
    return new Response(JSON.stringify({
      sources: [{ url: "https://example.com" }],
      claims: [{
        claim: "收入增长",
        status: "supported",
        evidence_urls: ["https://example.com"],
        evidence_passages: [{
          url: "https://example.com",
          snippet_hash: "passage-001",
        }],
        score: 0.9,
        notes: "有直接证据",
      }],
      quality_summary: { overall_score: 0.9 },
      fetched_pages: [],
      passages: [],
    }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  },
})

const result = await client.getSessionEvidence("thread/quality")
assert.equal(result.quality_summary.overall_score, 0.9)
assert.equal(
  result.claims[0].evidence_passages[0].snippet_hash,
  "passage-001",
)
assert.deepEqual(calls, [["GET", "/api/sessions/thread%2Fquality/evidence"]])
console.log("research-quality.test.mjs passed")
