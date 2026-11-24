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

    return new Response(
      JSON.stringify({
        ok: true,
        rag_embedding: {
          provider: "local-hash",
          available: true,
          dimensions: 256,
          reason: "",
        },
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

const health = await client.getAgentHealth()
assert.equal(health.rag_embedding.provider, "local-hash")
assert.equal(health.rag_embedding.available, true)
await client.getToolRegistry()
await client.refreshToolRegistry(true)
await client.getSearchProviders()
await client.resetSearchProviders()
await client.getPublicConfig()

await client.listAgents()
await client.getAgent("agent/one")

await client.createAgent({
  id: "agent-one",
  name: "Agent One",
})

await client.updateAgent("agent/one", {
  name: "Updated Agent",
})

await client.deleteAgent("agent/one")

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    ["GET", "/api/health/agent", ""],
    ["GET", "/api/tools/registry", ""],
    ["POST", "/api/tools/registry/refresh", "?reset=true"],
    ["GET", "/api/search/providers", ""],
    ["POST", "/api/search/providers/reset", ""],
    ["GET", "/api/config/public", ""],
    ["GET", "/api/agents", ""],
    ["GET", "/api/agents/agent%2Fone", ""],
    ["POST", "/api/agents", ""],
    ["PUT", "/api/agents/agent%2Fone", ""],
    ["DELETE", "/api/agents/agent%2Fone", ""],
  ],
)

assert.deepEqual(
  JSON.parse(calls[8].body),
  {
    id: "agent-one",
    name: "Agent One",
  },
)

assert.deepEqual(
  JSON.parse(calls[9].body),
  {
    name: "Updated Agent",
  },
)

console.log("capabilities.test.mjs passed")
