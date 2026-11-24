import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(
  new URL("../lib/api.ts", import.meta.url),
  "utf8",
)
const compiled = ts.transpile(source, {
  module: ts.ModuleKind.ES2022,
  target: ts.ScriptTarget.ES2022,
})
const api = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
)

const originalFetch = global.fetch
const requests = []
global.fetch = async (url, init = {}) => {
  requests.push({ url: String(url), init })
  return {
    ok: true,
    status: 200,
    json: async () => ({
      enabled: true,
      status: "degraded",
      ready: false,
      server_count: 2,
      connected_count: 1,
      servers: [
        {
          server_id: "local-tools",
          transport: "stdio",
          status: "connected",
          loaded_tools: 3,
          message: null,
        },
        {
          server_id: "remote-tools",
          transport: "sse",
          status: "failed",
          loaded_tools: 0,
          message: "连接或工具发现失败；详细异常不会通过此接口返回。",
        },
      ],
    }),
  }
}

try {
  const result = await api.getMcpDiagnostics()
  assert.equal(requests[0].url, "http://127.0.0.1:9000/api/mcp/diagnose")
  assert.equal(requests[0].init.cache, "no-store")
  assert.equal(requests[0].init.headers.Accept, "application/json")
  assert.equal(result.status, "degraded")
  assert.equal(result.connected_count, 1)
  assert.equal(result.servers[0].loaded_tools, 3)
  assert.equal(result.servers[1].status, "failed")

  global.fetch = async () => ({
    ok: false,
    status: 503,
    json: async () => ({}),
  })
  await assert.rejects(api.getMcpDiagnostics(), /MCP 诊断读取失败：503/)
} finally {
  global.fetch = originalFetch
}

console.log("mcp-diagnostics.test.mjs passed")
