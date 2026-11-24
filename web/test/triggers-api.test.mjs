import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(
  new URL("../lib/triggers-api.ts", import.meta.url),
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
  const payload = String(url).includes("/executions")
    ? { executions: [{ id: "execution-1", status: "success" }] }
    : String(url).includes("/api/triggers?")
      ? { triggers: [{ id: "trigger-1" }] }
      : init.method === "POST" && String(url).endsWith("/webhook")
        ? {
            success: true,
            trigger_id: "trigger-1",
            trigger: { id: "trigger-1", endpoint_path: "/api/webhook/trigger-1" },
            auth_token: "one-time-token",
          }
        : { success: true }
  return {
    ok: true,
    status: 200,
    json: async () => payload,
  }
}

try {
  const listed = await api.listTriggers({
    triggerType: "scheduled",
    status: "active",
  })
  assert.equal(listed[0].id, "trigger-1")
  assert.match(requests[0].url, /trigger_type=scheduled/)
  assert.match(requests[0].url, /status=active/)

  const created = await api.createWebhookTrigger({
    name: "webhook",
    task: "run research",
    require_auth: true,
    http_methods: ["POST"],
    agent_id: "default",
    task_params: {},
  })
  assert.equal(created.auth_token, "one-time-token")
  assert.equal(requests[1].init.method, "POST")
  assert.equal(JSON.parse(requests[1].init.body).require_auth, true)

  await api.pauseTrigger("thread/one")
  assert.match(requests[2].url, /thread%2Fone\/pause$/)

  const executions = await api.getTriggerExecutions("trigger-1")
  assert.equal(executions[0].id, "execution-1")
  assert.equal(
    api.getWebhookUrl("/api/webhook/trigger-1"),
    "http://127.0.0.1:9000/api/webhook/trigger-1",
  )

  global.fetch = async () => {
    throw new TypeError("Failed to fetch")
  }
  await assert.rejects(
    api.listTriggers(),
    /无法连接后端服务/,
  )
} finally {
  global.fetch = originalFetch
}

console.log("triggers-api.test.mjs passed")
