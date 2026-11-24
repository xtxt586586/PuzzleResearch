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

await client.createScheduledTrigger({
  name: "Daily research",
  schedule: "0 9 * * *",
  task: "Run research",
})

await client.createWebhookTrigger({
  name: "Webhook research",
  task: "Run webhook task",
})

await client.createEventTrigger({
  name: "Completion handler",
  event_type: "research.completed",
  task: "Process completion",
})

await client.listTriggers({
  triggerType: "event",
  status: "active",
})

await client.getTrigger("trigger/one")
await client.pauseTrigger("trigger/one")
await client.resumeTrigger("trigger/one")

await client.getTriggerExecutions(
  "trigger/one",
  {
    limit: 20,
  },
)

await client.invokeWebhook(
  "trigger/one",
  {
    value: 1,
  },
)

await client.deleteTrigger("trigger/one")

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    ["POST", "/api/triggers/scheduled", ""],
    ["POST", "/api/triggers/webhook", ""],
    ["POST", "/api/triggers/event", ""],
    [
      "GET",
      "/api/triggers",
      "?trigger_type=event&status=active",
    ],
    ["GET", "/api/triggers/trigger%2Fone", ""],
    [
      "POST",
      "/api/triggers/trigger%2Fone/pause",
      "",
    ],
    [
      "POST",
      "/api/triggers/trigger%2Fone/resume",
      "",
    ],
    [
      "GET",
      "/api/triggers/trigger%2Fone/executions",
      "?limit=20",
    ],
    ["POST", "/api/webhook/trigger%2Fone", ""],
    ["DELETE", "/api/triggers/trigger%2Fone", ""],
  ],
)

assert.deepEqual(
  JSON.parse(calls[0].body),
  {
    name: "Daily research",
    schedule: "0 9 * * *",
    task: "Run research",
  },
)

assert.deepEqual(
  JSON.parse(calls[8].body),
  {
    value: 1,
  },
)

console.log("triggers.test.mjs passed")