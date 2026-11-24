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

await client.createShare(
  "thread/one",
  {
    permissions: "view",
    expires_hours: 72,
  },
)

await client.getShare("share/one")
await client.deleteShare("share/one")

await client.addSessionComment(
  "thread/one",
  {
    content: "Important finding",
    message_id: "message-001",
  },
)

await client.getSessionComments(
  "thread/one",
  "message/001",
)

await client.getSessionVersions("thread/one")

await client.createSessionVersion(
  "thread/one",
  "Before revision",
)

await client.restoreSessionVersion(
  "thread/one",
  "version/001",
)

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    [
      "POST",
      "/api/sessions/thread%2Fone/share",
      "",
    ],
    [
      "GET",
      "/api/share/share%2Fone",
      "",
    ],
    [
      "DELETE",
      "/api/share/share%2Fone",
      "",
    ],
    [
      "POST",
      "/api/sessions/thread%2Fone/comments",
      "",
    ],
    [
      "GET",
      "/api/sessions/thread%2Fone/comments",
      "?message_id=message%2F001",
    ],
    [
      "GET",
      "/api/sessions/thread%2Fone/versions",
      "",
    ],
    [
      "POST",
      "/api/sessions/thread%2Fone/versions",
      "?label=Before+revision",
    ],
    [
      "POST",
      "/api/sessions/thread%2Fone/restore/version%2F001",
      "",
    ],
  ],
)

assert.deepEqual(
  JSON.parse(calls[0].body),
  {
    permissions: "view",
    expires_hours: 72,
  },
)

console.log("collaboration.test.mjs passed")