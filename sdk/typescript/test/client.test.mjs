import assert from "node:assert/strict"
import {
  PuzzleResearchApiError,
  PuzzleResearchClient,
} from "../dist/index.js"

const requests = []

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async (input, init) => {
    requests.push({
      input: String(input),
      init,
    })

    return new Response(
      JSON.stringify({
        thread_id: "thread-001",
        checkpoint_id: "checkpoint-001",
      }),
      {
        status: 200,
        headers: {
          "content-type": "application/json",
          "X-Thread-ID": "thread-001",
        },
      },
    )
  },
})

const state = await client.getSessionState("thread/state")

assert.equal(state.thread_id, "thread-001")
assert.equal(state.checkpoint_id, "checkpoint-001")
assert.equal(client.lastThreadId, "thread-001")
assert.equal(
  requests[0].input,
  "http://example.test/api/sessions/thread%2Fstate/state",
)

assert.equal(requests[0].init.method, "GET")

const body = JSON.stringify({
  error: "bad request",
})

const failingClient = new PuzzleResearchClient({
  baseUrl: "http://example.test",
  fetch: async () =>
    new Response(body, {
      status: 400,
      headers: {
        "content-type": "application/json",
      },
    }),
})

await assert.rejects(
  () => failingClient.getSessionState("missing"),
  (error) => {
    assert.ok(error instanceof PuzzleResearchApiError)
    assert.equal(error.status, 400)
    assert.deepEqual(error.payload, {
      error: "bad request",
    })
    return true
  },
)

console.log("client.test.mjs passed")