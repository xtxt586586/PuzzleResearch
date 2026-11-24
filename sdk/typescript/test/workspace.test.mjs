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
      body: init.body,
    })

    if (url.pathname.endsWith("/workspace/files/upload")) {
      assert.ok(init.body instanceof FormData)
      const uploaded = init.body.get("file")
      assert.ok(uploaded)
      assert.equal(uploaded.name, "研究报告.md")
      return new Response(
        JSON.stringify({
          thread_id: "thread/one",
          path: "研究报告.md",
          bytes: 6,
        }),
        { status: 201 },
      )
    }

    if (
      init.method === "GET" &&
      url.pathname.endsWith("research%20draft.txt")
    ) {
      return new Response("draft", {
        status: 200,
        headers: { "content-type": "text/plain" },
      })
    }

    if (init.method === "DELETE") {
      return new Response(
        JSON.stringify({
          thread_id: "thread/one",
          path: "notes/research draft.txt",
        }),
        { status: 200 },
      )
    }

    return new Response(
      JSON.stringify({
        thread_id: "thread/one",
        count: 1,
        files: ["notes/research.txt"],
      }),
      { status: 200 },
    )
  },
})

const listed = await client.listWorkspaceFiles("thread/one")
const uploaded = await client.uploadWorkspaceFile(
  "thread/one",
  new Blob(["草稿"], { type: "text/markdown" }),
  "研究报告.md",
)
const downloaded = await client.downloadWorkspaceFile(
  "thread/one",
  "notes/research draft.txt",
)
const deleted = await client.deleteWorkspaceFile(
  "thread/one",
  "notes/research draft.txt",
)

assert.deepEqual(listed.files, ["notes/research.txt"])
assert.equal(uploaded.path, "研究报告.md")
assert.equal(await downloaded.text(), "draft")
assert.equal(deleted.path, "notes/research draft.txt")

assert.deepEqual(
  calls.map((call) => [call.method, call.path]),
  [
    ["GET", "/api/sessions/thread%2Fone/workspace/files"],
    ["POST", "/api/sessions/thread%2Fone/workspace/files/upload"],
    [
      "GET",
      "/api/sessions/thread%2Fone/workspace/files/notes/research%20draft.txt",
    ],
    [
      "DELETE",
      "/api/sessions/thread%2Fone/workspace/files/notes/research%20draft.txt",
    ],
  ],
)

console.log("workspace.test.mjs passed")
