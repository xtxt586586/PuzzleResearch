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

    if (url.pathname === "/api/documents/upload") {
      assert.ok(init.body instanceof FormData)

      const uploadedFile = init.body.get("file")

      assert.ok(uploadedFile)
      assert.equal(uploadedFile.name, "notes.txt")
    }

    return new Response(
      JSON.stringify({
        ok: true,
        success: true,
        filename: "notes.txt",
        chunks: 1,
        message: "uploaded",
        total_chunks: 1,
        documents: [],
        source: "folder/notes.txt",
        query: "research",
        results: [],
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

const file = new Blob(
  ["PuzzleResearch document"],
  {
    type: "text/plain",
  },
)

const upload = await client.uploadDocument(
  file,
  "notes.txt",
)
assert.equal(upload.success, true)
assert.equal(upload.chunks, 1)

const listed = await client.listDocuments(20)
assert.equal(listed.total_chunks, 1)

const deleted = await client.deleteDocument(
  "folder/notes.txt",
)
assert.equal(deleted.source, "folder/notes.txt")

const searched = await client.searchDocuments(
  "research",
  8,
)
assert.equal(searched.query, "research")

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    [
      "POST",
      "/api/documents/upload",
      "",
    ],
    [
      "GET",
      "/api/documents/list",
      "?limit=20",
    ],
    [
      "DELETE",
      "/api/documents/folder%2Fnotes.txt",
      "",
    ],
    [
      "POST",
      "/api/documents/search",
      "?query=research&n_results=8",
    ],
  ],
)

console.log("documents.test.mjs passed")
