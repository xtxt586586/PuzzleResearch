import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import * as sdk from "../dist/index.js"

assert.equal(
  typeof sdk.PuzzleResearchClient,
  "function",
)

assert.equal(
  typeof sdk.PuzzleResearchApiError,
  "function",
)

assert.equal(
  typeof sdk.parseSseFrame,
  "function",
)

assert.equal(
  typeof sdk.normalizeSseEvent,
  "function",
)

assert.equal(
  typeof sdk.readSseEvents,
  "function",
)

assert.equal(
  typeof sdk.readDataStreamEvents,
  "function",
)

const client = new sdk.PuzzleResearchClient({
  baseUrl: "http://example.test",
})

assert.equal(
  client.baseUrl,
  "http://example.test",
)

const defaultClient =
  new sdk.PuzzleResearchClient()

assert.equal(
  defaultClient.baseUrl,
  "http://127.0.0.1:8001",
)

const declaration = await readFile(
  new URL("../dist/index.d.ts", import.meta.url),
  "utf8",
)

assert.match(
  declaration,
  /PuzzleResearchClient/,
)

const packageJson = JSON.parse(
  await readFile(
    new URL("../package.json", import.meta.url),
    "utf8",
  ),
)

assert.equal(
  packageJson.exports["."].default,
  "./dist/index.js",
)

assert.equal(
  packageJson.exports["."].types,
  "./dist/index.d.ts",
)

assert.deepEqual(
  packageJson.files,
  [
    "dist",
    "src",
  ],
)

console.log("exports.test.mjs passed")