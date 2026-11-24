import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"

const manifest = JSON.parse(
  await readFile(new URL("../app/manifest.json", import.meta.url), "utf8"),
)

assert.equal(manifest.name, "PuzzleResearch")
assert.equal(manifest.short_name, "PuzzleResearch")
assert.equal(manifest.start_url, "/")
assert.equal(manifest.display, "standalone")
assert.equal(manifest.theme_color, "#157eb8")
assert.equal(manifest.icons.length, 1)
assert.equal(manifest.icons[0].src, "/icon.svg")
assert.equal(manifest.icons[0].type, "image/svg+xml")
assert.match(manifest.icons[0].purpose, /maskable/)

const icon = await readFile(new URL("../app/icon.svg", import.meta.url), "utf8")
assert.match(icon, /<svg\s/)
assert.match(icon, /viewBox="0 0 512 512"/)

console.log("manifest.test.mjs passed")
