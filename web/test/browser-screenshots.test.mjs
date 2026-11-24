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
global.fetch = async (url, init) => {
  requests.push({ url: String(url), init })
  return {
    ok: true,
    status: 200,
    json: async () => ({
      screenshots: [
        {
          url: "/api/screenshots/thread_browser.png",
          filename: "thread_browser.png",
          action: "browser_manual",
          page_url: "https://example.com/research",
          title: "Research page",
          mime_type: "image/png",
          created_at: "2026-09-28T12:00:00+00:00",
        },
      ],
    }),
  }
}

try {
  const screenshots = await api.listThreadScreenshots("thread/one", 25)
  assert.equal(screenshots.length, 1)
  assert.equal(screenshots[0].action, "browser_manual")
  assert.equal(screenshots[0].page_url, "https://example.com/research")
  assert.equal(screenshots[0].title, "Research page")
  assert.match(requests[0].url, /thread_id=thread%2Fone/)
  assert.match(requests[0].url, /limit=25/)
  assert.equal(requests[0].init.headers.Accept, "application/json")

  global.fetch = async () => ({ ok: false, status: 403 })
  await assert.rejects(
    api.listThreadScreenshots("thread/one"),
    /截图记录加载失败：403/,
  )
} finally {
  global.fetch = originalFetch
}

console.log("browser-screenshots.test.mjs passed")
