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

    if (url.pathname === "/api/asr/upload") {
      assert.ok(init.body instanceof FormData)

      const file = init.body.get("file")

      assert.ok(file)
      assert.equal(file.name, "audio.wav")
    }

    if (url.pathname === "/api/screenshots/example.png") {
      return new Response("image-bytes", {
        status: 200,
        headers: {
          "content-type": "image/png",
        },
      })
    }

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

await client.recognizeSpeech({
  audio_data: "YXVkaW8=",
  format: "wav",
  sample_rate: 16000,
})

await client.recognizeSpeechUpload(
  new Blob(["audio"], {
    type: "audio/wav",
  }),
  "audio.wav",
  16000,
)

await client.getAsrStatus()

await client.synthesizeSpeech({
  text: "hello",
  voice: "default",
})

await client.getTtsVoices()
await client.getTtsStatus()

await client.diagnoseBrowser(true)

const screenshot = await client.getScreenshot(
  "example.png",
)

assert.equal(await screenshot.text(), "image-bytes")

await client.listScreenshots("thread/one", 10)
await client.cleanupScreenshots()

await client.getBrowserSessionInfo("thread/one")
await client.triggerBrowserScreenshot("thread/one")

assert.deepEqual(
  calls.map((call) => [
    call.method,
    call.path,
    call.search,
  ]),
  [
    ["POST", "/api/asr/recognize", ""],
    ["POST", "/api/asr/upload", "?sample_rate=16000"],
    ["GET", "/api/asr/status", ""],
    ["POST", "/api/tts/synthesize", ""],
    ["GET", "/api/tts/voices", ""],
    ["GET", "/api/tts/status", ""],
    ["GET", "/api/sandbox/browser/diagnose", "?deep=true"],
    ["GET", "/api/screenshots/example.png", ""],
    [
      "GET",
      "/api/screenshots",
      "?thread_id=thread%2Fone&limit=10",
    ],
    ["POST", "/api/screenshots/cleanup", ""],
    [
      "GET",
      "/api/browser/thread%2Fone/info",
      "",
    ],
    [
      "POST",
      "/api/browser/thread%2Fone/screenshot",
      "",
    ],
  ],
)

console.log("audio-browser.test.mjs passed")