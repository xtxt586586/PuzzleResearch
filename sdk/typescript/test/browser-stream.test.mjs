import assert from "node:assert/strict"
import { PuzzleResearchClient } from "../dist/index.js"

class FakeWebSocket {
  static instances = []

  constructor(url, protocols) {
    this.url = url
    this.protocols = protocols
    this.readyState = 0
    this.sent = []
    this.closed = false

    FakeWebSocket.instances.push(this)
  }

  send(value) {
    this.sent.push(value)
  }

  close() {
    this.closed = true
    this.readyState = 3
  }
}

const client = new PuzzleResearchClient({
  baseUrl: "http://example.test",
})

const socket = client.openBrowserStream(
  "thread/one",
  {
    webSocket: FakeWebSocket,
  },
)

assert.equal(
  socket.url,
  "ws://example.test/api/browser/thread%2Fone/stream",
)

assert.equal(
  socket.protocols,
  undefined,
)

assert.equal(
  FakeWebSocket.instances.length,
  1,
)

socket.send(
  JSON.stringify({
    action: "start",
    quality: 80,
    max_fps: 5,
  }),
)

assert.deepEqual(socket.sent, [
  JSON.stringify({
    action: "start",
    quality: 80,
    max_fps: 5,
  }),
])

socket.close()

assert.equal(socket.closed, true)

console.log("browser-stream.test.mjs passed")