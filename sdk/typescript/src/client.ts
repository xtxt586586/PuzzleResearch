export type JsonObject = Record<string, unknown>

export interface SseEvent {
  event: string
  id?: string
  data: string
  value?: unknown
}

export interface PuzzleResearchClientOptions {
  baseUrl?: string
  headers?: HeadersInit
  fetch?: typeof fetch
}

export class PuzzleResearchApiError extends Error {
  readonly status: number
  readonly payload: unknown

  constructor(status: number, payload: unknown) {
    const message =
      typeof payload === "string" && payload.length > 0
        ? payload
        : `PuzzleResearch API request failed with status ${status}`

    super(message)
    this.name = "PuzzleResearchApiError"
    this.status = status
    this.payload = payload
  }
}

type JsonRequestInit = Omit<RequestInit, "body"> & {
  body?: unknown
}

export class PuzzleResearchClient {
  readonly baseUrl: string

  private readonly defaultHeaders: Headers
  private readonly fetcher: typeof fetch
  private currentThreadId: string | null = null

  get lastThreadId(): string | null {
    return this.currentThreadId
  }

  get threadId(): string | null {
    return this.currentThreadId
  }

  constructor(options: PuzzleResearchClientOptions = {}) {
    this.baseUrl = (
      options.baseUrl ??
      "http://127.0.0.1:8000"
    ).replace(/\/+$/, "")

    this.defaultHeaders = new Headers(options.headers)
    this.fetcher = options.fetch ?? fetch
  }

  async requestJson<T>(
    path: string,
    init: JsonRequestInit = {},
  ): Promise<T> {
    return this.request<T>(path, init)
  }

  async chat<T = JsonObject>(
    payload: JsonObject,
  ): Promise<T> {
    return this.request<T>("/api/chat", {
      method: "POST",
      body: payload,
    })
  }

  chatSse(
    payload: JsonObject,
  ): AsyncGenerator<SseEvent> {
    return this.stream("/api/chat/sse", {
      method: "POST",
      body: payload,
    })
  }

  research(
    payload: JsonObject,
  ): AsyncGenerator<SseEvent> {
    return this.streamDataProtocol("/api/research", {
      method: "POST",
      body: payload,
    })
  }

  researchSse(
    payload: JsonObject,
  ): AsyncGenerator<SseEvent> {
    return this.stream("/api/research/sse", {
      method: "POST",
      body: payload,
    })
  }

  events(
    threadId: string,
  ): AsyncGenerator<SseEvent> {
    return this.stream(
      `/api/events/${this.encode(threadId)}`,
      {
        method: "GET",
      },
    )
  }

  async cancelChat<T = JsonObject>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/chat/cancel/${this.encode(threadId)}`,
      {
        method: "POST",
      },
    )
  }

  async cancelAllChats<T = JsonObject>(): Promise<T> {
    return this.request<T>("/api/chat/cancel-all", {
      method: "POST",
    })
  }

  async listSessions<T = unknown>(): Promise<T> {
    return this.request<T>("/api/sessions")
  }

  async getSessions<T = unknown>(): Promise<T> {
    return this.listSessions<T>()
  }

  async getSession<T = JsonObject>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}`,
    )
  }

  async getSessionState<T = JsonObject>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/state`,
    )
  }

  async getSessionEvidence<T = JsonObject>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/evidence`,
    )
  }

  async getResearchTree<T = JsonObject>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/research-tree`,
    )
  }

  async resumeSession<T = JsonObject>(
    threadId: string,
    payload: JsonObject = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/resume`,
      {
        method: "POST",
        body: payload,
      },
    )
  }

  async resumeInterrupt<T = JsonObject>(
    payload: JsonObject,
  ): Promise<T> {
    return this.request<T>("/api/interrupt/resume", {
      method: "POST",
      body: payload,
    })
  }

  async getMcpConfig<T = JsonObject>(): Promise<T> {
    return this.request<T>("/api/mcp/config")
  }

  async updateMcpConfig<T = JsonObject>(
    payload: JsonObject,
  ): Promise<T> {
    return this.request<T>("/api/mcp/config", {
      method: "POST",
      body: payload,
    })
  }

  async diagnoseBrowser<T = JsonObject>(): Promise<T> {
    return this.request<T>("/api/sandbox/browser/diagnose")
  }

  async getExportTemplates<T = unknown>(): Promise<T> {
    return this.request<T>("/api/export/templates")
  }

  async exportSession(
    threadId: string,
  ): Promise<Blob> {
    return this.requestBlob(
      `/api/export/${this.encode(threadId)}`,
    )
  }

  async downloadExport(
    threadId: string,
  ): Promise<Blob> {
    return this.exportSession(threadId)
  }

  private async request<T>(
    path: string,
    init: JsonRequestInit = {},
  ): Promise<T> {
    const response = await this.fetcher(
      this.url(path),
      this.buildRequest(init),
    )

    this.captureThreadId(response)

    if (!response.ok) {
      throw new PuzzleResearchApiError(
        response.status,
        await this.readPayload(response),
      )
    }

    if (response.status === 204) {
      return undefined as T
    }

    const text = await response.text()

    if (text.length === 0) {
      return undefined as T
    }

    try {
      return JSON.parse(text) as T
    } catch {
      return text as T
    }
  }

  private async requestBlob(path: string): Promise<Blob> {
    const response = await this.fetcher(
      this.url(path),
      this.buildRequest({
        method: "GET",
      }),
    )

    this.captureThreadId(response)

    if (!response.ok) {
      throw new PuzzleResearchApiError(
        response.status,
        await this.readPayload(response),
      )
    }

    return response.blob()
  }

  private async *stream(
    path: string,
    init: JsonRequestInit,
  ): AsyncGenerator<SseEvent> {
    const response = await this.fetcher(
      this.url(path),
      this.buildRequest(init, true),
    )

    this.captureThreadId(response)

    if (!response.ok) {
      throw new PuzzleResearchApiError(
        response.status,
        await this.readPayload(response),
      )
    }

    if (!response.body) {
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const result = await reader.read()

      if (result.done) {
        break
      }

      buffer += decoder.decode(result.value, {
        stream: true,
      })

      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ""

      for (const block of blocks) {
        const event = this.parseSseBlock(block)

        if (event !== null) {
          yield event
        }
      }
    }

    buffer += decoder.decode()

    if (buffer.trim().length > 0) {
      const event = this.parseSseBlock(buffer)

      if (event !== null) {
        yield event
      }
    }
  }

  private async *streamDataProtocol(
    path: string,
    init: JsonRequestInit,
  ): AsyncGenerator<SseEvent> {
    const response = await this.fetcher(
      this.url(path),
      this.buildRequest(init, false),
    )

    this.captureThreadId(response)

    if (!response.ok) {
      throw new PuzzleResearchApiError(
        response.status,
        await this.readPayload(response),
      )
    }

    if (!response.body) {
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const result = await reader.read()

      if (result.done) {
        break
      }

      buffer += decoder.decode(result.value, {
        stream: true,
      })

      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ""

      for (const line of lines) {
        const event = this.parseDataProtocolLine(line)

        if (event !== null) {
          yield event
        }
      }
    }

    buffer += decoder.decode()

    const event = this.parseDataProtocolLine(buffer)

    if (event !== null) {
      yield event
    }
  }

  private parseSseBlock(
    block: string,
  ): SseEvent | null {
    let eventName = "message"
    let eventId: string | undefined
    const dataLines: string[] = []

    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith(":")) {
        continue
      }

      const separator = line.indexOf(":")
      const field =
        separator === -1
          ? line
          : line.slice(0, separator)

      let value =
        separator === -1
          ? ""
          : line.slice(separator + 1)

      if (value.startsWith(" ")) {
        value = value.slice(1)
      }

      if (field === "event") {
        eventName = value
      } else if (field === "id") {
        eventId = value
      } else if (field === "data") {
        dataLines.push(value)
      }
    }

    if (
      dataLines.length === 0 &&
      eventId === undefined &&
      eventName === "message"
    ) {
      return null
    }

    const data = dataLines.join("\n")
    let value: unknown

    if (data.length > 0) {
      try {
        value = JSON.parse(data)
      } catch {
        value = data
      }
    }

    return {
      event: eventName,
      id: eventId,
      data,
      value,
    }
  }

  private parseDataProtocolLine(
    line: string,
  ): SseEvent | null {
    const trimmed = line.trim()

    if (!trimmed.startsWith("0:")) {
      return null
    }

    const data = trimmed.slice(2).trim()
    let value: unknown

    try {
      value = JSON.parse(data)
    } catch {
      value = data
    }

    return {
      event: "data",
      data,
      value,
    }
  }

  private buildRequest(
    init: JsonRequestInit,
    sse = false,
  ): RequestInit {
    const {
      body: rawBody,
      headers: rawHeaders,
      ...requestInit
    } = init

    const headers = new Headers(this.defaultHeaders)

    if (sse) {
      headers.set("Accept", "text/event-stream")
    }

    if (rawBody !== undefined) {
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json")
      }
    }

    new Headers(rawHeaders).forEach((value, key) => {
      headers.set(key, value)
    })

    const body =
      rawBody === undefined ||
      rawBody === null ||
      typeof rawBody === "string"
        ? rawBody
        : JSON.stringify(rawBody)

    return {
        ...requestInit,
        method: requestInit.method ?? "GET",
        headers,
        body,
      }
  }

  private captureThreadId(
    response: Response,
  ): void {
    const threadId = response.headers.get("X-Thread-ID")

    if (threadId) {
      this.currentThreadId = threadId
    }
  }

  private async readPayload(
    response: Response,
  ): Promise<unknown> {
    const text = await response.text()

    if (text.length === 0) {
      return null
    }

    try {
      return JSON.parse(text)
    } catch {
      return text
    }
  }

  private encode(value: string): string {
    return encodeURIComponent(value)
  }

  private url(path: string): string {
    if (path.startsWith("http://")) {
      return path
    }

    if (path.startsWith("https://")) {
      return path
    }

    return `${this.baseUrl}${path}`
  }
}