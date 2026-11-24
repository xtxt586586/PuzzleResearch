import {
  normalizeSseEvent,
  readDataStreamEvents,
  readSseEvents,
} from "./sse.js"

import type {
  AgentHealthResponse,
  DocumentDeleteResponse,
  DocumentListResponse,
  DocumentSearchResponse,
  DocumentUploadResponse,
  EvidenceResponse,
  ResearchTreeResponse,
  SseEvent,
  StreamEvent,
  WorkspaceDeleteResponse,
  WorkspaceFilesResponse,
  WorkspaceUploadResponse,
} from "./types.js"

export type JsonObject = Record<string, unknown>


export interface PuzzleResearchClientOptions {
  baseUrl?: string
  headers?: HeadersInit
  fetch?: typeof fetch
}

export interface RequestOptions {
  signal?: AbortSignal
}

export interface StreamOptions extends RequestOptions {
  lastEventId?: number | string
}

export interface SessionListOptions extends RequestOptions {
  limit?: number
  status?: string
}

export interface ExportOptions extends RequestOptions {
  format?: string
  title?: string
  template?: string
}

export interface TriggerListOptions extends RequestOptions {
  triggerType?: string
  status?: string
}

export interface TriggerExecutionOptions extends RequestOptions {
  limit?: number
}

export interface BrowserStreamOptions {
  protocols?: string | string[]
  webSocket?: new (
    url: string,
    protocols?: string | string[],
  ) => WebSocket
}

export interface CancelChatOptions extends RequestOptions {
  reason?: string
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
      "http://127.0.0.1:8001"
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

  async requestRaw(
    path: string,
    init: RequestInit = {},
  ): Promise<Response> {
    const headers = new Headers(this.defaultHeaders)
    new Headers(init.headers).forEach((value, key) => {
      headers.set(key, value)
    })

    const response = await this.fetcher(
      this.url(path),
      {
        ...init,
        headers,
      },
    )

    this.captureThreadId(response)

    if (!response.ok) {
      throw new PuzzleResearchApiError(
        response.status,
        await this.readPayload(response),
      )
    }

    return response
  }

  async chat<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/chat", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  chatSse(
  payload: JsonObject,
  options: RequestOptions = {},
): AsyncGenerator<StreamEvent> {
  return this.stream("/api/chat/sse", {
    method: "POST",
    body: payload,
    signal: options.signal,
  })
}

  research(
  input: string | JsonObject,
  options: RequestOptions = {},
): AsyncGenerator<StreamEvent> {
  const query =
    typeof input === "string"
      ? input
      : String(input.query ?? "")

  const params = new URLSearchParams({
    query,
  })

  return this.streamDataProtocol(
    `/api/research?${params.toString()}`,
    {
      method: "POST",
      signal: options.signal,
    },
  )
}

  researchSse(
  payload: JsonObject,
  options: RequestOptions = {},
): AsyncGenerator<StreamEvent> {
  return this.stream("/api/research/sse", {
    method: "POST",
    body: payload,
    signal: options.signal,
  })
}

  events(
  threadId: string,
  options: StreamOptions = {},
): AsyncGenerator<StreamEvent> {
  const headers: Record<string, string> = {}

  if (options.lastEventId !== undefined) {
    headers["Last-Event-ID"] = String(
      options.lastEventId,
    )
  }

  return this.stream(
    `/api/events/${this.encode(threadId)}`,
    {
      method: "GET",
      signal: options.signal,
      headers,
    },
  )
}

  researchStream(
    query: string,
    options: RequestOptions = {},
  ): AsyncGenerator<StreamEvent> {
    return this.research(query, options)
  }

  streamResearchEvents(
    threadId: string,
    options: StreamOptions = {},
  ): AsyncGenerator<StreamEvent> {
    return this.events(threadId, options)
  }

  async cancelChat<T = JsonObject>(
    threadId: string,
    options: CancelChatOptions = {},
  ): Promise<T> {
    const init: JsonRequestInit = {
      method: "POST",
      signal: options.signal,
    }

    if (options.reason !== undefined) {
      init.body = {
        reason: options.reason,
      }
    }

    return this.request<T>(
      `/api/chat/cancel/${this.encode(threadId)}`,
      init,
    )
  }

  async cancelAllChats<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/chat/cancel-all", {
      method: "POST",
      signal: options.signal,
    })
  }

  async getAgentHealth<T = AgentHealthResponse>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/health/agent", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getToolRegistry<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/tools/registry", {
      method: "GET",
      signal: options.signal,
    })
  }

  async refreshToolRegistry<T = JsonObject>(
    reset = false,
    options: RequestOptions = {},
  ): Promise<T> {
    const path = reset
      ? "/api/tools/registry/refresh?reset=true"
      : "/api/tools/registry/refresh"

    return this.request<T>(path, {
      method: "POST",
      signal: options.signal,
    })
  }

  async getSearchProviders<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/search/providers", {
      method: "GET",
      signal: options.signal,
    })
  }

  async resetSearchProviders<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/search/providers/reset", {
      method: "POST",
      signal: options.signal,
    })
  }

  async getSearchCacheStats<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/search/cache/stats", {
      method: "GET",
      signal: options.signal,
    })
  }

  async clearSearchCache<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/search/cache/clear", {
      method: "POST",
      signal: options.signal,
    })
  }

  async getPublicConfig<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/config/public", {
      method: "GET",
      signal: options.signal,
    })
  }

  async listAgents<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/agents", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getAgent<T = JsonObject>(
    agentId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/agents/${this.encode(agentId)}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async createAgent<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/agents", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async updateAgent<T = JsonObject>(
    agentId: string,
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/agents/${this.encode(agentId)}`,
      {
        method: "PUT",
        body: payload,
        signal: options.signal,
      },
    )
  }

  async deleteAgent<T = JsonObject>(
    agentId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/agents/${this.encode(agentId)}`,
      {
        method: "DELETE",
        signal: options.signal,
      },
    )
  }

  async getActiveTasks<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/tasks/active", {
      method: "GET",
      signal: options.signal,
    })
  }

  async deleteSession<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}`,
      {
        method: "DELETE",
        signal: options.signal,
      },
    )
  }

  async getInterruptStatus<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/interrupt/${this.encode(threadId)}/status`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async listRuns<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/runs", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getRunMetrics<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/runs/${this.encode(threadId)}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async getTrace<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/traces/${this.encode(threadId)}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async getTraceSummary<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/traces/${this.encode(threadId)}/summary`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async getAllTraces<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/traces/${this.encode(threadId)}/all`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

 async listSessions<T = unknown>(
  options: SessionListOptions = {},
): Promise<T> {
  const params = new URLSearchParams()

  if (options.limit !== undefined) {
    params.set("limit", String(options.limit))
  }

  if (options.status) {
    params.set("status", options.status)
  }

  const query = params.toString()
  const path = query
    ? `/api/sessions?${query}`
    : "/api/sessions"

  return this.request<T>(path, {
    method: "GET",
    signal: options.signal,
  })
}

async getSessions<T = unknown>(
  options: SessionListOptions = {},
): Promise<T> {
  return this.listSessions<T>(options)
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

  async listWorkspaceFiles<T = WorkspaceFilesResponse>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/workspace/files`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async uploadWorkspaceFile<T = WorkspaceUploadResponse>(
    threadId: string,
    file: Blob,
    filename = "upload.bin",
    options: RequestOptions = {},
  ): Promise<T> {
    const formData = new FormData()
    formData.append("file", file, filename)

    return this.requestFormData<T>(
      `/api/sessions/${this.encode(threadId)}/workspace/files/upload`,
      formData,
      options,
    )
  }

  async downloadWorkspaceFile(
    threadId: string,
    filePath: string,
    options: RequestOptions = {},
  ): Promise<Blob> {
    const encodedPath = filePath
      .split("/")
      .map((part) => this.encode(part))
      .join("/")

    return this.requestBlob(
      `/api/sessions/${this.encode(threadId)}/workspace/files/${encodedPath}`,
      options,
    )
  }

  async deleteWorkspaceFile<T = WorkspaceDeleteResponse>(
    threadId: string,
    filePath: string,
    options: RequestOptions = {},
  ): Promise<T> {
    const encodedPath = filePath
      .split("/")
      .map((part) => this.encode(part))
      .join("/")

    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/workspace/files/${encodedPath}`,
      {
        method: "DELETE",
        signal: options.signal,
      },
    )
  }

  async getSessionEvidence<T = EvidenceResponse>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/evidence`,
    )
  }

  async getEvidence<T = EvidenceResponse>(
    threadId: string,
  ): Promise<T> {
    return this.getSessionEvidence<T>(threadId)
  }

  async getResearchTree<T = ResearchTreeResponse>(
    threadId: string,
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/research-tree`,
    )
  }

  async resumeSession<T = JsonObject>(
    threadId: string,
    payload: JsonObject = {},
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/resume`,
      {
        method: "POST",
        body: payload,
        signal: options.signal,
      },
    )
  }

  async resumeInterrupt<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/interrupt/resume", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async resumeThreadInterrupt<T = JsonObject>(
    threadId: string,
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/interrupt/${this.encode(threadId)}/resume`,
      {
        method: "POST",
        body: payload,
        signal: options.signal,
      },
    )
  }

  async createShare<T = JsonObject>(
    threadId: string,
    payload: JsonObject = {},
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/share`,
      {
        method: "POST",
        body: payload,
        signal: options.signal,
      },
    )
  }

  async getShare<T = JsonObject>(
    shareId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/share/${this.encode(shareId)}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async deleteShare<T = JsonObject>(
    shareId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/share/${this.encode(shareId)}`,
      {
        method: "DELETE",
        signal: options.signal,
      },
    )
  }

  async addSessionComment<T = JsonObject>(
    threadId: string,
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/comments`,
      {
        method: "POST",
        body: payload,
        signal: options.signal,
      },
    )
  }

  async getSessionComments<T = JsonObject>(
    threadId: string,
    messageId?: string,
    options: RequestOptions = {},
  ): Promise<T> {
    const params = new URLSearchParams()

    if (messageId) {
      params.set("message_id", messageId)
    }

    const query = params.toString()
    const path = query
      ? `/api/sessions/${this.encode(threadId)}/comments?${query}`
      : `/api/sessions/${this.encode(threadId)}/comments`

    return this.request<T>(path, {
      method: "GET",
      signal: options.signal,
    })
  }

  async getSessionVersions<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/versions`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async createSessionVersion<T = JsonObject>(
    threadId: string,
    label?: string,
    options: RequestOptions = {},
  ): Promise<T> {
    const params = new URLSearchParams()

    if (label) {
      params.set("label", label)
    }

    const query = params.toString()
    const path = query
      ? `/api/sessions/${this.encode(threadId)}/versions?${query}`
      : `/api/sessions/${this.encode(threadId)}/versions`

    return this.request<T>(path, {
      method: "POST",
      signal: options.signal,
    })
  }

  async restoreSessionVersion<T = JsonObject>(
    threadId: string,
    versionId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/sessions/${this.encode(threadId)}/restore/${this.encode(versionId)}`,
      {
        method: "POST",
        signal: options.signal,
      },
    )
  }

  async createScheduledTrigger<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/triggers/scheduled", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async createWebhookTrigger<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/triggers/webhook", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async createEventTrigger<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/triggers/event", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async listTriggers<T = JsonObject>(
    options: TriggerListOptions = {},
  ): Promise<T> {
    const params = new URLSearchParams()

    if (options.triggerType) {
      params.set("trigger_type", options.triggerType)
    }

    if (options.status) {
      params.set("status", options.status)
    }

    const query = params.toString()
    const path = query
      ? `/api/triggers?${query}`
      : "/api/triggers"

    return this.request<T>(path, {
      method: "GET",
      signal: options.signal,
    })
  }

  async getTrigger<T = JsonObject>(
    triggerId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/triggers/${this.encode(triggerId)}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async deleteTrigger<T = JsonObject>(
    triggerId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/triggers/${this.encode(triggerId)}`,
      {
        method: "DELETE",
        signal: options.signal,
      },
    )
  }

  async pauseTrigger<T = JsonObject>(
    triggerId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/triggers/${this.encode(triggerId)}/pause`,
      {
        method: "POST",
        signal: options.signal,
      },
    )
  }

  async resumeTrigger<T = JsonObject>(
    triggerId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/triggers/${this.encode(triggerId)}/resume`,
      {
        method: "POST",
        signal: options.signal,
      },
    )
  }

  async getTriggerExecutions<T = JsonObject>(
    triggerId: string,
    options: TriggerExecutionOptions = {},
  ): Promise<T> {
    const params = new URLSearchParams()

    if (options.limit !== undefined) {
      params.set("limit", String(options.limit))
    }

    const query = params.toString()
    const path = query
      ? `/api/triggers/${this.encode(triggerId)}/executions?${query}`
      : `/api/triggers/${this.encode(triggerId)}/executions`

    return this.request<T>(path, {
      method: "GET",
      signal: options.signal,
    })
  }

  async invokeWebhook<T = JsonObject>(
    triggerId: string,
    payload: JsonObject = {},
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/webhook/${this.encode(triggerId)}`,
      {
        method: "POST",
        body: payload,
        signal: options.signal,
      },
    )
  }

  async supportChat<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/support/chat", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async getMemoryStatus<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/memory/status", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getHealth<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/health", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getMetrics(
    options: RequestOptions = {},
  ): Promise<string> {
    return this.request<string>("/metrics", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getRoot<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getMcpConfig<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/mcp/config", {
      method: "GET",
      signal: options.signal,
    })
  }

  async updateMcpConfig<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/mcp/config", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async diagnoseMcp<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/mcp/diagnose", {
      method: "GET",
      signal: options.signal,
    })
  }

  async diagnoseBrowser<T = JsonObject>(
    deep = false,
    options: RequestOptions = {},
  ): Promise<T> {
    const params = new URLSearchParams({
      deep: String(deep),
    })

    return this.request<T>(
      `/api/sandbox/browser/diagnose?${params.toString()}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  openBrowserStream(
    threadId: string,
    options: BrowserStreamOptions = {},
  ): WebSocket {
    const socketConstructor =
      options.webSocket ?? globalThis.WebSocket

    if (!socketConstructor) {
      throw new Error(
        "WebSocket is not available in the current runtime",
      )
    }

    const base = new URL(this.baseUrl)

    base.protocol =
      base.protocol === "https:"
        ? "wss:"
        : "ws:"

    const baseUrl = base.toString().replace(/\/+$/, "")

    return new socketConstructor(
      `${baseUrl}/api/browser/${this.encode(threadId)}/stream`,
      options.protocols,
    )
  }

  async getBrowserSessionInfo<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/browser/${this.encode(threadId)}/info`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async triggerBrowserScreenshot<T = JsonObject>(
    threadId: string,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>(
      `/api/browser/${this.encode(threadId)}/screenshot`,
      {
        method: "POST",
        signal: options.signal,
      },
    )
  }

  async getScreenshot(
    filename: string,
    options: RequestOptions = {},
  ): Promise<Blob> {
    return this.requestBlob(
      `/api/screenshots/${this.encode(filename)}`,
      options,
    )
  }

  async listScreenshots<T = JsonObject>(
    threadId?: string,
    limit = 50,
    options: RequestOptions = {},
  ): Promise<T> {
    const params = new URLSearchParams()

    if (threadId) {
      params.set("thread_id", threadId)
    }

    params.set("limit", String(limit))

    return this.request<T>(
      `/api/screenshots?${params.toString()}`,
      {
        method: "GET",
        signal: options.signal,
      },
    )
  }

  async cleanupScreenshots<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/screenshots/cleanup", {
      method: "POST",
      signal: options.signal,
    })
  }

  async uploadDocument<T = DocumentUploadResponse>(
  file: Blob,
  filename = "upload.bin",
  options: RequestOptions = {},
): Promise<T> {
  const formData = new FormData()

  formData.append("file", file, filename)

  return this.requestFormData<T>(
    "/api/documents/upload",
    formData,
    options,
  )
}

async listDocuments<T = DocumentListResponse>(
  limit = 100,
  options: RequestOptions = {},
): Promise<T> {
  const params = new URLSearchParams({
    limit: String(limit),
  })

  return this.request<T>(
    `/api/documents/list?${params.toString()}`,
    {
      method: "GET",
      signal: options.signal,
    },
  )
}

async deleteDocument<T = DocumentDeleteResponse>(
  source: string,
  options: RequestOptions = {},
): Promise<T> {
  return this.request<T>(
    `/api/documents/${this.encode(source)}`,
    {
      method: "DELETE",
      signal: options.signal,
    },
  )
}

async searchDocuments<T = DocumentSearchResponse>(
  query: string,
  nResults = 5,
  options: RequestOptions = {},
): Promise<T> {
  const params = new URLSearchParams({
    query,
    n_results: String(nResults),
  })

  return this.request<T>(
    `/api/documents/search?${params.toString()}`,
    {
      method: "POST",
      signal: options.signal,
    },
  )
}

  async recognizeSpeech<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/asr/recognize", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async recognizeSpeechUpload<T = JsonObject>(
    file: Blob,
    filename = "audio.wav",
    sampleRate = 16000,
    options: RequestOptions = {},
  ): Promise<T> {
    const formData = new FormData()

    formData.append("file", file, filename)

    return this.requestFormData<T>(
      `/api/asr/upload?sample_rate=${encodeURIComponent(String(sampleRate))}`,
      formData,
      options,
    )
  }

  async getAsrStatus<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/asr/status", {
      method: "GET",
      signal: options.signal,
    })
  }

  async synthesizeSpeech<T = JsonObject>(
    payload: JsonObject,
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/tts/synthesize", {
      method: "POST",
      body: payload,
      signal: options.signal,
    })
  }

  async getTtsVoices<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/tts/voices", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getTtsStatus<T = JsonObject>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/tts/status", {
      method: "GET",
      signal: options.signal,
    })
  }

  async getExportTemplates<T = unknown>(
    options: RequestOptions = {},
  ): Promise<T> {
    return this.request<T>("/api/export/templates", {
      method: "GET",
      signal: options.signal,
    })
  }

  async exportSession(
  threadId: string,
  options: ExportOptions = {},
): Promise<Blob> {
  const params = new URLSearchParams()

  if (options.format) {
    params.set("format", options.format)
  }

  if (options.title) {
    params.set("title", options.title)
  }

  if (options.template) {
    params.set("template", options.template)
  }

  const query = params.toString()
  const path = query
    ? `/api/export/${this.encode(threadId)}?${query}`
    : `/api/export/${this.encode(threadId)}`

    return this.requestBlob(path, {
      signal: options.signal,
    })
  }

  async exportReport(
    threadId: string,
    options: ExportOptions = {},
  ): Promise<{
    contentType: string | null
    bytes: ArrayBuffer
  }> {
    const params = new URLSearchParams()

    if (options.format) {
      params.set("format", options.format)
    }

    if (options.title) {
      params.set("title", options.title)
    }

    if (options.template) {
      params.set("template", options.template)
    }

    const query = params.toString()
    const path = query
      ? `/api/export/${this.encode(threadId)}?${query}`
      : `/api/export/${this.encode(threadId)}`

    const response = await this.requestRaw(path, {
      method: "GET",
      signal: options.signal,
    })

    return {
      contentType: response.headers.get("content-type"),
      bytes: await response.arrayBuffer(),
    }
  }

  async downloadExport(
    threadId: string,
    options: ExportOptions = {},
  ): Promise<Blob> {
    return this.exportSession(threadId, options)
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

  private async requestFormData<T>(
  path: string,
  formData: FormData,
  options: RequestOptions = {},
): Promise<T> {
  const headers = new Headers(this.defaultHeaders)

  headers.delete("Content-Type")
  headers.set("Accept", "application/json")

  const response = await this.fetcher(
    this.url(path),
    {
      method: "POST",
      headers,
      body: formData,
      signal: options.signal,
    },
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

  private async requestBlob(
  path: string,
  options: RequestOptions = {},
): Promise<Blob> {
  const response = await this.fetcher(
    this.url(path),
    this.buildRequest({
      method: "GET",
      signal: options.signal,
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
): AsyncGenerator<StreamEvent> {
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

  for await (const event of readSseEvents(response)) {
    const normalized = normalizeSseEvent(event)

    if (normalized !== null) {
      yield normalized
    }
  }
}

  private async *streamDataProtocol(
  path: string,
  init: JsonRequestInit,
): AsyncGenerator<StreamEvent> {
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

  for await (
    const event of readDataStreamEvents(response)
  ) {
    yield event
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
