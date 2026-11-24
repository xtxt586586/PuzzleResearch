export type ChatMessage = {
  role: "user" | "assistant"
  content: string
}

export type ImagePayload = {
  name: string
  mime: string
  data: string
}

export type RuntimeEvent = {
  type: string
  data: Record<string, unknown>
}

export type ChatMode =
  | "direct"
  | "web"
  | "agent"
  | "deep"
  | "mcp"

export type McpConfig = {
  enabled: boolean
  servers: Record<string, unknown>
  loaded_tools: number
}

export type McpDiagnosticServer = {
  server_id: string
  transport: string
  status: string
  loaded_tools: number
  message: string | null
}

export type McpDiagnostics = {
  enabled: boolean
  status: string
  ready: boolean
  server_count: number
  connected_count: number
  servers: McpDiagnosticServer[]
}

export type PublicModelConfig = {
  default: string
  options: string[]
}

export type SearchProviderStatus = {
  name: string
  configured: boolean
  available: boolean
  healthy: boolean
  reason: string
  calls: number
  successes: number
  failures: number
  empty_results: number
  last_error: string
  total_calls?: number
  success_count?: number
  error_count?: number
  success_rate?: number
  avg_latency_ms?: number
  avg_result_quality?: number
  last_error_time?: string | null
  circuit?: {
    is_open: boolean
    consecutive_failures: number
    opened_for_seconds?: number | null
    resets_in_seconds?: number | null
  }
}

export type SearchCacheStats = {
  entries: number
  hits: number
  misses: number
  hit_rate: number
}

export type RagEmbeddingStatus = {
  provider: string
  available: boolean
  dimensions: number | null
  reason: string
}

export type AgentHealthStatus = {
  rag_embedding: RagEmbeddingStatus
}

const apiBase =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://127.0.0.1:9000"

export function getApiBaseUrl(): string {
  return apiBase
}

export type StoredScreenshot = {
  url: string
  filename: string
  action?: string | null
  page_url?: string | null
  title?: string | null
  mime_type: string
  created_at: string
}

export async function listThreadScreenshots(
  threadId: string,
  limit = 100,
  signal?: AbortSignal,
): Promise<StoredScreenshot[]> {
  const query = new URLSearchParams({
    thread_id: threadId,
    limit: String(limit),
  })
  const response = await fetch(
    `${apiBase}/api/screenshots?${query.toString()}`,
    {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    },
  )

  if (!response.ok) {
    throw new Error(`截图记录加载失败：${response.status}`)
  }

  const payload = (await response.json()) as {
    screenshots?: unknown
  }
  if (!Array.isArray(payload.screenshots)) return []

  return payload.screenshots.filter(
    (item): item is StoredScreenshot =>
      item !== null &&
      typeof item === "object" &&
      typeof (item as StoredScreenshot).url === "string" &&
      typeof (item as StoredScreenshot).filename === "string" &&
      typeof (item as StoredScreenshot).created_at === "string",
  )
}

export async function getAgentHealthStatus(): Promise<AgentHealthStatus> {
  const response = await fetch(`${apiBase}/api/health/agent`, {
    cache: "no-store",
    headers: { Accept: "application/json" },
  })

  if (!response.ok) {
    throw new Error(`Agent 状态读取失败：${response.status}`)
  }

  const payload = (await response.json()) as {
    rag_embedding?: Partial<RagEmbeddingStatus>
  }
  const embedding = payload.rag_embedding ?? {}

  return {
    rag_embedding: {
      provider:
        typeof embedding.provider === "string"
          ? embedding.provider
          : "unknown",
      available: embedding.available === true,
      dimensions:
        typeof embedding.dimensions === "number"
          ? embedding.dimensions
          : null,
      reason:
        typeof embedding.reason === "string"
          ? embedding.reason
          : "状态信息不可用。",
    },
  }
}

export async function getPublicModelConfig(): Promise<PublicModelConfig> {
  const response = await fetch(`${apiBase}/api/config/public`, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`公开模型配置读取失败：${response.status}`)
  }

  const payload = (await response.json()) as {
    models?: {
      default?: unknown
      options?: unknown
    }
    defaults?: {
      primary_model?: unknown
    }
  }
  const configuredOptions = Array.isArray(payload.models?.options)
    ? payload.models.options.filter(
        (option): option is string =>
          typeof option === "string" && option.trim().length > 0,
      ).map((option) => option.trim())
    : []
  const options = [...new Set(configuredOptions)]
  const modelDefault =
    typeof payload.models?.default === "string"
      ? payload.models.default.trim()
      : ""
  const configuredDefault =
    modelDefault ||
    (typeof payload.defaults?.primary_model === "string"
      ? payload.defaults.primary_model.trim()
      : "")

  if (options.length === 0 && configuredDefault) {
    options.push(configuredDefault)
  }

  if (options.length === 0) {
    throw new Error("公开模型配置中没有可用模型")
  }

  return {
    default: options.includes(configuredDefault)
      ? configuredDefault
      : options[0],
    options,
  }
}

export async function getMcpConfig(): Promise<McpConfig> {
  const response = await fetch(`${apiBase}/api/mcp/config`, {
    headers: {
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`MCP 配置读取失败：${response.status}`)
  }

  return (await response.json()) as McpConfig
}

export async function updateMcpConfig(
  enabled: boolean,
  servers: Record<string, unknown>,
): Promise<McpConfig> {
  const response = await fetch(`${apiBase}/api/mcp/config`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      enable: enabled,
      servers,
    }),
  })

  if (!response.ok) {
    throw new Error(`MCP 配置保存失败：${response.status}`)
  }

  return (await response.json()) as McpConfig
}

export async function getMcpDiagnostics(): Promise<McpDiagnostics> {
  const response = await fetch(`${apiBase}/api/mcp/diagnose`, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`MCP 诊断读取失败：${response.status}`)
  }

  return (await response.json()) as McpDiagnostics
}

export async function getSearchProviderStatus(): Promise<SearchProviderStatus[]> {
  const response = await fetch(`${apiBase}/api/search/providers`, {
    cache: "no-store",
    headers: { Accept: "application/json" },
  })

  if (!response.ok) {
    throw new Error(`搜索服务状态读取失败：${response.status}`)
  }

  const payload = (await response.json()) as {
    providers?: SearchProviderStatus[]
  }
  return Array.isArray(payload.providers) ? payload.providers : []
}

export async function resetSearchProviderStatus(): Promise<void> {
  const response = await fetch(`${apiBase}/api/search/providers/reset`, {
    method: "POST",
    headers: { Accept: "application/json" },
  })

  if (!response.ok) {
    throw new Error(`搜索服务状态重置失败：${response.status}`)
  }
}

export async function getSearchCacheStats(): Promise<SearchCacheStats> {
  const response = await fetch(`${apiBase}/api/search/cache/stats`, {
    cache: "no-store",
    headers: { Accept: "application/json" },
  })

  if (!response.ok) {
    throw new Error(`搜索缓存状态读取失败：${response.status}`)
  }

  return (await response.json()) as SearchCacheStats
}

export async function clearSearchCache(): Promise<number> {
  const response = await fetch(`${apiBase}/api/search/cache/clear`, {
    method: "POST",
    headers: { Accept: "application/json" },
  })

  if (!response.ok) {
    throw new Error(`搜索缓存清理失败：${response.status}`)
  }

  const payload = (await response.json()) as { cleared?: number }
  return typeof payload.cleared === "number" ? payload.cleared : 0
}

function parseSseFrame(frame: string): RuntimeEvent | null {
  let eventType = "message"
  let frameId = ""
  const dataLines: string[] = []

  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("id:")) {
      frameId = line.slice(3).trim()
    }

    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim()
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""))
    }
  }

  if (dataLines.length === 0) {
    return null
  }

  const raw = dataLines.join("\n")

  try {
    const parsed = JSON.parse(raw) as unknown

    if (
      parsed !== null &&
      typeof parsed === "object" &&
      "type" in parsed &&
      "data" in parsed
    ) {
      const envelope = parsed as {
        type: unknown
        data: unknown
        event_id?: unknown
        seq?: unknown
      }
      const envelopeData: Record<string, unknown> =
        envelope.data !== null &&
        typeof envelope.data === "object" &&
        !Array.isArray(envelope.data)
          ? {
              ...(envelope.data as Record<string, unknown>),
            }
          : { value: envelope.data }

      if (
        envelope.seq !== undefined &&
        envelopeData.seq === undefined
      ) {
        envelopeData.seq = envelope.seq
      } else if (
        frameId &&
        envelopeData.seq === undefined
      ) {
        const numericFrameId = Number(frameId)
        envelopeData.seq = Number.isFinite(
          numericFrameId,
        )
          ? numericFrameId
          : frameId
      }

      if (
        envelope.event_id !== undefined &&
        envelopeData.event_id === undefined
      ) {
        envelopeData.event_id = envelope.event_id
      }

      return {
        type: String(envelope.type),
        data: envelopeData,
      }
    }

    return {
      type: eventType,
      data:
        parsed !== null && typeof parsed === "object"
          ? (parsed as Record<string, unknown>)
          : { value: parsed },
    }
  } catch {
    return {
      type: eventType,
      data: {
        content: raw,
      },
    }
  }
}


export async function cancelChat(
  threadId: string,
): Promise<void> {
  const response = await fetch(
    `${apiBase}/api/chat/cancel/${encodeURIComponent(
      threadId,
    )}`,
    {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
    },
  )

  if (!response.ok && response.status !== 404) {
    throw new Error(
      `Cancel request failed: ${response.status}`,
    )
  }
}

export async function streamThreadEvents(
  threadId: string,
  signal: AbortSignal,
  onEvent: (event: RuntimeEvent) => void,
  lastEventId?: number,
): Promise<void> {
  const query =
    lastEventId !== undefined
      ? `?last_event_id=${encodeURIComponent(
          String(lastEventId),
        )}`
      : ""
  const response = await fetch(
    `${apiBase}/api/events/${encodeURIComponent(threadId)}${query}`,
    {
      headers: {
        Accept: "text/event-stream",
      },
      signal,
    },
  )

  if (!response.ok) {
    throw new Error(`事件流连接失败：${response.status}`)
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

    const frames = buffer.split(/\r?\n\r?\n/)
    buffer = frames.pop() ?? ""

    for (const frame of frames) {
      const event = parseSseFrame(frame)

      if (event) {
        onEvent(event)
      }
    }
  }

  buffer += decoder.decode()

  if (buffer.trim()) {
    const event = parseSseFrame(buffer)

    if (event) {
      onEvent(event)
    }
  }
}

export function normalizeChatMode(
  mode: ChatMode | boolean | undefined,
): ChatMode {
  if (typeof mode === "boolean") {
    return mode ? "deep" : "direct"
  }

  return mode ?? "direct"
}

export function buildChatRequest(
  messages: ChatMessage[],
  mode: ChatMode | boolean | undefined,
  threadId?: string,
  model?: string,
  images?: ImagePayload[],
  mcpServerId?: string,
): RequestInit {
  const selectedMode = normalizeChatMode(mode)

  return {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages,
      stream: true,
      model: model || undefined,
      search_mode: {
        mode: selectedMode,
        useWebSearch: selectedMode === "web",
        useAgent:
          selectedMode === "agent" || selectedMode === "mcp",
        useDeepSearch: selectedMode === "deep",
      },
      ...(threadId
        ? {
            thread_id: threadId,
          }
        : {}),
      ...(images?.length ? { images } : {}),
      ...(selectedMode === "mcp" && mcpServerId
        ? { mcp_server_id: mcpServerId }
        : {}),
    }),
  }
}

export async function* streamChat(
  messages: ChatMessage[],
  mode: ChatMode | boolean,
  signal: AbortSignal,
  threadId?: string,
  onThreadId?: (threadId: string) => void,
  model?: string,
  images?: ImagePayload[],
  mcpServerId?: string,
): AsyncGenerator<RuntimeEvent> {
  const response = await fetch(
    `${apiBase}/api/chat/sse`,
    {
      ...buildChatRequest(
        messages,
        mode,
        threadId,
        model,
        images,
        mcpServerId,
      ),
      signal,
    },
  )

  const responseThreadId =
  response.headers.get("x-thread-id")

  if (responseThreadId) {
    onThreadId?.(responseThreadId)
  }


  if (!response.ok) {
    throw new Error(
      `Chat request failed: ${response.status}`,
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

    const frames = buffer.split(/\r?\n\r?\n/)
    buffer = frames.pop() ?? ""

    for (const frame of frames) {
      const event = parseSseFrame(frame)

      if (event) {
        yield event
      }
    }
  }

  buffer += decoder.decode()

  if (buffer.trim()) {
    const event = parseSseFrame(buffer)

    if (event) {
      yield event
    }
  }
}
