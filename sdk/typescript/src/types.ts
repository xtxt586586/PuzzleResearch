export type JsonObject = Record<string, unknown>

export type StreamEvent = {
  type: string
  data: unknown
}

export type SseEvent = {
  id?: number
  event?: string
  data?: unknown
}

export type Message = {
  role: string
  content: string
}

export type ChatRequest = {
  messages: Message[]
  stream?: boolean
  model?: string
  search_mode?: JsonObject | string
  agent_id?: string
  user_id?: string
  images?: JsonObject[]
}

export type ResearchRequest = {
  query: string
  model?: string
  search_mode?: JsonObject | string
  agent_id?: string
  user_id?: string
  images?: JsonObject[]
}

export type SessionStateResponse = {
  thread_id: string
  status: string
  state: JsonObject
  interrupts: unknown[]
  checkpoint_id?: string | null
  checkpoint_ts?: string
  parent_checkpoint_id?: string | null
  deepsearch_artifacts?: JsonObject
}

export type SessionSummaryResponse = {
  thread_id: string
  status: string
  topic: string
  created_at: string
  updated_at: string
  route: string
  has_report: boolean
  revision_count: number
  message_count: number
  agent_id: string
  user_id: string
  owner_id?: string
}

export type SessionsListResponse = {
  count: number
  sessions: SessionSummaryResponse[]
}

export type McpConfigResponse = {
  enabled: boolean
  servers: Record<string, JsonObject>
  loaded_tools: number
}

export type SandboxBrowserDiagnoseResponse = {
  ready: boolean
  missing: string[]
  sandbox_mode: string
  allow_internet: boolean
  configured: Record<string, boolean>
  deep: JsonObject | null
}