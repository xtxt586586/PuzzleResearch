export type JsonObject = Record<string, unknown>

export type StreamEvent = {
  type: string
  data: unknown
}

export type SseEvent = {
  id?: number
  event?: string
  data?: unknown
  value?: unknown
}

export type Message = {
  role: string
  content: string
}

export type RagEmbeddingStatus = {
  provider: string
  available: boolean
  dimensions: number | null
  reason: string
}

export type AgentHealthResponse = {
  agents_count: number
  agent_ids: string[]
  tool_registry_total_tools: number
  enhanced_tool_discovery_enabled: boolean
  enhanced_tool_discovery_recursive: boolean
  rag_enabled: boolean
  rag_embedding: RagEmbeddingStatus
  search_strategy: string
  search_engines: string[]
  search_providers_available: string[]
}

export type KnowledgeDocument = {
  source: string
  filename: string
  size: number
  chunks: number
  uploaded_at: string
}

export type DocumentUploadResponse = {
  success: boolean
  filename: string
  chunks: number
  message: string
}

export type DocumentListResponse = {
  total_chunks: number
  documents: KnowledgeDocument[]
}

export type DocumentDeleteResponse = {
  success: boolean
  source: string
}

export type DocumentSearchResult = {
  source: string
  chunk_index: number
  text: string
  score: number
}

export type DocumentSearchResponse = {
  query: string
  results: DocumentSearchResult[]
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

export type WorkspaceFilesResponse = {
  thread_id: string
  count: number
  files: string[]
}

export type WorkspaceUploadResponse = {
  thread_id: string
  path: string
  bytes: number
}

export type WorkspaceDeleteResponse = {
  thread_id: string
  path: string
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

export type McpServerDiagnostic = {
  server_id: string
  transport: string
  status: string
  loaded_tools: number
  message?: string | null
}

export type McpDiagnoseResponse = {
  enabled: boolean
  status: string
  ready: boolean
  server_count: number
  connected_count: number
  servers: McpServerDiagnostic[]
}

export type SandboxBrowserDiagnoseResponse = {
  ready: boolean
  missing: string[]
  sandbox_mode: string
  allow_internet: boolean
  configured: Record<string, boolean>
  deep: JsonObject | null
}

export type EvidenceSource = {
  title: string
  url: string
  rawUrl?: string | null
  domain?: string | null
  provider?: string | null
  publishedDate?: string | null
}

export type EvidenceClaimEvidence = {
  url: string
  snippet_hash?: string | null
  quote?: string | null
  heading_path?: string[] | null
}

export type EvidenceClaim = {
  claim: string
  status: string
  evidence_urls: string[]
  evidence_passages: EvidenceClaimEvidence[]
  score: number
  notes: string
}

export type FetchPageItem = {
  url: string
  raw_url: string
  method: string
  text?: string | null
  title?: string | null
  published_date?: string | null
  retrieved_date?: string | null
  retrieved_at?: string | null
  markdown?: string | null
  http_status?: number | null
  error?: string | null
  attempts: number
}

export type EvidencePassageItem = {
  url: string
  text: string
  start_char: number
  end_char: number
  heading?: string | null
  heading_path?: string[] | null
  page_title?: string | null
  retrieved_at?: string | null
  method?: string | null
  quote?: string | null
  snippet_hash?: string | null
}

export type EvidenceResponse = {
  sources: EvidenceSource[]
  claims: EvidenceClaim[]
  quality_summary: JsonObject
  fetched_pages: FetchPageItem[]
  passages: EvidencePassageItem[]
}

export type ResearchTreeResponse = {
  thread_id: string
  tree: JsonObject
  quality: JsonObject
}
