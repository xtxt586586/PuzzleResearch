export type TriggerType = "scheduled" | "webhook" | "event"
export type TriggerStatus = "active" | "paused" | "disabled" | "error"

export type Trigger = {
  id: string
  name: string
  description: string
  trigger_type: TriggerType
  status: TriggerStatus
  agent_id: string
  task: string
  task_params: Record<string, unknown>
  created_at: string
  updated_at: string
  last_executed_at: string | null
  execution_count: number
  failure_count: number
  schedule?: string
  timezone?: string
  run_immediately?: boolean
  next_run_at?: string | null
  endpoint_path?: string
  http_methods?: string[]
  require_auth?: boolean
  event_type?: string
  event_source?: string | null
  event_filters?: Record<string, unknown>
}

export type TriggerExecution = {
  id: string
  trigger_id: string
  trigger_name: string
  started_at: string
  completed_at: string | null
  duration_ms: number | null
  status: "running" | "success" | "failed" | string
  error_message: string | null
  thread_id: string | null
  task: string
  output_text: string | null
}

export type TriggerCreationResult = {
  success: boolean
  trigger_id: string
  trigger: Trigger
  auth_token?: string
}

const apiBase =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://127.0.0.1:9000"

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers ?? {}),
      },
    })
  } catch (cause) {
    if (cause instanceof Error && cause.name === "AbortError") {
      throw cause
    }
    throw new Error("无法连接后端服务，请确认后端已启动并检查 API 地址。")
  }
  const payload = await response.json().catch(() => ({})) as {
    detail?: unknown
    message?: unknown
  }

  if (!response.ok) {
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload.message === "string"
          ? payload.message
          : `请求失败（${response.status}）`
    throw new Error(detail)
  }

  return payload as T
}

export async function listTriggers(filters: {
  triggerType?: TriggerType | "all"
  status?: TriggerStatus | "all"
} = {}): Promise<Trigger[]> {
  const query = new URLSearchParams()
  if (filters.triggerType && filters.triggerType !== "all") {
    query.set("trigger_type", filters.triggerType)
  }
  if (filters.status && filters.status !== "all") {
    query.set("status", filters.status)
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : ""
  const payload = await requestJson<{ triggers?: Trigger[] }>(
    `/api/triggers${suffix}`,
  )
  return Array.isArray(payload.triggers) ? payload.triggers : []
}

export function createScheduledTrigger(
  input: {
    name: string
    task: string
    schedule: string
    timezone: string
    run_immediately: boolean
    agent_id: string
    task_params: Record<string, unknown>
  },
): Promise<TriggerCreationResult> {
  return requestJson("/api/triggers/scheduled", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  })
}

export function createWebhookTrigger(
  input: {
    name: string
    task: string
    require_auth: boolean
    http_methods: string[]
    agent_id: string
    task_params: Record<string, unknown>
  },
): Promise<TriggerCreationResult> {
  return requestJson("/api/triggers/webhook", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  })
}

export function createEventTrigger(
  input: {
    name: string
    task: string
    event_type: string
    event_source?: string
    event_filters: Record<string, unknown>
    agent_id: string
    task_params: Record<string, unknown>
  },
): Promise<TriggerCreationResult> {
  return requestJson("/api/triggers/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  })
}

export async function pauseTrigger(triggerId: string): Promise<void> {
  await requestJson(`/api/triggers/${encodeURIComponent(triggerId)}/pause`, {
    method: "POST",
  })
}

export async function resumeTrigger(triggerId: string): Promise<void> {
  await requestJson(`/api/triggers/${encodeURIComponent(triggerId)}/resume`, {
    method: "POST",
  })
}

export async function deleteTrigger(triggerId: string): Promise<void> {
  await requestJson(`/api/triggers/${encodeURIComponent(triggerId)}`, {
    method: "DELETE",
  })
}

export async function getTriggerExecutions(
  triggerId: string,
): Promise<TriggerExecution[]> {
  const payload = await requestJson<{
    executions?: TriggerExecution[]
  }>(`/api/triggers/${encodeURIComponent(triggerId)}/executions`)
  return Array.isArray(payload.executions) ? payload.executions : []
}

export function getWebhookUrl(endpointPath: string): string {
  return `${apiBase}${endpointPath.startsWith("/") ? endpointPath : `/${endpointPath}`}`
}
