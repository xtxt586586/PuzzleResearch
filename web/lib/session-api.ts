export type SessionSummary = {
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
  owner_id: string
}

export type SessionStateResponse = {
  thread_id: string
  status: string
  state: Record<string, unknown>
  interrupts: unknown[]
  checkpoint_id: string | null
  checkpoint_ts: string
  parent_checkpoint_id: string | null
  deepsearch_artifacts: Record<string, unknown>
}

export type ResumeInterruptResponse = {
  status?: string
  thread_id?: string
  result?: unknown
  interrupts?: unknown[]
  content?: string
  execution_status?: string
}

export type EditedToolAction = {
  type: "edit"
  edited_action: {
    name: string
    args: Record<string, unknown>
  }
}

export type EditedResearchPlan = {
  type: "edit_plan"
  queries: string[]
}

export type InterruptResumeDecision =
  | boolean
  | EditedToolAction
  | EditedResearchPlan
  | { content: string }

const apiBase =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://127.0.0.1:9000"

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    throw new Error(
      `Request failed: ${response.status}`,
    )
  }

  return (await response.json()) as T
}

export async function listSessions(): Promise<
  SessionSummary[]
> {
  const payload = await requestJson<{
    count: number
    sessions: SessionSummary[]
  }>("/api/sessions")

  return payload.sessions
}

export async function getSessionState(
  threadId: string,
): Promise<SessionStateResponse> {
  return requestJson<SessionStateResponse>(
    `/api/sessions/${encodeURIComponent(threadId)}/state`,
  )
}

export async function deleteSession(
  threadId: string,
): Promise<void> {
  await requestJson(
    `/api/sessions/${encodeURIComponent(threadId)}`,
    {
      method: "DELETE",
    },
  )
}

export async function resumeInterrupt(
  threadId: string,
  decision: InterruptResumeDecision,
): Promise<ResumeInterruptResponse> {
  return requestJson<ResumeInterruptResponse>(
    `/api/interrupt/${encodeURIComponent(threadId)}/resume`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(
        typeof decision === "boolean"
          ? { approved: decision }
          : { resume_value: decision },
      ),
    },
  )
}
