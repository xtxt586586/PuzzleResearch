import { getApiBaseUrl } from "./api"

export type KnowledgeDocument = {
  source: string
  filename: string
  size: number
  chunks: number
  uploaded_at: string
}

export type KnowledgeSearchResult = {
  source: string
  chunk_index: number
  text: string
  score: number
}

async function requireOk(response: Response): Promise<void> {
  if (response.ok) {
    return
  }

  let detail = "文档服务请求失败。"
  try {
    const payload = (await response.json()) as { detail?: unknown }
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      detail = payload.detail.trim()
    }
  } catch {
    detail = `文档服务请求失败（HTTP ${response.status}）。`
  }

  throw new Error(detail)
}

export async function listKnowledgeDocuments(): Promise<KnowledgeDocument[]> {
  const response = await fetch(`${getApiBaseUrl()}/api/documents/list`, {
    cache: "no-store",
    headers: { Accept: "application/json" },
  })
  await requireOk(response)

  const payload = (await response.json()) as {
    documents?: KnowledgeDocument[]
  }
  return Array.isArray(payload.documents) ? payload.documents : []
}

export async function uploadKnowledgeDocument(
  file: File,
): Promise<{ filename: string; chunks: number; message: string }> {
  const form = new FormData()
  form.append("file", file)

  const response = await fetch(`${getApiBaseUrl()}/api/documents/upload`, {
    method: "POST",
    headers: { Accept: "application/json" },
    body: form,
  })
  await requireOk(response)
  return (await response.json()) as {
    filename: string
    chunks: number
    message: string
  }
}

export async function deleteKnowledgeDocument(
  source: string,
): Promise<void> {
  const response = await fetch(
    `${getApiBaseUrl()}/api/documents/${encodeURIComponent(source)}`,
    {
      method: "DELETE",
      headers: { Accept: "application/json" },
    },
  )
  await requireOk(response)
}

export async function searchKnowledgeDocuments(
  query: string,
  nResults = 5,
): Promise<KnowledgeSearchResult[]> {
  const params = new URLSearchParams({
    query,
    n_results: String(nResults),
  })
  const response = await fetch(
    `${getApiBaseUrl()}/api/documents/search?${params.toString()}`,
    {
      method: "POST",
      headers: { Accept: "application/json" },
    },
  )
  await requireOk(response)

  const payload = (await response.json()) as {
    results?: KnowledgeSearchResult[]
  }
  return Array.isArray(payload.results) ? payload.results : []
}
