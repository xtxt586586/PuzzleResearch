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

export type EvidencePassage = {
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

export type FetchPage = {
  url: string
  raw_url: string
  method: string
  title?: string | null
  retrieved_at?: string | null
  http_status?: number | null
  error?: string | null
}

export type EvidenceResponse = {
  sources: EvidenceSource[]
  claims: EvidenceClaim[]
  quality_summary: Record<string, unknown>
  fetched_pages: FetchPage[]
  passages: EvidencePassage[]
}

const apiBase =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://127.0.0.1:9000"

export async function getSessionEvidence(
  threadId: string,
): Promise<EvidenceResponse> {
  const response = await fetch(
    `${apiBase}/api/sessions/${encodeURIComponent(
      threadId,
    )}/evidence`,
    {
      headers: {
        Accept: "application/json",
      },
    },
  )

  if (!response.ok) {
    throw new Error(
      `Evidence request failed: ${response.status}`,
    )
  }

  return (await response.json()) as EvidenceResponse
}
