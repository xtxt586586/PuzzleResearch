export type ResearchTreeNodeStatus =
  | "pending"
  | "running"
  | "completed"
  | "empty"
  | "failed"
  | "skipped"

export type ResearchTreeNodeSummary = {
  id: string
  topic: string
  query: string
  depth: number
  parentId?: string
  status: ResearchTreeNodeStatus
  resultCount: number
  sourceCount: number
  findings: ResearchTreeFinding[]
}

export type ResearchTreeFinding = {
  query: string
  title: string
  url: string
  snippet: string
}

export type ResearchTreeBudget = {
  maxSeconds?: number
  maxTokens?: number
  elapsedSeconds?: number
  tokensUsed?: number
  searchesUsed?: number
  stopReason?: string
}

export type ResearchTreeFreshness = {
  totalResults?: number
  knownCount?: number
  unknownCount?: number
  fresh7Count?: number
  fresh30Count?: number
  stale180Count?: number
  fresh30Ratio?: number
  stale180Ratio?: number
}

export type ResearchTreeQuality = {
  queryCoverageScore?: number
  dimensionsCovered: string[]
  dimensionsMissing: string[]
  freshness?: ResearchTreeFreshness
  timeSensitiveQuery?: boolean
  freshnessWarning?: string
}

export type ResearchTreeSummary = {
  topic?: string
  epoch?: number
  total: number
  completed: number
  failed: number
  active: number
  pending: number
  nodes: ResearchTreeNodeSummary[]
  budget?: ResearchTreeBudget
  quality?: ResearchTreeQuality
}

export function countResearchTreeNodes(
  tree: Pick<ResearchTreeSummary, "nodes">,
): Pick<ResearchTreeSummary, "total" | "completed" | "failed" | "active" | "pending"> {
  const nodes = tree.nodes ?? []
  return {
    total: nodes.length,
    completed: nodes.filter((node) => node.status === "completed" || node.status === "empty").length,
    failed: nodes.filter((node) => node.status === "failed").length,
    active: nodes.filter((node) => node.status === "running").length,
    pending: nodes.filter((node) => node.status === "pending" || node.status === "skipped").length,
  }
}

type UnknownRecord = Record<string, unknown>

function asRecord(value: unknown): UnknownRecord | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : undefined
}

function numberValue(value: unknown, fallback = 0): number {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

function normalizeStatus(value: unknown): ResearchTreeNodeStatus {
  const status = String(value ?? "pending").toLowerCase()
  if (status === "in_progress" || status === "in-progress") return "running"
  if (
    status === "pending" ||
    status === "running" ||
    status === "completed" ||
    status === "empty" ||
    status === "failed" ||
    status === "skipped"
  ) {
    return status
  }
  return "pending"
}

function normalizeNode(value: unknown, fallbackId = ""): ResearchTreeNodeSummary | undefined {
  const node = asRecord(value)
  if (!node) return undefined

  const id = String(node.id ?? node.node_id ?? fallbackId).trim()
  if (!id) return undefined
  const sources = Array.isArray(node.sources)
    ? node.sources.length
    : Array.isArray(node.source_urls)
      ? node.source_urls.length
      : undefined

  return {
    id,
    topic: String(node.topic ?? "").trim(),
    query: String(node.query ?? "").trim(),
    depth: Math.max(0, numberValue(node.depth)),
    parentId:
      typeof (node.parent_id ?? node.parentId) === "string"
        ? String(node.parent_id ?? node.parentId)
        : undefined,
    status: normalizeStatus(node.status),
    resultCount: Math.max(
      0,
      numberValue(node.result_count ?? node.findings_count ?? node.finding_count),
    ),
    sourceCount: Math.max(
      0,
      numberValue(node.sources_count ?? sources),
    ),
    findings: normalizeFindings(node.findings ?? node.results),
  }
}

function normalizeFindings(value: unknown): ResearchTreeFinding[] {
  if (!Array.isArray(value)) return []
  const findings: ResearchTreeFinding[] = []
  const seen = new Set<string>()
  for (const item of value) {
    const finding = asRecord(item)
    if (!finding) continue
    const normalized = {
      query: String(finding.query ?? "").trim(),
      title: String(finding.title ?? "").trim(),
      url: String(finding.url ?? "").trim(),
      snippet: String(finding.snippet ?? finding.summary ?? "").trim(),
    }
    const key = normalized.url || `${normalized.title}:${normalized.snippet}`
    if ((!normalized.title && !normalized.snippet) || seen.has(key)) continue
    seen.add(key)
    findings.push(normalized)
  }
  return findings.slice(-5)
}

function normalizeBudget(value: unknown): ResearchTreeBudget | undefined {
  const root = asRecord(value)
  if (!root) return undefined
  const budget =
    asRecord(root.research_budget) ??
    asRecord(root.linear_budget) ??
    asRecord(root.tree_budget) ??
    asRecord(root.budget) ??
    root
  const result: ResearchTreeBudget = {
    maxSeconds: optionalNumber(budget.max_seconds ?? budget.maxSeconds),
    maxTokens: optionalNumber(budget.max_tokens ?? budget.maxTokens),
    elapsedSeconds: optionalNumber(
      budget.elapsed_seconds ?? budget.deepsearch_elapsed_seconds ?? budget.elapsedSeconds,
    ),
    tokensUsed: optionalNumber(
      budget.tokens_used ?? budget.deepsearch_tokens_used ?? budget.tokensUsed,
    ),
    searchesUsed: optionalNumber(budget.searches_used ?? budget.searchesUsed),
    stopReason: optionalString(
      budget.stop_reason ?? budget.budget_stop_reason ?? budget.stopReason,
    ),
  }

  return Object.values(result).some((item) => item !== undefined)
    ? result
    : undefined
}

export function normalizeResearchBudget(
  value: unknown,
): ResearchTreeBudget | undefined {
  return normalizeBudget(value)
}

export function mergeResearchBudget(
  current: ResearchTreeBudget | undefined,
  value: unknown,
): ResearchTreeBudget | undefined {
  const budget = normalizeResearchBudget(value)
  return budget ? { ...current, ...budget } : current
}

function normalizeQuality(value: unknown): ResearchTreeQuality | undefined {
  const root = asRecord(value)
  if (!root) return undefined
  const summary = asRecord(root.quality_summary)
  const nested = asRecord(root.quality)
  const source = summary ?? nested ?? root
  const coverage = asRecord(source.query_coverage) ?? asRecord(root.query_coverage)
  const freshness =
    asRecord(source.freshness_summary) ?? asRecord(root.freshness_summary)

  const score = optionalRatio(
    source.query_coverage_score ??
      root.query_coverage_score ??
      coverage?.score,
  )
  const dimensionsCovered = stringList(
    source.query_dimensions_covered ??
      root.query_dimensions_covered ??
      coverage?.covered_dimensions,
  )
  const dimensionsMissing = stringList(
    source.query_dimensions_missing ??
      root.query_dimensions_missing ??
      coverage?.missing_dimensions,
  )
  const freshnessSummary = freshness
    ? {
        totalResults: optionalNumber(freshness.total_results ?? freshness.totalResults),
        knownCount: optionalNumber(freshness.known_count ?? freshness.knownCount),
        unknownCount: optionalNumber(freshness.unknown_count ?? freshness.unknownCount),
        fresh7Count: optionalNumber(freshness.fresh_7_count ?? freshness.fresh7Count),
        fresh30Count: optionalNumber(freshness.fresh_30_count ?? freshness.fresh30Count),
        stale180Count: optionalNumber(freshness.stale_180_count ?? freshness.stale180Count),
        fresh30Ratio: optionalRatio(freshness.fresh_30_ratio ?? freshness.fresh30Ratio),
        stale180Ratio: optionalRatio(freshness.stale_180_ratio ?? freshness.stale180Ratio),
      }
    : undefined
  const timeSensitive =
    typeof source.time_sensitive_query === "boolean"
      ? source.time_sensitive_query
      : typeof root.time_sensitive_query === "boolean"
        ? root.time_sensitive_query
        : undefined
  const warning = optionalString(
    source.freshness_warning ?? root.freshness_warning,
  )

  if (
    score === undefined &&
    !freshnessSummary &&
    timeSensitive === undefined &&
    !warning
  ) {
    return undefined
  }

  return {
    queryCoverageScore: score,
    dimensionsCovered,
    dimensionsMissing,
    freshness: freshnessSummary,
    timeSensitiveQuery: timeSensitive,
    freshnessWarning: warning,
  }
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => String(item).trim()).filter(Boolean)
    : []
}

function optionalRatio(value: unknown): number | undefined {
  const number = optionalNumber(value)
  return number === undefined ? undefined : Math.min(1, number)
}

function optionalNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, number) : undefined
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim()
    ? value.trim()
    : undefined
}

function withNodes(
  tree: Omit<
    ResearchTreeSummary,
    "total" | "completed" | "failed" | "active" | "pending" | "nodes"
  >,
  nodes: ResearchTreeNodeSummary[],
): ResearchTreeSummary {
  const counts = countResearchTreeNodes({ nodes })
  return {
    ...tree,
    nodes,
    ...counts,
  }
}

export function normalizeResearchTree(
  value: unknown,
  quality?: unknown,
): ResearchTreeSummary | undefined {
  const record = asRecord(value)
  if (!record) return undefined

  const rawNodes = Array.isArray(record.nodes)
    ? record.nodes.map((node) => normalizeNode(node))
    : asRecord(record.nodes)
      ? Object.entries(record.nodes as UnknownRecord).map(([id, node]) => normalizeNode(node, id))
      : []
  const nodes = rawNodes.filter(
    (node): node is ResearchTreeNodeSummary => node !== undefined,
  )
  const tree = {
    topic: optionalString(record.topic),
    epoch: optionalNumber(record.epoch),
    budget:
      normalizeBudget(quality) ??
      normalizeBudget(record.budget) ??
      normalizeBudget(record.tree_budget),
    quality:
      normalizeQuality(quality) ??
      normalizeQuality(record.quality_summary) ??
      normalizeQuality(record.quality) ??
      normalizeQuality(record),
  }

  return withNodes(tree, nodes)
}

export function normalizeResearchSessionSnapshot(
  value: unknown,
): { tree?: ResearchTreeSummary; budget?: ResearchTreeBudget } {
  const snapshot = asRecord(value)
  if (!snapshot) return {}

  const state = asRecord(snapshot.state) ?? {}
  const artifacts =
    asRecord(state.deepsearch_artifacts) ??
    asRecord(snapshot.deepsearch_artifacts) ??
    {}
  const tree = normalizeResearchTree(
    state.research_tree ?? artifacts.research_tree,
    artifacts,
  )
  const budget =
    normalizeResearchBudget(artifacts) ??
    normalizeResearchBudget(snapshot.deepsearch_artifacts) ??
    normalizeResearchBudget(state)

  return {
    ...(tree ? { tree } : {}),
    ...(budget ? { budget } : {}),
  }
}

export function mergeResearchNodeUpdate(
  current: ResearchTreeSummary | undefined,
  value: unknown,
  statusOverride?: ResearchTreeNodeStatus,
): ResearchTreeSummary | undefined {
  const payload = asRecord(value)
  const node = normalizeNode(
    statusOverride && payload
      ? { ...payload, status: payload.status ?? statusOverride }
      : payload,
  )
  if (!node) return current

  const tree = current ?? withNodes({}, [])
  const index = tree.nodes.findIndex((item) => item.id === node.id)
  const nodes = [...tree.nodes]
  if (index < 0) {
    nodes.push(node)
  } else {
    const previous = nodes[index]
    nodes[index] = {
      id: node.id,
      topic: typeof payload?.topic === "string" ? node.topic : previous.topic,
      query: typeof payload?.query === "string" ? node.query : previous.query,
      depth: payload?.depth !== undefined ? node.depth : previous.depth,
      parentId:
        payload && ("parent_id" in payload || "parentId" in payload)
          ? node.parentId
          : previous.parentId,
      status:
        payload?.status !== undefined || statusOverride
          ? node.status
          : previous.status,
      resultCount:
        payload &&
        ("result_count" in payload ||
          "findings_count" in payload ||
          "finding_count" in payload)
          ? node.resultCount
          : previous.resultCount,
      sourceCount:
        payload &&
        ("sources_count" in payload ||
          "sources" in payload ||
          "source_urls" in payload)
          ? node.sourceCount
          : previous.sourceCount,
      findings:
        payload && ("findings" in payload || "results" in payload)
          ? normalizeFindings([...previous.findings, ...node.findings])
          : previous.findings,
    }
  }
  return withNodes(tree, nodes)
}

export function mergeResearchTreeBudget(
  current: ResearchTreeSummary | undefined,
  value: unknown,
): ResearchTreeSummary | undefined {
  if (!current) return current
  const budget = mergeResearchBudget(current.budget, value)
  return budget ? { ...current, budget } : current
}

export function mergeResearchTreeQuality(
  current: ResearchTreeSummary | undefined,
  value: unknown,
): ResearchTreeSummary | undefined {
  if (!current) return current
  const quality = normalizeQuality(value)
  if (!quality) return current
  return {
    ...current,
    quality: {
      ...current.quality,
      ...quality,
      freshness: {
        ...current.quality?.freshness,
        ...quality.freshness,
      },
    },
  }
}
