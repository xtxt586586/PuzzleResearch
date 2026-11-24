"use client"

import {
  ArrowDown,
  ArrowUp,
  BarChart3,
  BookmarkCheck,
  BookmarkPlus,
  BookOpen,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code2,
  Compass,
  Copy,
  Download,
  ExternalLink,
  FileText,
  FolderOpen,
  Globe2,
  History,
  LayoutDashboard,
  LoaderCircle,
  ListChecks,
  Maximize2,
  MessageSquare,
  Mic,
  MicOff,
  Minimize2,
  Moon,
  Monitor,
  Network,
  Paperclip,
  PanelLeft,
  PanelRight,
  Pencil,
  Pin,
  PinOff,
  Plus,
  Plug,
  Rocket,
  RefreshCw,
  Settings,
  Sparkles,
  Sun,
  Search,
  Trash2,
  Upload,
  type LucideIcon,
  Volume2,
  VolumeX,
  Zap,
  Wrench,
  X,
} from "lucide-react"
import ReactMarkdown from "react-markdown"
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism"
import rehypeKatex from "rehype-katex"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import {
  Children,
  isValidElement,
  lazy,
  memo,
  Suspense,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react"

import {
  cancelChat,
  getMcpConfig,
  getApiBaseUrl,
  getPublicModelConfig,
  streamChat,
  streamThreadEvents,
  updateMcpConfig,
  type ChatMessage,
  type ChatMode,
  type ImagePayload,
  type RuntimeEvent,
} from "../lib/api"

import {
  deleteSession,
  getSessionState,
  listSessions,
  resumeInterrupt,
  type InterruptResumeDecision,
  type SessionSummary,
} from "../lib/session-api"

import {
  getSessionEvidence,
  type EvidenceClaim,
  type EvidencePassage,
  type EvidenceResponse,
  type EvidenceSource,
} from "../lib/evidence-api"
import {
  firstPendingInterrupt,
  pendingInterruptAfterRuntimeEvent,
} from "../lib/runtime-event-lifecycle"
import { createStreamTextBatcher } from "../lib/stream-text-batcher"
import {
  deleteKnowledgeDocument,
  listKnowledgeDocuments,
  searchKnowledgeDocuments,
  uploadKnowledgeDocument,
  type KnowledgeDocument,
  type KnowledgeSearchResult,
} from "../lib/document-api"
import {
  countResearchTreeNodes,
  mergeResearchNodeUpdate,
  mergeResearchBudget,
  mergeResearchTreeBudget,
  mergeResearchTreeQuality,
  normalizeResearchBudget,
  normalizeResearchSessionSnapshot,
  normalizeResearchTree,
  type ResearchTreeBudget,
  type ResearchTreeSummary,
} from "../lib/research-tree"

import {
  BrowserPreviewPanel,
  type BrowserPreviewOperation,
} from "./browser-preview"
import { WorkspaceFilesPanel } from "./workspace-files-panel"
import { McpDiagnosticsPanel } from "./mcp-diagnostics-panel"
import { MermaidDiagram } from "./mermaid-diagram"
import { SearchHealthPanel } from "./search-health-panel"
import { TriggersPanel } from "./triggers-panel"
import {
  parseStructuredData,
  StructuredDataTable,
  type StructuredDataFormat,
} from "./structured-data-table"

import {
  createSavedAnswer,
  loadSavedAnswers,
  persistSavedAnswers,
  type SavedAnswer,
} from "../lib/saved-answers"
import {
  createSavedArtifact,
  loadSavedArtifacts,
  persistSavedArtifacts,
  savedArtifactSourceKey,
  type GeneratedArtifact,
  type SavedArtifact,
} from "../lib/saved-artifacts"

import {
  loadSessionPreferences,
  persistSessionPreferences,
  type SessionPreferences,
} from "../lib/session-preferences"
import { getBrowserOperationUpdate } from "../lib/browser-events"

const SyntaxHighlighter = lazy(() =>
  import("react-syntax-highlighter").then(({ Prism }) => ({
    default: Prism,
  })),
)

type VoiceInputState =
  | "idle"
  | "checking"
  | "recording"
  | "recognizing"
  | "processing"

type ChatImageAttachment = ImagePayload & {
  id: string
  preview: string
  size: number
}

type BrowserSpeechRecognitionResult = {
  0: { transcript: string }
  isFinal: boolean
}

type BrowserSpeechRecognitionEvent = {
  resultIndex: number
  results: ArrayLike<BrowserSpeechRecognitionResult>
}

type BrowserSpeechRecognitionLike = {
  lang: string
  interimResults: boolean
  continuous: boolean
  onresult: ((event: BrowserSpeechRecognitionEvent) => void) | null
  onerror: ((event: { error: string }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
  abort(): void
}

type BrowserSpeechRecognitionConstructor = new () => BrowserSpeechRecognitionLike

function audioFormatFromMimeType(mimeType: string): string {
  const normalized = mimeType.toLowerCase()

  if (normalized.includes("webm")) return "webm"
  if (normalized.includes("ogg")) return "ogg"
  if (normalized.includes("mp4") || normalized.includes("m4a")) return "m4a"
  if (normalized.includes("wav")) return "wav"

  return "webm"
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  const chunkSize = 0x8000
  let binary = ""

  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(
      ...bytes.subarray(offset, offset + chunkSize),
    )
  }

  return window.btoa(binary)
}

async function imageFileToAttachment(
  file: File,
): Promise<ChatImageAttachment> {
  const mime = file.type.toLowerCase()
  if (!mime.startsWith("image/")) {
    throw new Error(`“${file.name}”不是支持的图片文件。`)
  }

  const data = arrayBufferToBase64(await file.arrayBuffer())
  return {
    id: crypto.randomUUID(),
    name: file.name,
    mime,
    data,
    preview: `data:${mime};base64,${data}`,
    size: file.size,
  }
}

type UiMessage = ChatMessage & {
  id: string
  attachments?: ChatImageAttachment[]
  status?: string
  steps?: string[]
  toolCalls?: ToolCall[]
  researchPlan?: ResearchPlan
  researchTree?: ResearchTreeSummary
  researchBudget?: ResearchTreeBudget
  quality?: ResearchQuality
  taskBoard?: TaskBoard
  browserActivities?: BrowserActivity[]
  generatedArtifacts?: GeneratedArtifact[]
}

type PendingInterrupt = {
  assistantId: string
  kind:
    | "tool_approval"
    | "human_review"
    | "plan_review"
    | "sources_review"
    | "draft_review"
    | "generic"
  message: string
  availableActions: string[]
  allowedTools: string[]
  toolName: string
  argumentsText: string
  contentText: string
  snapshotText: string
}

function interruptSubject(kind: PendingInterrupt["kind"]): string {
  switch (kind) {
    case "human_review": return "报告"
    case "draft_review": return "初稿"
    case "sources_review": return "来源"
    case "plan_review": return "研究计划"
    case "tool_approval": return "工具"
    default: return "操作"
  }
}

function interruptWaitMessage(kind: PendingInterrupt["kind"]): string {
  switch (kind) {
    case "human_review": return "等待最终报告审核"
    case "draft_review": return "等待初稿审核"
    case "sources_review": return "等待来源审核"
    case "plan_review": return "等待研究计划审核"
    case "tool_approval": return "等待工具审批"
    default: return "等待确认"
  }
}

function interruptApprovedMessage(kind: PendingInterrupt["kind"]): string {
  switch (kind) {
    case "human_review": return "报告审核通过，研究已恢复。"
    case "draft_review": return "初稿审核通过，开始质量评估。"
    case "sources_review": return "来源审核通过，开始撰写报告。"
    case "plan_review": return "研究计划已批准，开始搜索。"
    default: return `${interruptSubject(kind)}已批准，研究已恢复。`
  }
}

function interruptEditedMessage(kind: PendingInterrupt["kind"]): string {
  switch (kind) {
    case "human_review": return "报告修改已提交，研究已恢复。"
    case "draft_review": return "初稿修改已提交，开始质量评估。"
    case "sources_review": return "补充要求已提交，开始撰写报告。"
    case "plan_review": return "研究计划已修改，研究已恢复。"
    default: return "修改后的工具已执行，研究已恢复。"
  }
}

function interruptRejectedMessage(kind: PendingInterrupt["kind"]): string {
  switch (kind) {
    case "human_review": return "报告未通过人工审核。"
    case "draft_review": return "初稿审核后取消了本次研究。"
    case "sources_review": return "来源审核后取消了本次研究。"
    case "plan_review": return "研究计划已取消。"
    default: return `已拒绝${interruptSubject(kind)}。`
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function normalizePendingInterrupt(
  payload: unknown,
  assistantId: string,
): PendingInterrupt | null {
  const root = asRecord(payload)
  const first = firstPendingInterrupt(payload)
  if (first === null) return null
  const firstRecord = asRecord(first)
  const value = asRecord(firstRecord?.value) ?? firstRecord ?? root

  if (!value) return null

  const prompts = Array.isArray(value.prompts)
    ? value.prompts.map(asRecord).filter(Boolean) as Array<Record<string, unknown>>
    : []
  const prompt = prompts[0]
  const rawActionRequests = value.action_requests ?? prompt?.action_requests
  const actionRequests = Array.isArray(rawActionRequests)
    ? rawActionRequests.map(asRecord).filter(Boolean) as Array<Record<string, unknown>>
    : []
  const rawToolCalls =
    value.tool_calls ??
    prompt?.tool_calls ??
    root?.tool_calls ??
    rawActionRequests
  const parsedToolCalls = Array.isArray(rawToolCalls)
    ? rawToolCalls.map(asRecord).filter(Boolean) as Array<Record<string, unknown>>
    : []
  const toolCalls = parsedToolCalls.length > 0
    ? parsedToolCalls
    : typeof value.tool_name === "string"
      ? [{
          tool_name: value.tool_name,
          arguments: value.arguments,
        }]
      : []
  const firstToolCall = toolCalls[0]
  const rawAllowedTools = Array.isArray(value.allowed_tools)
    ? value.allowed_tools
    : actionRequests.map((item) => item.name)
  const allowedTools = rawAllowedTools
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean)
  const toolName = String(
    firstToolCall?.tool_name ?? firstToolCall?.name ?? "",
  ).trim()
  const rawContent = value.content ?? prompt?.content ?? root?.content
  const contentText = typeof rawContent === "string" ? rawContent : ""
  const rawSnapshot = value.snapshot ?? prompt?.snapshot ?? root?.snapshot
  const snapshotText = typeof rawSnapshot === "string" ? rawSnapshot : ""
  const kind: PendingInterrupt["kind"] =
    value.type === "plan_review" ||
    prompt?.type === "plan_review"
      ? "plan_review"
      : value.type === "sources_review" ||
        prompt?.type === "sources_review"
        ? "sources_review"
      : value.type === "draft_review" ||
        prompt?.type === "draft_review"
        ? "draft_review"
      : value.type === "human_review" ||
    prompt?.type === "human_review" ||
    (!toolName && Boolean(contentText) && (
      value.checkpoint === "final" ||
      typeof value.instruction === "string" ||
      typeof prompt?.instruction === "string"
    ))
        ? "human_review"
        : value.type === "tool_approval" || toolName
        ? "tool_approval"
        : "generic"

  const stringList = (candidate: unknown): string[] => {
    if (typeof candidate === "string") return [candidate]
    return Array.isArray(candidate)
      ? candidate.filter((item): item is string => typeof item === "string")
      : []
  }
  const configuredActions = [
    ...stringList(value.available_actions),
    ...stringList(value.allowed_decisions),
    ...stringList(prompt?.available_actions),
  ]
  if (Array.isArray(value.review_configs)) {
    for (const item of value.review_configs) {
      configuredActions.push(
        ...stringList(asRecord(item)?.allowed_decisions),
      )
    }
  }
  const actionAliases: Record<string, string> = {
    accept: "approve",
    approved: "approve",
    modify: "edit",
    deny: "reject",
    rejected: "reject",
  }
  const availableActions = Array.from(new Set(
    configuredActions
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean)
      .map((item) => actionAliases[item] ?? item),
  ))
  if (availableActions.length === 0) {
    availableActions.push("approve", "edit")
  }

  const rawArguments = firstToolCall?.arguments ?? firstToolCall?.args ?? {}
  let argumentsText = "{}"
  try {
    argumentsText = JSON.stringify(rawArguments, null, 2) ?? "{}"
  } catch {
    argumentsText = "{}"
  }
  const message = String(
    value.message ??
      prompt?.message ??
      value.instruction ??
      prompt?.instruction ??
      value.reason ??
      root?.message ??
      (kind === "human_review"
        ? "请审核最终报告。"
        : kind === "plan_review"
          ? "请审核研究计划。"
        : "执行前需要你的确认。"),
  ).trim()

  return {
    assistantId,
    kind,
    message: message || "执行前需要你的确认。",
    availableActions,
    allowedTools: allowedTools.length > 0
      ? allowedTools
      : toolName
        ? [toolName]
        : [],
    toolName,
    argumentsText,
    contentText,
    snapshotText,
  }
}

type ResearchPlan = {
  queries: string[]
  reasoning?: string
  mode?: string
}

type ResearchQuality = {
  overallScore?: number
  citationCoverage?: number
  sourceCount?: number
  citedSourceCount?: number
  gapCount?: number
  sufficient?: boolean
  summary?: string
  missingTopics: string[]
}

function researchNodeStatusLabel(
  status: ResearchTreeSummary["nodes"][number]["status"],
): string {
  switch (status) {
    case "running":
      return "进行中"
    case "completed":
      return "已完成"
    case "empty":
      return "无结果"
    case "failed":
      return "失败"
    case "skipped":
      return "已跳过"
    default:
      return "待开始"
  }
}

function researchDimensionLabel(value: string): string {
  const labels: Record<string, string> = {
    freshness: "时效",
    official: "官方资料",
    evidence: "数据证据",
    risk: "风险与局限",
    implementation: "实践与实施",
  }
  return labels[value] ?? value
}

function researchBudgetReasonLabel(reason?: string): string {
  switch (reason) {
    case "time_budget_exceeded":
      return "达到时间上限"
    case "token_budget_exceeded":
      return "达到令牌上限"
    case "search_budget_exceeded":
      return "达到检索次数上限"
    default:
      return "预算使用情况"
  }
}

type TaskStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"

type TaskBoardItem = {
  id: string
  content: string
  status: TaskStatus
  progress: number
  result?: string
  error?: string
}

type TaskBoard = {
  sections: Array<{
    id: string
    title: string
    tasks: TaskBoardItem[]
  }>
  progress: {
    total: number
    completed: number
    running: number
    failed: number
    pending: number
    cancelled: number
    progressPercent: number
  }
}

type ToolCall = {
  id: string
  name: string
  status: "running" | "completed" | "failed"
  detail?: string
  callCount?: number
  attempt?: number
}

type ToolCallMetadata = {
  callCount?: number
  attempt?: number
}

type BrowserActivity = {
  id: string
  tool: string
  action: string
  url?: string
  title?: string
  screenshotUrl?: string
}

type BrowserEventScreenshot = {
  id: string
  url: string
  action: string
  pageUrl?: string
  title?: string
  timestamp: number
}

function normalizeToolSequence(
  value: unknown,
): number | undefined {
  const numberValue =
    typeof value === "number"
      ? value
      : Number(value)

  return Number.isInteger(numberValue) &&
    numberValue > 0
    ? numberValue
    : undefined
}

function summarizeToolCalls(
  toolCalls: ToolCall[],
): Array<ToolCall & { count: number }> {
  const summary: Array<ToolCall & { count: number }> = []

  for (const toolCall of toolCalls) {
    const existing = summary.find(
      (item) =>
        item.name === toolCall.name &&
        item.status === toolCall.status,
    )

    if (existing) {
      existing.count += 1
      existing.detail ||= toolCall.detail
      continue
    }

    summary.push({
      ...toolCall,
      count: 1,
    })
  }

  return summary
}

function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException &&
      error.name === "AbortError") ||
    (error instanceof Error &&
      error.name === "AbortError")
  )
}

const finalResponseNodes = new Set([
  "direct_answer",
  "writer",
  "deep_writer",
  "finalize",
])

function isPlannerPayload(value: string): boolean {
  const candidate = value.trim()

  if (!candidate.startsWith("{") || !candidate.endsWith("}")) {
    return false
  }

  try {
    const parsed = JSON.parse(candidate) as unknown

    return (
      parsed !== null &&
      typeof parsed === "object" &&
      "queries" in parsed &&
      "reasoning" in parsed
    )
  } catch {
    return false
  }
}

function isFinalTextEvent(
  event: RuntimeEvent,
  content: string,
): boolean {
  const node =
    typeof event.data.node === "string"
      ? event.data.node
      : ""

  return (
    !isPlannerPayload(content) &&
    (!node || finalResponseNodes.has(node))
  )
}

function userFacingErrorMessage(value: string): string {
  const normalized = value.toLowerCase()

  if (
    normalized.includes("insufficient balance") ||
    normalized.includes("insufficient_balance")
  ) {
    return "模型服务余额不足，请更换模型或补充服务额度。"
  }

  if (
    normalized.includes("invalid_api_key") ||
    normalized.includes("incorrect api key")
  ) {
    return "模型密钥无效，请检查相关配置。"
  }

  if (
    normalized.includes("failed to fetch") ||
    normalized.includes("networkerror") ||
    normalized.includes("network request failed")
  ) {
    return "无法连接到服务，请检查后端是否已启动。"
  }

  const requestStatus = value.match(
    /(?:chat request failed|request failed)\D*(\d{3})/i,
  )
  if (requestStatus) {
    return `请求未能完成（HTTP ${requestStatus[1]}），请稍后重试。`
  }

  return /[\u3400-\u9fff]/.test(value)
    ? value
    : "研究过程中发生错误，请稍后重试。"
}

function localizeRuntimeStatus(value: string): string {
  const status = value.trim()

  if (!status) {
    return "正在处理请求…"
  }

  if (status.startsWith("开始执行：")) {
    return "正在执行研究流程"
  }

  if (/[\u3400-\u9fff]/.test(status)) {
    return status
  }

  const knownStatuses: Record<string, string> = {
    processing: "正在处理请求…",
    planning: "正在规划研究步骤…",
    searching: "正在搜索资料…",
    completed: "研究已完成",
    cancelled: "研究已取消",
    failed: "研究失败",
  }

  return knownStatuses[status.toLowerCase()] ?? "正在处理请求…"
}

function messageStatusLabel(status?: string): string {
  if (!status) {
    return ""
  }

  const labels: Record<string, string> = {
    Processing: "处理中",
    Completed: "已完成",
    Cancelled: "已取消",
    Failed: "失败",
    Restored: "已恢复",
  }

  return labels[status] ?? status
}

function sessionStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    active: "进行中",
    running: "进行中",
    paused: "已暂停",
    completed: "已完成",
    complete: "已完成",
    failed: "失败",
    cancelled: "已取消",
    canceled: "已取消",
    pending: "等待中",
  }

  return labels[status.toLowerCase()] ?? "状态未知"
}

function sessionRouteLabel(route: string): string {
  const labels: Record<string, string> = {
    direct: "直接回答",
    agent: "智能体",
    deep: "深度研究",
  }

  return labels[route.toLowerCase()] ?? "研究"
}

function toolDisplayName(name: string): string {
  const labels: Record<string, string> = {
    ask_human: "向用户提问",
    web_search: "网页搜索",
    crawl_url: "网页抓取",
    crawl_urls: "批量网页抓取",
    write_file: "创建工作区文件",
    update_file: "更新工作区文件",
    read_file: "读取工作区文件",
    list_files: "查看工作区文件",
    str_replace: "替换文件内容",
    delete_file: "删除工作区文件",
    upload_file: "上传二进制文件",
    download_file: "读取二进制文件",
    execute_command: "执行工作区命令",
    check_output: "查看命令输出",
    kill_process: "终止后台命令",
    list_processes: "列出后台命令",
    install_package: "安装工作区依赖",
    expose_port: "查看本地端口",
    scaffold_web_project: "创建 Web 项目",
    deploy_web_project: "启动 Web 项目",
    python_executor: "Python 代码执行器",
    browser_search: "浏览器搜索",
    browser_navigate: "浏览器导航",
    browser_click: "浏览器点击",
    browser_back: "返回上一页",
    browser_extract_text: "提取网页文本",
    browser_list_links: "列出网页链接",
    browser_reset: "重置浏览器",
    browser_screenshot: "网页截图",
  }

  return labels[name] ?? name
}

function resolveApiUrl(value: string): string {
  if (/^(?:https?:|data:|blob:)/i.test(value)) {
    return value
  }

  const normalized = value.startsWith("/")
    ? value
    : `/${value}`

  return `${getApiBaseUrl()}${normalized}`
}

function safeMarkdownUrl(value: string): string {
  if (/^data:image\/svg\+xml;base64,[A-Za-z0-9+/]+={0,2}$/i.test(value)) {
    return value
  }

  const colon = value.indexOf(":")
  const questionMark = value.indexOf("?")
  const numberSign = value.indexOf("#")
  if (
    colon === -1 ||
    (questionMark !== -1 && colon > questionMark) ||
    (numberSign !== -1 && colon > numberSign) ||
    /^(https?|ircs?|mailto|xmpp)$/i.test(value.slice(0, colon))
  ) {
    return value
  }

  return ""
}

function speechTextFromMarkdown(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!?\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*(?:[-*+] |\d+\. )/gm, "")
    .replace(/[>*_~]/g, "")
    .replace(/\s+/g, " ")
    .trim()
}

async function requestReportExport(
  threadId: string,
  format: "markdown" | "html" | "json" | "pdf" | "docx",
  messages: ChatMessage[],
): Promise<Blob> {
  const response = await fetch(
    `${getApiBaseUrl()}/api/export/${encodeURIComponent(
      threadId,
    )}?format=${format}`,
    {
      method: "POST",
      credentials: "include",
      headers: {
        Accept:
          format === "json"
            ? "application/json"
            : format === "html"
              ? "text/html"
              : format === "markdown"
                ? "text/markdown"
                : format === "pdf"
                  ? "application/pdf"
                  : "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: messages.map(({ role, content }) => ({
          role,
          content,
        })),
      }),
    },
  )

  if (!response.ok) {
    throw new Error(`报告导出失败：${response.status}`)
  }

  return response.blob()
}

function browserActionLabel(action: string): string {
  const labels: Record<string, string> = {
    search_completed: "搜索完成",
    navigate_completed: "页面已打开",
    click_completed: "已点击链接",
    back_completed: "已返回上一页",
    text_extracted: "已提取正文",
    links_listed: "已整理链接",
    session_reset: "浏览器会话已重置",
    screenshot: "已截取页面",
  }

  return labels[action] ?? action.replaceAll("_", " ")
}

function normalizeGeneratedArtifact(
  value: unknown,
  fallbackId: string,
): GeneratedArtifact | undefined {
  if (value === null || typeof value !== "object") {
    return undefined
  }

  const record = value as Record<string, unknown>
  const rawType = String(record.type ?? "text").toLowerCase()
  const type = [
    "code",
    "image",
    "text",
    "file",
    "report",
    "chart",
  ].includes(rawType)
    ? (rawType as GeneratedArtifact["type"])
    : "text"
  const title = String(
    record.title ?? record.name ?? "生成内容",
  ).trim()
  const content =
    record.content !== undefined
      ? valueToDisplayText(record.content)
      : record.text !== undefined
        ? valueToDisplayText(record.text)
        : undefined
  const url = String(
    record.url ?? record.download_url ?? "",
  ).trim()
  const mimeType = String(
    record.mime_type ?? record.mimeType ?? "",
  ).trim()

  if (!content && !url) {
    return undefined
  }

  return {
    id: String(record.id ?? fallbackId),
    type,
    title: title || "生成内容",
    content: content || undefined,
    url: url || undefined,
    mimeType: mimeType || undefined,
  }
}

function generatedArtifactTypeLabel(
  type: GeneratedArtifact["type"],
): string {
  const labels: Record<GeneratedArtifact["type"], string> = {
    code: "代码",
    image: "图片",
    text: "文本",
    file: "文件",
    report: "报告",
    chart: "图表",
  }

  return labels[type]
}

function GeneratedArtifactList({
  artifacts,
  canSave = true,
  isSaved,
  onSave,
}: {
  artifacts: GeneratedArtifact[]
  canSave?: boolean
  isSaved?: (artifact: GeneratedArtifact) => boolean
  onSave?: (artifact: GeneratedArtifact) => void
}) {
  if (artifacts.length === 0) {
    return null
  }

  return (
    <div
      className="generated-artifact-list"
      aria-label="生成制品列表"
    >
      {artifacts.map((artifact) => {
        const isVisual =
          (artifact.type === "image" ||
            artifact.type === "chart") &&
          Boolean(artifact.url)

        return (
          <div
            className="generated-artifact"
            key={artifact.id}
          >
            <div className="generated-artifact-heading">
              <span>
                {artifact.type === "code" ? (
                  <Code2 size={14} />
                ) : artifact.type === "chart" ? (
                  <BarChart3 size={14} />
                ) : (
                  <FileText size={14} />
                )}
                <strong>{artifact.title}</strong>
              </span>
              <div className="generated-artifact-actions">
                <small>
                  {generatedArtifactTypeLabel(artifact.type)}
                </small>
                {onSave && (
                  <button
                    className="generated-artifact-save"
                    type="button"
                    title={!canSave ? "正在读取资料库" : isSaved?.(artifact) ? "已保存到资料库" : "保存到资料库"}
                    aria-label={!canSave ? "正在读取资料库" : isSaved?.(artifact) ? "已保存到资料库" : "保存到资料库"}
                    disabled={!canSave || isSaved?.(artifact)}
                    onClick={() => onSave(artifact)}
                  >
                    {isSaved?.(artifact) ? (
                      <BookmarkCheck size={14} />
                    ) : (
                      <BookmarkPlus size={14} />
                    )}
                  </button>
                )}
              </div>
            </div>

            {isVisual && artifact.url && (
              <a
                className="generated-artifact-preview"
                href={resolveApiUrl(artifact.url)}
                target="_blank"
                rel="noreferrer"
              >
                <img
                  src={resolveApiUrl(artifact.url)}
                  alt={artifact.title}
                  loading="lazy"
                />
              </a>
            )}

            {artifact.content && (
              <pre className="generated-artifact-content">
                {artifact.content}
              </pre>
            )}

            {artifact.url && !isVisual && (
              <a
                className="generated-artifact-download"
                href={resolveApiUrl(artifact.url)}
                download
                target="_blank"
                rel="noreferrer"
              >
                <Download size={13} />
                下载制品
              </a>
            )}
          </div>
        )
      })}
    </div>
  )
}

function evidencePercent(value: unknown): number | undefined {
  const number = Number(value)

  if (!Number.isFinite(number)) {
    return undefined
  }

  return Math.round(Math.max(0, Math.min(1, number)) * 100)
}

function evidenceClaimStatus(status: string): {
  label: string
  className: string
} {
  const normalized = status.trim().toLowerCase()

  if (
    normalized.includes("contradict") ||
    normalized.includes("conflict")
  ) {
    return { label: "存在冲突", className: "contradicted" }
  }

  if (
    normalized.includes("unsupported") ||
    normalized.includes("missing")
  ) {
    return { label: "缺少证据", className: "unsupported" }
  }

  if (
    normalized.includes("support") ||
    normalized.includes("verified") ||
    normalized.includes("confirm")
  ) {
    return { label: "有证据支持", className: "supported" }
  }

  return { label: status || "待核验", className: "pending" }
}

function EvidencePanel({
  evidence,
  fallbackSources,
}: {
  evidence: EvidenceResponse | null
  fallbackSources: EvidenceSource[]
}) {
  const sources = evidence?.sources.length
    ? evidence.sources
    : fallbackSources
  const claims = evidence?.claims ?? []
  const passages = evidence?.passages ?? []
  const quality = evidence?.quality_summary ?? {}
  const overallScore = evidencePercent(
    quality.overall_score ?? quality.quality_overall_score,
  )
  const citationCoverage = evidencePercent(
    quality.citation_coverage ?? quality.citation_coverage_score,
  )
  const sourceCount = Number(quality.source_count)
  const gapCount = Number(quality.gap_count)
  const missingTopics = Array.isArray(quality.missing_topics)
    ? quality.missing_topics.map(String).filter(Boolean)
    : []
  const hasQuality =
    overallScore !== undefined ||
    citationCoverage !== undefined ||
    Number.isFinite(sourceCount) ||
    Number.isFinite(gapCount) ||
    typeof quality.summary === "string" ||
    missingTopics.length > 0

  if (sources.length === 0 && claims.length === 0 && passages.length === 0 && !hasQuality) {
    return null
  }

  return (
    <section
      id="evidence-sources"
      className="evidence-tray"
      aria-label="证据与来源"
    >
      <div className="evidence-header">
        <span>
          <BookOpen size={14} />
          证据与来源
        </span>
        <span>
          {sources.length > 0 ? `${sources.length} 个来源` : "研究证据"}
        </span>
      </div>

      {hasQuality && (
        <div className="evidence-quality-summary">
          {overallScore !== undefined && (
            <span>
              <strong>{overallScore}%</strong>
              综合质量
            </span>
          )}
          {citationCoverage !== undefined && (
            <span>
              <strong>{citationCoverage}%</strong>
              引用覆盖
            </span>
          )}
          {Number.isFinite(sourceCount) && (
            <span>
              <strong>{sourceCount}</strong>
              已引用来源
            </span>
          )}
          {Number.isFinite(gapCount) && gapCount > 0 && (
            <span className="evidence-quality-warning">
              <strong>{gapCount}</strong>
              个待补充问题
            </span>
          )}
        </div>
      )}

      {typeof quality.summary === "string" && quality.summary.trim() && (
        <p className="evidence-quality-note">{quality.summary}</p>
      )}

      {missingTopics.length > 0 && (
        <div className="evidence-missing-topics">
          待补充：{missingTopics.join("、")}
        </div>
      )}

      {claims.length > 0 && (
        <details className="evidence-section" open>
          <summary>
            <span>主张核验</span>
            <small>{claims.length} 条</small>
          </summary>
          <div className="evidence-claims">
            {claims.map((claim: EvidenceClaim, index) => {
              const status = evidenceClaimStatus(claim.status)
              const score = evidencePercent(claim.score)

              return (
                <article className="evidence-claim" key={`${claim.claim}-${index}`}>
                  <div className="evidence-claim-heading">
                    <strong>{claim.claim}</strong>
                    <span className={`evidence-claim-status ${status.className}`}>
                      {status.label}
                    </span>
                  </div>
                  {(score !== undefined || claim.notes) && (
                    <div className="evidence-claim-meta">
                      {score !== undefined && <span>匹配度 {score}%</span>}
                      {claim.notes && <span>{claim.notes}</span>}
                    </div>
                  )}
                  {claim.evidence_passages.length > 0 && (
                    <div className="evidence-claim-quotes">
                      {claim.evidence_passages.slice(0, 3).map((passage, passageIndex) => (
                        <blockquote key={`${passage.snippet_hash ?? passage.url}-${passageIndex}`}>
                          {passage.quote || "已关联证据片段"}
                        </blockquote>
                      ))}
                    </div>
                  )}
                </article>
              )
            })}
          </div>
        </details>
      )}

      {passages.length > 0 && (
        <details className="evidence-section">
          <summary>
            <span>证据片段</span>
            <small>{passages.length} 条</small>
          </summary>
          <div className="evidence-passages">
            {passages.slice(0, 12).map((passage: EvidencePassage, index) => (
              <article className="evidence-passage" key={`${passage.snippet_hash ?? passage.url}-${index}`}>
                <div className="evidence-passage-heading">
                  <strong>{passage.heading || passage.page_title || `证据片段 ${index + 1}`}</strong>
                  <a href={passage.url} target="_blank" rel="noreferrer" title="打开来源">
                    <ExternalLink size={13} />
                  </a>
                </div>
                <p>{passage.quote || passage.text}</p>
              </article>
            ))}
          </div>
        </details>
      )}

      {sources.length > 0 && (
        <div className="evidence-list">
          {sources.map((source, index) => (
            <a
              className="evidence-item"
              href={source.rawUrl || source.url}
              id={`evidence-source-${index + 1}`}
              target="_blank"
              rel="noreferrer"
              key={source.url}
            >
              <span className="evidence-item-copy">
                <strong>{source.title || source.domain || source.url}</strong>
                <small>{source.domain || source.url}</small>
              </span>
              <ExternalLink size={14} />
            </a>
          ))}
        </div>
      )}
    </section>
  )
}

function extractGeneratedArtifacts(
  event: RuntimeEvent,
): GeneratedArtifact[] {
  const rawResult = event.data.result
  const resultRecord =
    rawResult !== null &&
    typeof rawResult === "object" &&
    !Array.isArray(rawResult)
      ? (rawResult as Record<string, unknown>)
      : undefined
  const resultTool =
    resultRecord?.tool_name ??
    resultRecord?.tool ??
    resultRecord?.name
  const tool = String(
    event.data.tool ??
      event.data.tool_name ??
      event.data.name ??
      event.data.tool_display_name ??
      resultTool ??
      "tool",
  ).trim()
  const normalizedTool = tool
    .toLowerCase()
    .replace(/[\s_-]+/g, "")
  const isPythonExecutor =
    normalizedTool === "pythonexecutor" ||
    (normalizedTool.includes("python") &&
      (normalizedTool.includes("executor") ||
        normalizedTool.includes("执行器") ||
        normalizedTool.includes("代码执行")))
  const action = String(
    event.data.action ?? event.type,
  ).trim()
  const url = String(
    event.data.page_url ?? event.data.url ?? "",
  ).trim()
  const title = String(
    event.data.title ?? "",
  ).trim()
  const screenshotUrl = String(
    event.data.screenshot_url ?? "",
  ).trim()
  const id = String(
    event.data.event_id ??
      `${tool}:${action}:${url}:${screenshotUrl}`,
  )
  const generatedArtifacts: GeneratedArtifact[] = []

  if (event.type === "tool_screenshot" && screenshotUrl) {
    generatedArtifacts.push({
      id: `screenshot:${id}`,
      type: "image",
      title: title || "浏览器页面截图",
      url: screenshotUrl,
      mimeType: "image/png",
    })
  }

  const rawArtifacts =
    event.data.artifacts ?? resultRecord?.artifacts

  if (Array.isArray(rawArtifacts)) {
    rawArtifacts.forEach((item, index) => {
      const normalized = normalizeGeneratedArtifact(
        item,
        `artifact:${id}:${index}`,
      )

      if (normalized) {
        generatedArtifacts.push(normalized)
      }
    })
  }

  const eventName = String(
    event.data.event_type ?? "",
  ).toLowerCase()
  const isToolResult =
    event.type === "tool_result" ||
    event.type === "tool_end" ||
    (event.type === "tool" &&
      !eventName.includes("start") &&
      !eventName.includes("retry"))

  if (
    isToolResult &&
    generatedArtifacts.length === 0 &&
    isPythonExecutor
  ) {
    const rawOutput = [
      event.data.output,
      event.data.stdout,
      resultRecord?.output,
      resultRecord?.stdout,
      resultRecord?.value,
      resultRecord?.content,
      typeof rawResult === "string" ? rawResult : undefined,
      event.data.value,
    ].find(
      (value) =>
        value !== undefined &&
        value !== null &&
        (typeof value !== "string" || value.trim().length > 0),
    )
    const output = valueToDisplayText(rawOutput).trim()

    if (output) {
      generatedArtifacts.push({
        id: `python-output:${id}`,
        type: "text",
        title: "Python 执行结果",
        content: output,
        mimeType: "text/plain",
      })
    }
  }

  return generatedArtifacts
}

function renderCitationChildren(
  children: ReactNode,
  sourceCount: number,
): ReactNode {
  return Children.toArray(children).flatMap<ReactNode>(
    (child, childIndex): ReactNode | ReactNode[] => {
      if (typeof child !== "string") {
        return [child]
      }

      return child
        .split(/(\[(?:来源\s*)?\d+\])/g)
        .map((part, partIndex) => {
          const match = part.match(
            /^\[(?:来源\s*)?(\d+)\]$/,
          )

          if (!match) {
            return part
          }

          const sourceNumber = match[1]

          if (
            sourceCount === 0 ||
            Number(sourceNumber) > sourceCount
          ) {
            return part
          }

          return (
            <a
              className="citation-badge"
              href={`#evidence-source-${sourceNumber}`}
              key={`${childIndex}-${partIndex}`}
              title={`查看来源 ${sourceNumber}`}
            >
              [{sourceNumber}]
            </a>
          )
        })
    },
  )
}

const AssistantCodeBlock = memo(function AssistantCodeBlock({
  code,
  language,
}: {
  code: string
  language: string
}) {
  const [copied, setCopied] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  async function copyCode() {
    if (!code) {
      return
    }

    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="assistant-code-shell">
      <div className="assistant-code-toolbar">
        <span>{language || "text"}</span>
        <div className="assistant-code-actions">
          <button
            className="assistant-code-copy"
            type="button"
            title={copied ? "已复制" : "复制代码"}
            aria-label={copied ? "已复制" : "复制代码"}
            onClick={() => void copyCode()}
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <button
            className="assistant-code-copy"
            type="button"
            title={collapsed ? "展开代码" : "收起代码"}
            aria-label={collapsed ? "展开代码" : "收起代码"}
            aria-expanded={!collapsed}
            onClick={() => setCollapsed((value) => !value)}
          >
            {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
          </button>
        </div>
      </div>
      {!collapsed && (
        <Suspense
          fallback={
            <pre className="assistant-code-loading">
              <code>{code}</code>
            </pre>
          }
        >
          <SyntaxHighlighter
            language={language || "text"}
            style={oneDark}
            customStyle={{
              margin: 0,
              padding: "12px 15px",
              background: "#282c34",
              fontFamily: "var(--font-mono), ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              fontSize: "13px",
              lineHeight: 1.55,
              whiteSpace: "pre-wrap",
              wordBreak: "normal",
              overflowWrap: "anywhere",
            }}
            codeTagProps={{
              style: {
                whiteSpace: "pre-wrap",
                wordBreak: "normal",
                overflowWrap: "anywhere",
              },
            }}
            showLineNumbers={false}
            wrapLongLines={false}
            PreTag="pre"
          >
            {code}
          </SyntaxHighlighter>
        </Suspense>
      )}
    </div>
  )
})

const AssistantMarkdown = memo(function AssistantMarkdown({
  content,
  sourceCount,
}: {
  content: string
  sourceCount: number
}) {
  return (
    <div className="assistant-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
        urlTransform={safeMarkdownUrl}
        components={{
          a: ({ children, ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noreferrer"
            >
              {children}
            </a>
          ),
          code: ({
            className,
            children,
            ...props
          }) => {
            const codeText = String(children)
            const isBlock =
              Boolean(className) ||
              codeText.includes("\n")

            return (
              <code
                {...props}
                className={
                  isBlock
                    ? className
                    : "assistant-inline-code"
                }
              >
                {children}
              </code>
            )
          },
          pre: ({ children }) => {
            const codeElement = Children.toArray(children)[0]
            let code = ""
            let language = "text"

            if (isValidElement<{
              className?: string
              children?: ReactNode
            }>(codeElement)) {
              code = String(codeElement.props.children ?? "")
              const languages = codeElement.props.className?.split(/\s+/) ?? []
              language = languages
                .find((value) => value.startsWith("language-"))
                ?.slice("language-".length)
                .toLowerCase() || "text"

              if (languages.includes("language-mermaid")) {
                return <MermaidDiagram code={code} />
              }

              const format = languages
                .find((language) => language === "language-json" || language === "language-csv")
                ?.slice("language-".length) as StructuredDataFormat | undefined

              if (format) {
                const dataset = parseStructuredData(format, code)
                if (dataset) {
                  return <StructuredDataTable dataset={dataset} />
                }
              }
            }

            return (
              <AssistantCodeBlock code={code} language={language} />
            )
          },
          p: ({ children, ...props }) => (
            <p {...props}>
              {renderCitationChildren(
                children,
                sourceCount,
              )}
            </p>
          ),
          li: ({ children, ...props }) => (
            <li {...props}>
              {renderCitationChildren(
                children,
                sourceCount,
              )}
            </li>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
})

function contentToText(value: unknown): string {
  if (typeof value === "string") {
    return value
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (
          item !== null &&
          typeof item === "object" &&
          "text" in item
        ) {
          return String(
            (item as { text: unknown }).text,
          )
        }

        return String(item)
      })
      .join("")
  }

  if (
    value !== null &&
    typeof value === "object" &&
    "text" in value
  ) {
    return String(
      (value as { text: unknown }).text,
    )
  }

  return value === undefined || value === null
    ? ""
    : String(value)
}

function valueToDisplayText(value: unknown): string {
  if (typeof value === "string") {
    return value
  }

  if (
    value !== null &&
    typeof value === "object"
  ) {
    try {
      return JSON.stringify(value)
    } catch {
      return contentToText(value)
    }
  }

  return contentToText(value)
}

function normalizeEvidenceSources(
  value: unknown,
): EvidenceSource[] {
  if (!Array.isArray(value)) {
    return []
  }

  const sources: EvidenceSource[] = []
  const seen = new Set<string>()

  for (const item of value) {
    if (item === null || typeof item !== "object") {
      continue
    }

    const record = item as Record<string, unknown>
    const url = String(
      record.url ?? record.rawUrl ?? "",
    ).trim()

    if (!url || seen.has(url)) {
      continue
    }

    seen.add(url)

    let domain: string | null = null
    try {
      domain = new URL(url).hostname
    } catch {
      domain = null
    }

    sources.push({
      title: String(
        record.title ?? domain ?? url,
      ).trim(),
      url,
      rawUrl:
        typeof record.rawUrl === "string"
          ? record.rawUrl
          : null,
      domain:
        typeof record.domain === "string"
          ? record.domain
          : domain,
      provider:
        typeof record.provider === "string"
          ? record.provider
          : null,
      publishedDate:
        typeof record.publishedDate === "string"
          ? record.publishedDate
          : typeof record.published_date === "string"
            ? record.published_date
            : null,
    })
  }

  return sources
}

function normalizeResearchQuality(
  value: Record<string, unknown>,
): ResearchQuality | undefined {
  const asNumber = (candidate: unknown) => {
    const number = Number(candidate)
    return Number.isFinite(number) ? number : undefined
  }

  const missingTopics = Array.isArray(
    value.missing_topics,
  )
    ? value.missing_topics
        .map((topic) => String(topic).trim())
        .filter(Boolean)
    : []

  const overallScore = asNumber(
    value.overall_score ?? value.quality_overall_score,
  )
  const citationCoverage = asNumber(
    value.citation_coverage ?? value.citation_coverage_score,
  )
  const sourceCount = asNumber(value.source_count)
  const citedSourceCount = asNumber(value.cited_source_count)
  const gapCount = asNumber(value.gap_count)
  const sufficient =
    typeof value.sufficient === "boolean" ? value.sufficient : undefined
  const summary =
    typeof value.summary === "string" ? value.summary : undefined

  if (
    overallScore === undefined &&
    citationCoverage === undefined &&
    sourceCount === undefined &&
    citedSourceCount === undefined &&
    gapCount === undefined &&
    sufficient === undefined &&
    !summary &&
    missingTopics.length === 0
  ) {
    return undefined
  }

  return {
    overallScore,
    citationCoverage,
    sourceCount,
    citedSourceCount,
    gapCount,
    sufficient,
    summary,
    missingTopics,
  }
}

function normalizeTaskBoard(
  value: Record<string, unknown>,
): TaskBoard | undefined {
  if (!Array.isArray(value.task_list)) {
    return undefined
  }

  const allowedStatuses = new Set<TaskStatus>([
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
  ])
  const sections: TaskBoard["sections"] = []

  for (const group of value.task_list) {
    if (group === null || typeof group !== "object") continue
    const record = group as Record<string, unknown>
    const rawSection = record.section
    const section =
      rawSection !== null && typeof rawSection === "object"
        ? (rawSection as Record<string, unknown>)
        : {}
    const tasks: TaskBoardItem[] = []

    if (Array.isArray(record.tasks)) {
      for (const rawTask of record.tasks) {
        if (rawTask === null || typeof rawTask !== "object") continue
        const task = rawTask as Record<string, unknown>
        const status = String(task.status ?? "pending") as TaskStatus
        if (!allowedStatuses.has(status)) continue
        tasks.push({
          id: String(task.id ?? crypto.randomUUID()),
          content: String(task.content ?? ""),
          status,
          progress: Math.max(
            0,
            Math.min(100, Number(task.progress) || 0),
          ),
          result:
            typeof task.result === "string" ? task.result : undefined,
          error:
            typeof task.error === "string" ? task.error : undefined,
        })
      }
    }

    sections.push({
      id: String(section.id ?? `section-${sections.length}`),
      title: String(section.title ?? "任务"),
      tasks,
    })
  }

  const rawProgress =
    value.progress !== null && typeof value.progress === "object"
      ? (value.progress as Record<string, unknown>)
      : {}
  const numberValue = (key: string) => {
    const number = Number(rawProgress[key])
    return Number.isFinite(number) ? Math.max(0, number) : 0
  }

  return {
    sections,
    progress: {
      total: numberValue("total"),
      completed: numberValue("completed"),
      running: numberValue("running"),
      failed: numberValue("failed"),
      pending: numberValue("pending"),
      cancelled: numberValue("cancelled"),
      progressPercent: Math.min(100, numberValue("progress_percent")),
    },
  }
}

function restoreSessionMessages(
  value: unknown,
): UiMessage[] {
  if (!Array.isArray(value)) {
    return []
  }

  const restored: UiMessage[] = []

  value.forEach((item, index) => {
    if (
      item === null ||
      typeof item !== "object"
    ) {
      return
    }

    const record = item as Record<string, unknown>

    const rawRole = String(
      record.role ?? record.type ?? "",
    ).toLowerCase()

    if (
      rawRole.includes("system") ||
      rawRole.includes("tool")
    ) {
      return
    }

    const role: ChatMessage["role"] =
      rawRole.includes("human") ||
      rawRole === "user"
        ? "user"
        : "assistant"

    const content = contentToText(
      record.content,
    )

    if (!content.trim()) {
      return
    }

    restored.push({
      id: `restored-${index}`,
      role,
      content,
      status:
        role === "assistant"
          ? "Restored"
          : undefined,
    })
  })

  return restored
}

type NavigationItem = {
  key: "Dashboard" | "Discover" | "Library" | "Triggers"
  label: string
  icon: LucideIcon
}

const navigationItems: NavigationItem[] = [
  {
    key: "Dashboard",
    label: "工作台",
    icon: LayoutDashboard,
  },
  {
    key: "Discover",
    label: "发现",
    icon: Compass,
  },
  {
    key: "Library",
    label: "资料库",
    icon: FolderOpen,
  },
  {
    key: "Triggers",
    label: "自动化",
    icon: Zap,
  },
]

type LibraryFilter = "all" | "active" | "completed" | "failed"
type LibraryKindFilter =
  | "all"
  | "sessions"
  | "saved"
  | "pinned"
  | "artifacts"
  | "documents"

type DiscoverCategory =
  | "Research"
  | "Academic"
  | "Writing"
  | "Coding"

type DiscoverPrompt = {
  title: string
  description: string
  category: DiscoverCategory
  prompt: string
  icon: LucideIcon
}

const discoverPrompts: DiscoverPrompt[] = [
  {
    title: "技术与市场趋势调研",
    description:
      "梳理新兴技术市场的驱动因素、主要参与者与发展方向。",
    category: "Research",
    prompt:
      "请调研【技术或行业】的当前市场趋势，比较主要企业、近期进展、推广障碍及未来三年可能的变化。引用可靠来源，并区分已证实的事实与预测。",
    icon: BarChart3,
  },
  {
    title: "学术研究进展",
    description:
      "查找并比较近期论文、研究方法、主要发现与待解决问题。",
    category: "Academic",
    prompt:
      "请综述【主题】的近期学术研究，列出有影响力及最新的论文，比较研究方法与结论，指出局限并总结尚未解决的问题。请提供引用来源和发表日期。",
    icon: BookOpen,
  },
  {
    title: "智能体框架比较",
    description:
      "从架构、工具调用、记忆机制和生产适用性比较主流智能体框架。",
    category: "Research",
    prompt:
      "请比较适用于生产应用的主流智能体框架，从架构、工具调用、记忆、编排、可观测性、优势与局限等方面进行评估，并使用最新文档和可靠来源支持结论。",
    icon: Compass,
  },
  {
    title: "Python 数据可视化",
    description:
      "根据数据集制作清晰的图表，并说明实现方法与数据假设。",
    category: "Coding",
    prompt:
      "请编写 Python 脚本，读取【数据来源】并清理相关字段，围绕【问题或指标】生成清晰的可视化图表。说明运行方法，并指出数据处理中的假设。",
    icon: Code2,
  },
  {
    title: "论文解读与总结",
    description:
      "提炼论文的研究问题、方法、证据、贡献与局限。",
    category: "Academic",
    prompt:
      "请解读论文【标题或链接】，说明其研究问题、核心方法、主要结果、贡献、假设与局限，并区分作者的主张和你的分析。",
    icon: Search,
  },
  {
    title: "创意写作大纲",
    description:
      "将故事设定整理为包含人物动机和关键情节的结构化大纲。",
    category: "Writing",
    prompt:
      "请根据以下设定创作故事大纲：【故事设定】。提出核心冲突、人物动机、关键转折和结局，确保情节连贯，并避免陈词滥调。",
    icon: Sparkles,
  },
]

const discoverCategories = [
  "All",
  "Research",
  "Academic",
  "Writing",
  "Coding",
] as const

const discoverCategoryLabels: Record<
  (typeof discoverCategories)[number],
  string
> = {
  All: "全部",
  Research: "研究",
  Academic: "学术",
  Writing: "写作",
  Coding: "编程",
}

const fallbackModelOptions = [
  "deepseek-chat",
  "deepseek-reasoner",
]
const selectedModelStorageKey = "puzzleresearch-selected-model"

const suggestions = [
  {
    label: "调研智能体领域的最新进展",
    prompt:
      "请梳理人工智能智能体领域的最新研究与应用进展，比较主流框架的架构、工具调用和记忆机制，并引用可靠来源。",
    icon: BarChart3,
  },
  {
    label: "比较不同技术方案",
    prompt:
      "请比较以下方案：【方案 A】与【方案 B】。从原理、性能、成本、适用场景和局限进行分析，并说明判断依据。",
    icon: Code2,
  },
  {
    label: "解读一篇研究论文",
    prompt:
      "请解读这篇论文：【论文标题或链接】。概括研究问题、方法、主要发现、贡献与局限，并区分作者结论和你的分析。",
    icon: BookOpen,
  },
  {
    label: "撰写结构化研究报告",
    prompt:
      "请围绕【研究主题】撰写结构化研究报告，包含背景、核心问题、证据与来源、分析、结论及仍待验证的问题。",
    icon: Sparkles,
  },
]

export function PuzzleResearchWorkspace() {
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [draft, setDraft] = useState("")
  const [attachments, setAttachments] =
    useState<ChatImageAttachment[]>([])
  const [attachmentsLoading, setAttachmentsLoading] =
    useState(false)
  const [isDraggingImages, setIsDraggingImages] = useState(false)
  const [chatMode, setChatMode] = useState<ChatMode>("direct")
  const [commandMenuOpen, setCommandMenuOpen] =
    useState(false)
  const [mcpMenuOpen, setMcpMenuOpen] = useState(false)
  const [mcpMenuPosition, setMcpMenuPosition] = useState<{
    top: number
    left: number
  } | null>(null)
  const [selectedMcp, setSelectedMcp] =
    useState("")
  const [running, setRunning] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [model, setModel] = useState("deepseek-chat")
  const [modelOptions, setModelOptions] =
    useState<string[]>(fallbackModelOptions)
  const [defaultModel, setDefaultModel] =
    useState("deepseek-chat")
  const [modelConfigStatus, setModelConfigStatus] = useState(
    "正在读取模型配置…",
  )
  const [darkMode, setDarkMode] = useState(false)
  const [activeNav, setActiveNav] = useState("Dashboard")
  const [activity, setActivity] = useState("")
  const [voiceInputState, setVoiceInputState] =
    useState<VoiceInputState>("idle")
  const [copiedMessageId, setCopiedMessageId] =
    useState<string | null>(null)
  const [speakingMessageId, setSpeakingMessageId] =
    useState<string | null>(null)
  const [speechLoadingMessageId, setSpeechLoadingMessageId] =
    useState<string | null>(null)
  const [editingMessageId, setEditingMessageId] =
    useState<string | null>(null)
  const [editingMessageContent, setEditingMessageContent] =
    useState("")
  const [libraryQuery, setLibraryQuery] = useState("")
  const [libraryKindFilter, setLibraryKindFilter] =
    useState<LibraryKindFilter>("all")
  const [libraryFilter, setLibraryFilter] =
    useState<LibraryFilter>("all")
  const [knowledgeDocuments, setKnowledgeDocuments] =
    useState<KnowledgeDocument[]>([])
  const [knowledgeDocumentsLoading, setKnowledgeDocumentsLoading] =
    useState(false)
  const [knowledgeUploadLoading, setKnowledgeUploadLoading] =
    useState(false)
  const [knowledgeDocumentsError, setKnowledgeDocumentsError] =
    useState("")
  const [knowledgeDocumentsNotice, setKnowledgeDocumentsNotice] =
    useState("")
  const [knowledgeQuery, setKnowledgeQuery] = useState("")
  const [knowledgeSearchLoading, setKnowledgeSearchLoading] =
    useState(false)
  const [knowledgeSearchResults, setKnowledgeSearchResults] =
    useState<KnowledgeSearchResult[] | null>(null)
  const [discoverQuery, setDiscoverQuery] = useState("")
  const [discoverCategory, setDiscoverCategory] =
    useState<(typeof discoverCategories)[number]>("All")

  const [sessions, setSessions] = useState<
    SessionSummary[]
  >([])
  const [savedAnswers, setSavedAnswers] =
    useState<SavedAnswer[]>([])
  const [savedAnswersLoaded, setSavedAnswersLoaded] =
    useState(false)
  const [savedArtifacts, setSavedArtifacts] =
    useState<SavedArtifact[]>([])
  const [savedArtifactsLoaded, setSavedArtifactsLoaded] =
    useState(false)
  const [sessionPreferences, setSessionPreferences] =
    useState<SessionPreferences>({})
  const [sessionPreferencesLoaded, setSessionPreferencesLoaded] =
    useState(false)
  const [renameSessionId, setRenameSessionId] =
    useState<string | null>(null)
  const [renameSessionDraft, setRenameSessionDraft] =
    useState("")

  const [activeThreadId, setActiveThreadId] =
    useState<string | null>(null)
  const [browserEventScreenshots, setBrowserEventScreenshots] =
    useState<Record<string, BrowserEventScreenshot[]>>({})
  const [browserOperation, setBrowserOperation] =
    useState<(BrowserPreviewOperation & { threadId: string; eventId: string }) | null>(null)
  const [browserPreviewOpen, setBrowserPreviewOpen] =
    useState(false)
  const [workspaceFilesOpen, setWorkspaceFilesOpen] =
    useState(false)
  const [artifactsPanelOpen, setArtifactsPanelOpen] =
    useState(true)
  const [artifactsFullscreenOpen, setArtifactsFullscreenOpen] =
    useState(false)
  const artifactsFullscreenTriggerRef = useRef<HTMLButtonElement>(null)
  const artifactsFullscreenCloseRef = useRef<HTMLButtonElement>(null)

  const [sessionsLoading, setSessionsLoading] =
    useState(false)

  const [evidenceSources, setEvidenceSources] =
    useState<EvidenceSource[]>([])
  const [evidenceDetails, setEvidenceDetails] =
    useState<EvidenceResponse | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [mcpEnabled, setMcpEnabled] = useState(false)
  const [mcpServers, setMcpServers] = useState<Record<string, unknown>>({})
  const [mcpConfigDraft, setMcpConfigDraft] =
    useState("{}")
  const [mcpLoadedTools, setMcpLoadedTools] =
    useState(0)
  const [mcpDiagnosticsVersion, setMcpDiagnosticsVersion] =
    useState(0)
  const [settingsLoading, setSettingsLoading] =
    useState(false)
  const [settingsNotice, setSettingsNotice] =
    useState("")
  const [pendingInterrupt, setPendingInterrupt] =
    useState<PendingInterrupt | null>(null)
  const [interruptSubmitting, setInterruptSubmitting] =
    useState(false)
  const interruptSubmitInFlightRef = useRef(false)

  const mcpOptions = mcpEnabled
    ? Object.entries(mcpServers)
        .filter(([id]) => !id.startsWith("__"))
        .map(([id, value]) => {
          const config =
            value !== null && typeof value === "object"
              ? (value as Record<string, unknown>)
              : {}
          const configuredLabel =
            config.display_name ?? config.label ?? config.name
          return {
            id,
            label:
              typeof configuredLabel === "string" &&
              configuredLabel.trim()
                ? configuredLabel.trim()
                : id,
          }
        })
    : []

  const controllerRef = useRef<AbortController | null>(null)
  const pendingAssistantTextIdRef = useRef<string | null>(null)
  const streamTextBatcherRef = useRef<ReturnType<
    typeof createStreamTextBatcher
  > | null>(null)
  if (streamTextBatcherRef.current === null) {
    streamTextBatcherRef.current = createStreamTextBatcher((content) => {
      const assistantId = pendingAssistantTextIdRef.current
      if (!assistantId) {
        return
      }

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: `${message.content}${content}`,
                status: "Processing",
              }
            : message,
        ),
      )
    })
  }
  const browserEventControllerRef =
    useRef<AbortController | null>(null)
  const browserEventCursorRef = useRef(
    new Map<string, number>(),
  )
  const browserOperationTimerRef = useRef<number | null>(null)
  const threadIdRef = useRef<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const imageInputRef = useRef<HTMLInputElement | null>(null)
  const knowledgeDocumentInputRef = useRef<HTMLInputElement | null>(null)
  const mcpButtonRef = useRef<HTMLButtonElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioStreamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const speechRecognitionRef =
    useRef<BrowserSpeechRecognitionLike | null>(null)
  const speechAudioRef = useRef<HTMLAudioElement | null>(null)
  const speechRequestRef = useRef<AbortController | null>(null)
  const activeSpeechMessageRef = useRef<string | null>(null)
  const conversationRef = useRef<HTMLElement | null>(null)
  const stickToBottomRef = useRef(true)
  const focusComposerAfterNavigation = useRef(false)
  const modelSelectionTouchedRef = useRef(false)

  useEffect(() => {
    if (!artifactsFullscreenOpen) {
      return
    }

    const previousOverflow = document.body.style.overflow
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return
      setArtifactsFullscreenOpen(false)
      window.requestAnimationFrame(() =>
        artifactsFullscreenTriggerRef.current?.focus(),
      )
    }

    document.body.style.overflow = "hidden"
    window.addEventListener("keydown", closeOnEscape)
    window.requestAnimationFrame(() =>
      artifactsFullscreenCloseRef.current?.focus(),
    )

    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener("keydown", closeOnEscape)
    }
  }, [artifactsFullscreenOpen])

  useEffect(
    () => () => {
      streamTextBatcherRef.current?.dispose()
      if (browserOperationTimerRef.current !== null) {
        window.clearTimeout(browserOperationTimerRef.current)
      }
    },
    [],
  )

  useEffect(() => {
    if (!mcpMenuOpen) {
      return
    }

    const updateMcpMenuPosition = () => {
      const button = mcpButtonRef.current

      if (!button) {
        return
      }

      const rect = button.getBoundingClientRect()
      const menuWidth = Math.min(220, window.innerWidth - 16)
      const menuHeight = Math.min(190, window.innerHeight - 16)
      const maxLeft = Math.max(8, window.innerWidth - menuWidth - 8)
      const maxTop = Math.max(8, window.innerHeight - menuHeight - 8)
      const rightAlignedLeft = rect.right + 8

      if (rightAlignedLeft + menuWidth <= window.innerWidth - 8) {
        setMcpMenuPosition({
          top: Math.min(Math.max(8, rect.top), maxTop),
          left: rightAlignedLeft,
        })
        return
      }

      const opensBelow = rect.bottom + 8 + menuHeight <= window.innerHeight - 8

      setMcpMenuPosition({
        top: opensBelow
          ? rect.bottom + 8
          : Math.max(8, rect.top - menuHeight - 8),
        left: Math.min(Math.max(8, rect.left), maxLeft),
      })
    }

    updateMcpMenuPosition()
    window.addEventListener("resize", updateMcpMenuPosition)
    window.addEventListener("scroll", updateMcpMenuPosition, true)

    return () => {
      window.removeEventListener("resize", updateMcpMenuPosition)
      window.removeEventListener("scroll", updateMcpMenuPosition, true)
    }
  }, [mcpMenuOpen])

  const deepSearch = chatMode === "deep"
  const agentMode = chatMode === "agent"
  const generatedArtifactCount = messages.reduce(
    (count, message) =>
      count + (message.generatedArtifacts?.length ?? 0),
    0,
  )

  useEffect(() => {
    if (generatedArtifactCount > 0) {
      setArtifactsPanelOpen(true)
    }
  }, [generatedArtifactCount])

  useEffect(() => {
    threadIdRef.current = activeThreadId
  }, [activeThreadId])

  useEffect(() => {
    let cancelled = false

    void getMcpConfig()
      .then((payload) => {
        if (cancelled) {
          return
        }

        const servers = payload.servers ?? {}
        const options = payload.enabled
          ? Object.keys(servers).filter((id) => !id.startsWith("__"))
          : []
        setMcpEnabled(payload.enabled)
        setMcpServers(servers)
        setMcpLoadedTools(payload.loaded_tools)
        setSelectedMcp((current) =>
          options.includes(current) ? current : options[0] ?? "",
        )
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const syncSavedAnswers = () => {
      setSavedAnswers(loadSavedAnswers())
      setSavedAnswersLoaded(true)
    }

    syncSavedAnswers()
    window.addEventListener("storage", syncSavedAnswers)
    return () => window.removeEventListener("storage", syncSavedAnswers)
  }, [])

  useEffect(() => {
    const syncSavedArtifacts = () => {
      setSavedArtifacts(loadSavedArtifacts())
      setSavedArtifactsLoaded(true)
    }

    syncSavedArtifacts()
    window.addEventListener("storage", syncSavedArtifacts)
    return () => window.removeEventListener("storage", syncSavedArtifacts)
  }, [])

  useEffect(() => {
    const syncSessionPreferences = () => {
      setSessionPreferences(loadSessionPreferences())
      setSessionPreferencesLoaded(true)
    }

    syncSessionPreferences()
    window.addEventListener("storage", syncSessionPreferences)
    return () =>
      window.removeEventListener("storage", syncSessionPreferences)
  }, [])

  useEffect(
    () => () => {
      const recorder = mediaRecorderRef.current
      if (recorder) {
        recorder.onstop = null
        recorder.onerror = null
        if (recorder.state !== "inactive") recorder.stop()
      }

      audioStreamRef.current?.getTracks().forEach((track) => track.stop())
      speechRecognitionRef.current?.abort()
      speechRequestRef.current?.abort()
      activeSpeechMessageRef.current = null
      const speechAudio = speechAudioRef.current
      if (speechAudio) {
        speechAudio.pause()
        speechAudio.onended = null
        speechAudio.onerror = null
        speechAudioRef.current = null
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel()
      }
    },
    [],
  )

  function stopAssistantSpeech() {
    speechRequestRef.current?.abort()
    speechRequestRef.current = null

    const audio = speechAudioRef.current
    if (audio) {
      audio.pause()
      audio.currentTime = 0
      audio.onended = null
      audio.onerror = null
      speechAudioRef.current = null
    }

    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel()
    }

    activeSpeechMessageRef.current = null
    setSpeakingMessageId(null)
    setSpeechLoadingMessageId(null)
  }

  async function copyMessage(message: UiMessage) {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopiedMessageId(message.id)
      window.setTimeout(() => {
        setCopiedMessageId((current) =>
          current === message.id ? null : current,
        )
      }, 1800)
    } catch {
      setActivity("复制失败，请检查浏览器剪贴板权限")
    }
  }

  function beginEditingMessage(message: UiMessage) {
    if (running || message.role !== "user") return
    setEditingMessageId(message.id)
    setEditingMessageContent(message.content)
  }

  function cancelEditingMessage() {
    setEditingMessageId(null)
    setEditingMessageContent("")
  }

  async function resubmitEditedMessage(message: UiMessage) {
    const updatedContent = editingMessageContent.trim()
    const messageIndex = messages.findIndex(
      (item) => item.id === message.id,
    )

    if (
      running ||
      !updatedContent ||
      message.role !== "user" ||
      messageIndex < 0
    ) {
      return
    }

    const historyBeforeMessage = messages.slice(0, messageIndex)
    const updatedMessage: UiMessage = {
      ...message,
      content: updatedContent,
      status: undefined,
    }

    cancelEditingMessage()
    browserEventControllerRef.current?.abort()
    setEvidenceSources([])
    setEvidenceDetails(null)
    setPendingInterrupt(null)
    await submitMessage(updatedContent, {
      history: historyBeforeMessage,
      userMessage: updatedMessage,
      forkThread: true,
      preserveComposerDraft: true,
    })
  }

  function saveAssistantMessage(message: UiMessage) {
    if (!savedAnswersLoaded) return

    if (
      savedAnswers.some(
        (answer) => answer.sourceMessageId === message.id,
      )
    ) {
      return
    }

    const answer = createSavedAnswer(
      message.content,
      message.id,
      activeThreadId,
    )
    const nextAnswers = [answer, ...savedAnswers]

    if (!persistSavedAnswers(nextAnswers)) {
      setActivity("保存失败，请检查浏览器本地存储空间")
      return
    }

    setSavedAnswers(nextAnswers)
    setActivity("回答已保存到资料库")
  }

  function saveGeneratedArtifact(artifact: GeneratedArtifact) {
    if (!savedArtifactsLoaded) return

    const sourceKey = savedArtifactSourceKey(
      artifact.id,
      activeThreadId,
    )
    if (
      savedArtifacts.some(
        (item) =>
          savedArtifactSourceKey(
            item.sourceArtifactId,
            item.sessionId,
          ) === sourceKey,
      )
    ) {
      return
    }

    const saved = createSavedArtifact(artifact, activeThreadId)
    const nextArtifacts = [saved, ...savedArtifacts]
    if (!persistSavedArtifacts(nextArtifacts)) {
      setActivity("制品保存失败，请检查浏览器本地存储空间")
      return
    }

    setSavedArtifacts(nextArtifacts)
    setActivity("制品已保存到资料库")
  }

  async function speakAssistantMessage(message: UiMessage) {
    if (
      activeSpeechMessageRef.current === message.id ||
      speechLoadingMessageId === message.id
    ) {
      stopAssistantSpeech()
      return
    }

    const text = speechTextFromMarkdown(message.content)
    if (!text) return

    stopAssistantSpeech()
    const controller = new AbortController()
    speechRequestRef.current = controller
    activeSpeechMessageRef.current = message.id
    setSpeechLoadingMessageId(message.id)

    const startBrowserSpeech = (fallbackText: string) => {
      if (!("speechSynthesis" in window)) {
        throw new Error("此浏览器不支持语音朗读")
      }

      const utterance = new SpeechSynthesisUtterance(
        fallbackText.slice(0, 500),
      )
      utterance.lang = "zh-CN"
      utterance.onend = () => {
        if (activeSpeechMessageRef.current === message.id) {
          activeSpeechMessageRef.current = null
          setSpeakingMessageId(null)
        }
      }
      utterance.onerror = () => {
        if (activeSpeechMessageRef.current === message.id) {
          activeSpeechMessageRef.current = null
          setSpeakingMessageId(null)
          setActivity("浏览器语音朗读失败")
        }
      }

      speechRequestRef.current = null
      setSpeechLoadingMessageId(null)
      setSpeakingMessageId(message.id)
      window.speechSynthesis.speak(utterance)
    }

    try {
      const response = await fetch(
        `${getApiBaseUrl()}/api/tts/synthesize`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: text.slice(0, 2000),
            voice: "longxiaochun",
          }),
          signal: controller.signal,
        },
      )

      if (!response.ok) {
        if (response.status !== 503) {
          throw new Error(`语音合成失败（${response.status}）`)
        }
        startBrowserSpeech(text)
        return
      }

      const result = (await response.json()) as {
        success?: boolean
        audio?: string
        format?: string
      }

      if (controller.signal.aborted) return

      if (!result.success || !result.audio) {
        startBrowserSpeech(text)
        return
      }

      const format = (result.format || "mp3").toLowerCase()
      const mime = format === "mp3" ? "audio/mpeg" : `audio/${format}`
      const audio = new Audio(`data:${mime};base64,${result.audio}`)
      speechAudioRef.current = audio
      audio.onended = () => {
        if (speechAudioRef.current === audio) {
          speechAudioRef.current = null
          speechRequestRef.current = null
          activeSpeechMessageRef.current = null
          setSpeakingMessageId(null)
        }
      }
      audio.onerror = () => {
        if (speechAudioRef.current === audio) {
          speechAudioRef.current = null
          speechRequestRef.current = null
          activeSpeechMessageRef.current = null
          setSpeakingMessageId(null)
          setActivity("语音播放失败")
        }
      }

      await audio.play()
      if (speechRequestRef.current === controller) {
        setSpeechLoadingMessageId(null)
        setSpeakingMessageId(message.id)
      }
    } catch (error) {
      if (controller.signal.aborted) return

      const audio = speechAudioRef.current
      if (audio) {
        audio.pause()
        audio.onended = null
        audio.onerror = null
        speechAudioRef.current = null
      }

      try {
        startBrowserSpeech(text)
      } catch (fallbackError) {
        activeSpeechMessageRef.current = null
        setSpeechLoadingMessageId(null)
        setActivity(
          fallbackError instanceof Error
            ? fallbackError.message
            : error instanceof Error
              ? error.message
              : "无法播放语音",
        )
      }
    }
  }

  useEffect(() => {
    function resizeTextarea() {
      const textarea = textareaRef.current

      if (!textarea) {
        return
      }

      textarea.style.height = "auto"

      const maxHeight = Math.min(
        240,
        window.innerHeight * 0.35,
      )
      const contentHeight = textarea.scrollHeight

      textarea.style.height = `${Math.min(
        contentHeight,
        maxHeight,
      )}px`
      textarea.style.overflowY =
        contentHeight > maxHeight ? "auto" : "hidden"
    }

    resizeTextarea()
    window.addEventListener("resize", resizeTextarea)

    return () => {
      window.removeEventListener("resize", resizeTextarea)
    }
  }, [draft])

  useEffect(() => {
    const conversation = conversationRef.current

    if (
      !conversation ||
      !stickToBottomRef.current
    ) {
      return
    }

    const frame = window.requestAnimationFrame(() => {
      conversation.scrollTo({
        top: conversation.scrollHeight,
        behavior: "auto",
      })
    })

    return () => {
      window.cancelAnimationFrame(frame)
    }
  }, [messages, running])

  useEffect(() => {
    if (
      activeNav === "Dashboard" &&
      focusComposerAfterNavigation.current
    ) {
      textareaRef.current?.focus()
      focusComposerAfterNavigation.current = false
    }
  }, [activeNav])

  async function refreshSessions() {
    setSessionsLoading(true)

    try {
      const items = await listSessions()

      setSessions(items)
    } catch {
      setSessions([])
    } finally {
      setSessionsLoading(false)
    }
  }

  useEffect(() => {
    void refreshSessions()
  }, [])

  async function refreshKnowledgeDocuments() {
    setKnowledgeDocumentsLoading(true)
    setKnowledgeDocumentsError("")

    try {
      setKnowledgeDocuments(await listKnowledgeDocuments())
    } catch (error) {
      setKnowledgeDocuments([])
      setKnowledgeDocumentsError(
        error instanceof Error ? error.message : "文档列表读取失败。",
      )
    } finally {
      setKnowledgeDocumentsLoading(false)
    }
  }

  useEffect(() => {
    if (activeNav === "Library" && libraryKindFilter === "documents") {
      void refreshKnowledgeDocuments()
    }
  }, [activeNav, libraryKindFilter])

  async function handleKnowledgeDocumentUpload(files: FileList | null) {
    const selectedFiles = Array.from(files ?? [])
    if (selectedFiles.length === 0) {
      return
    }

    setKnowledgeUploadLoading(true)
    setKnowledgeDocumentsError("")
    setKnowledgeDocumentsNotice("")
    let uploadedCount = 0
    let uploadFailure = ""

    try {
      for (const file of selectedFiles) {
        await uploadKnowledgeDocument(file)
        uploadedCount += 1
      }
      setKnowledgeDocumentsNotice(`已添加 ${uploadedCount} 个文档。`)
    } catch (error) {
      const reason =
        error instanceof Error ? error.message : "文档上传失败。"
      uploadFailure =
        uploadedCount > 0
          ? `已添加 ${uploadedCount} 个文档，其余上传失败：${reason}`
          : reason
    } finally {
      await refreshKnowledgeDocuments()
      if (uploadFailure) {
        setKnowledgeDocumentsError(uploadFailure)
      }
      setKnowledgeUploadLoading(false)
    }
  }

  async function removeKnowledgeDocument(document: KnowledgeDocument) {
    if (
      !window.confirm(`确定从知识库删除“${document.filename}”吗？`)
    ) {
      return
    }

    setKnowledgeDocumentsError("")
    setKnowledgeDocumentsNotice("")
    try {
      await deleteKnowledgeDocument(document.source)
      setKnowledgeDocumentsNotice(`已删除“${document.filename}”。`)
      setKnowledgeDocuments((current) =>
        current.filter((item) => item.source !== document.source),
      )
      setKnowledgeSearchResults(null)
    } catch (error) {
      setKnowledgeDocumentsError(
        error instanceof Error ? error.message : "文档删除失败。",
      )
    }
  }

  async function runKnowledgeSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const query = knowledgeQuery.trim()
    if (!query) {
      setKnowledgeSearchResults(null)
      return
    }

    setKnowledgeSearchLoading(true)
    setKnowledgeDocumentsError("")
    try {
      setKnowledgeSearchResults(
        await searchKnowledgeDocuments(query, 8),
      )
    } catch (error) {
      setKnowledgeSearchResults(null)
      setKnowledgeDocumentsError(
        error instanceof Error ? error.message : "文档搜索失败。",
      )
    } finally {
      setKnowledgeSearchLoading(false)
    }
  }

  useEffect(() => {
    let cancelled = false

    void getPublicModelConfig()
      .then((config) => {
        if (cancelled) {
          return
        }

        setModelOptions(config.options)
        setDefaultModel(config.default)
        setModelConfigStatus("")
        setModel((current) => {
          let savedModel = ""

          if (!modelSelectionTouchedRef.current) {
            try {
              savedModel =
                window.localStorage.getItem(selectedModelStorageKey) ?? ""
            } catch {
              savedModel = ""
            }
          }

          const preferred = modelSelectionTouchedRef.current
            ? current
            : savedModel
          return config.options.includes(preferred)
            ? preferred
            : config.default
        })
      })
      .catch(() => {
        if (!cancelled) {
          setModelConfigStatus(
            "无法读取后端模型配置，当前显示备用模型。",
          )
          setModel((current) => {
            if (modelSelectionTouchedRef.current) {
              return current
            }

            try {
              const savedModel =
                window.localStorage.getItem(selectedModelStorageKey)
              return savedModel &&
                fallbackModelOptions.includes(savedModel)
                ? savedModel
                : current
            } catch {
              return current
            }
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  function chooseModel(nextModel: string) {
    modelSelectionTouchedRef.current = true
    setModel(nextModel)
    setModelMenuOpen(false)

    try {
      window.localStorage.setItem(
        selectedModelStorageKey,
        nextModel,
      )
    } catch {
      // Model selection still works when browser storage is unavailable.
    }
  }

  useEffect(() => {
    if (!settingsOpen) {
      return
    }

    let cancelled = false
    setSettingsLoading(true)
    setSettingsNotice("")

    void getMcpConfig()
      .then((payload) => {
        if (cancelled) {
          return
        }

        const servers = payload.servers ?? {}
        const options = payload.enabled
          ? Object.keys(servers).filter((id) => !id.startsWith("__"))
          : []
        setMcpEnabled(payload.enabled)
        setMcpServers(servers)
        setMcpLoadedTools(payload.loaded_tools)
        setSelectedMcp((current) =>
          options.includes(current) ? current : options[0] ?? "",
        )
        setMcpConfigDraft(
          JSON.stringify(payload.servers ?? {}, null, 2),
        )
      })
      .catch(() => {
        if (!cancelled) {
          setSettingsNotice("无法读取 MCP 配置，请确认后端已启动。")
        }
      })
      .finally(() => {
        if (!cancelled) {
          setSettingsLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [settingsOpen])

  async function saveMcpSettings() {
    let servers: unknown

    try {
      servers = JSON.parse(mcpConfigDraft)
    } catch {
      setSettingsNotice("服务器配置不是有效的 JSON。")
      return
    }

    if (
      servers === null ||
      typeof servers !== "object" ||
      Array.isArray(servers)
    ) {
      setSettingsNotice("服务器配置必须是 JSON 对象。")
      return
    }

    setSettingsLoading(true)
    setSettingsNotice("")

    try {
      const payload = await updateMcpConfig(
        mcpEnabled,
        servers as Record<string, unknown>,
      )
      const savedServers = payload.servers ?? {}
      const options = payload.enabled
        ? Object.keys(savedServers).filter((id) => !id.startsWith("__"))
        : []
      setMcpEnabled(payload.enabled)
      setMcpServers(savedServers)
      setMcpLoadedTools(payload.loaded_tools)
      setSelectedMcp((current) =>
        options.includes(current) ? current : options[0] ?? "",
      )
      setMcpConfigDraft(
        JSON.stringify(payload.servers ?? {}, null, 2),
      )
      setMcpDiagnosticsVersion((version) => version + 1)
      setSettingsNotice("MCP 配置已保存。")
    } catch {
      setSettingsNotice("MCP 配置保存失败，请稍后重试。")
    } finally {
      setSettingsLoading(false)
    }
  }

  function handleBrowserEvent(
    event: RuntimeEvent,
    assistantId: string,
    threadId: string,
  ) {
    const operationUpdate = getBrowserOperationUpdate(
      event.type,
      event.data,
    )

    if (operationUpdate) {
      if (browserOperationTimerRef.current !== null) {
        window.clearTimeout(browserOperationTimerRef.current)
        browserOperationTimerRef.current = null
      }

      if (operationUpdate.kind === "running") {
        const eventId = `${threadId}:${event.data.seq ?? Date.now()}`
        setBrowserOperation({
          threadId,
          eventId,
          status: "running",
          label: operationUpdate.label,
          progress: operationUpdate.progress,
        })
      } else if (operationUpdate.kind === "error") {
        const eventId = `${threadId}:${event.data.seq ?? Date.now()}`
        const message = /[\u3400-\u9fff]/.test(operationUpdate.message)
          ? `浏览器操作失败：${operationUpdate.message}`
          : "浏览器操作失败，详细错误见聊天消息。"
        setBrowserOperation({
          threadId,
          eventId,
          status: "error",
          label: operationUpdate.label,
          message,
        })
        browserOperationTimerRef.current = window.setTimeout(() => {
          setBrowserOperation((current) =>
            current?.eventId === eventId ? null : current,
          )
          browserOperationTimerRef.current = null
        }, 1800)
      } else {
        setBrowserOperation((current) =>
          current?.threadId === threadId ? null : current,
        )
      }
    }

    if (
      event.type === "search" &&
      event.data.mode === "tree" &&
      typeof event.data.node_id === "string"
    ) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                researchTree: mergeResearchNodeUpdate(
                  message.researchTree,
                  {
                    ...event.data,
                    finding_count: event.data.result_count ?? event.data.count,
                  },
                  "running",
                ),
              }
            : message,
        ),
      )
      const topic = String(event.data.topic ?? "").trim()
      const query = String(event.data.query ?? "").trim()
      setActivity(
        topic && query
          ? `研究分支「${topic}」正在检索：${query}`
          : "研究分支正在检索…",
      )
      return
    }

    if (
      event.type === "research_node_start" ||
      event.type === "research_node_complete"
    ) {
      const status = event.type === "research_node_start"
        ? "running"
        : undefined
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                researchTree: mergeResearchNodeUpdate(
                  message.researchTree,
                  event.data,
                  status,
                ),
              }
            : message,
        ),
      )
      const topic = String(event.data.topic ?? "").trim()
      setActivity(
        event.type === "research_node_start"
          ? topic
            ? `正在研究：${topic}`
            : "正在展开研究分支…"
          : topic
            ? `研究分支已完成：${topic}`
            : "研究分支已更新",
      )
      return
    }

    if (event.type === "research_tree_update") {
      const tree = normalizeResearchTree(
        event.data.tree,
        event.data.quality,
      )
      if (tree) {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? { ...message, researchTree: tree }
              : message,
          ),
        )
      }
      return
    }

    if (event.type === "quality_update") {
      const budget = normalizeResearchBudget(event.data)
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                researchBudget: mergeResearchBudget(
                  message.researchBudget,
                  budget,
                ),
                researchTree: mergeResearchTreeBudget(
                  mergeResearchTreeQuality(message.researchTree, event.data),
                  event.data,
                ),
              }
            : message,
        ),
      )
      return
    }

    if (event.type === "task_update") {
      const taskBoard = normalizeTaskBoard(event.data)
      if (taskBoard) {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? { ...message, taskBoard }
              : message,
          ),
        )
        setActivity("任务清单已更新")
      }
      return
    }

    const tool = String(
      event.data.tool ??
        event.data.tool_name ??
        "browser",
    ).trim()
    const isBrowserEvent =
      event.type === "tool_progress" ||
      event.type === "tool_screenshot"
    const action = String(
      event.data.action ?? event.type,
    ).trim()
    const url = String(
      event.data.page_url ?? event.data.url ?? "",
    ).trim()
    const title = String(
      event.data.title ?? "",
    ).trim()
    const screenshotUrl = String(
      event.data.screenshot_url ?? "",
    ).trim()
    if (event.type === "tool_screenshot") {
      const rawScreenshotUrl = String(
        event.data.screenshot_url ?? event.data.url ?? "",
      ).trim()
      const imageData = String(event.data.image ?? "").trim()
      const sourceTool = String(
        event.data.tool ?? event.data.tool_name ?? "",
      ).toLowerCase()
      const pageUrl = String(event.data.page_url ?? "").trim()
      const isBrowserScreenshot =
        sourceTool.includes("browser") ||
        /^https?:\/\//i.test(pageUrl)
      const isImageUrl =
        rawScreenshotUrl.startsWith("data:image/") ||
        rawScreenshotUrl.includes("/api/screenshots/") ||
        /\.(png|jpe?g|webp)(?:[?#]|$)/i.test(rawScreenshotUrl)

      if (isBrowserScreenshot && (imageData || isImageUrl)) {
        const screenshotUrlForView = imageData
          ? `data:${String(event.data.mime_type ?? "image/jpeg")};base64,${imageData}`
          : rawScreenshotUrl.startsWith("/")
            ? `${getApiBaseUrl()}${rawScreenshotUrl}`
            : rawScreenshotUrl
        const rawTimestamp = event.data.timestamp
        const numericTimestamp = Number(rawTimestamp)
        const parsedTimestamp =
          typeof rawTimestamp === "string" && Number.isNaN(numericTimestamp)
            ? Date.parse(rawTimestamp)
            : numericTimestamp > 1_000_000_000_000
              ? numericTimestamp
              : numericTimestamp * 1000
        const screenshot: BrowserEventScreenshot = {
          id: String(event.data.event_id ?? screenshotUrlForView),
          url: screenshotUrlForView,
          action,
          pageUrl: pageUrl || undefined,
          title: title || undefined,
          timestamp: Number.isFinite(parsedTimestamp) && parsedTimestamp > 0
            ? parsedTimestamp
            : Date.now(),
        }

        setBrowserEventScreenshots((current) => {
          const existing = current[threadId] ?? []
          if (existing.some((item) => item.id === screenshot.id || item.url === screenshot.url)) {
            return current
          }
          return {
            [threadId]: [...existing, screenshot].slice(-100),
          }
        })
      }
    }
    const id = String(
      event.data.event_id ??
        `${tool}:${action}:${url}:${screenshotUrl}`,
    )
    const generatedArtifacts =
      event.type === "tool_screenshot"
        ? extractGeneratedArtifacts(event)
        : []

    if (generatedArtifacts.length > 0) {
      setMessages((current) =>
        current.map((message) => {
          if (message.id !== assistantId) {
            return message
          }

          const currentArtifacts =
            message.generatedArtifacts ?? []
          const existingIds = new Set(
            currentArtifacts.map((item) => item.id),
          )

          return {
            ...message,
            generatedArtifacts: [
              ...currentArtifacts,
              ...generatedArtifacts.filter(
                (item) => !existingIds.has(item.id),
              ),
            ].slice(-12),
          }
        }),
      )
    }

    if (!isBrowserEvent) {
      return
    }

    setMessages((current) =>
      current.map((message) => {
        if (message.id !== assistantId) {
          return message
        }

        const activities = message.browserActivities ?? []

        if (activities.some((item) => item.id === id)) {
          return message
        }

        return {
          ...message,
          browserActivities: [
            ...activities,
            {
              id,
              tool,
              action,
              url: url || undefined,
              title: title || undefined,
              screenshotUrl:
                screenshotUrl || undefined,
            },
          ].slice(-12),
        }
      }),
    )

    setActivity(
      title
        ? `浏览器：${title}`
        : `浏览器：${toolDisplayName(tool)}`,
    )
  }

  async function addImageFiles(files: File[]) {
    if (attachmentsLoading || files.length === 0) return

    const remainingSlots = 5 - attachments.length
    const selected = files.slice(0, remainingSlots)

    if (remainingSlots <= 0) {
      setActivity("一条消息最多添加 5 张图片。")
      return
    }

    if (selected.some((file) => !file.type.startsWith("image/"))) {
      setActivity("目前只支持添加图片文件。")
      return
    }

    if (selected.some((file) => file.size > 12 * 1024 * 1024)) {
      setActivity("单张图片不能超过 12 MB。")
      return
    }

    const totalSize = [
      ...attachments.map((image) => image.size),
      ...selected.map((file) => file.size),
    ].reduce((sum, size) => sum + size, 0)

    if (totalSize > 25 * 1024 * 1024) {
      setActivity("单条消息的图片总大小不能超过 25 MB。")
      return
    }

    setAttachmentsLoading(true)
    try {
      const loaded = await Promise.all(
        selected.map(imageFileToAttachment),
      )
      setAttachments((current) => [...current, ...loaded].slice(0, 5))
      setActivity("")
    } catch (error) {
      setActivity(
        error instanceof Error
          ? error.message
          : "图片读取失败，请重试。",
      )
    } finally {
      setAttachmentsLoading(false)
    }
  }

  async function submitMessage(
    messageOverride?: string,
    options?: {
      history?: UiMessage[]
      userMessage?: UiMessage
      forkThread?: boolean
      preserveComposerDraft?: boolean
    },
  ) {
    if (chatMode === "mcp" && !selectedMcp) {
      setActivity("尚无已启用的 MCP 服务器，请先在设置中配置并启用。")
      setMcpMenuOpen(true)
      return
    }

    const requestedContent = (messageOverride ?? draft).trim()
    const pendingAttachments = options?.userMessage
      ? options.userMessage.attachments ?? []
      : attachments
    const content =
      requestedContent ||
      (pendingAttachments.length > 0 ? "请分析我上传的图片。" : "")

    if (
      (!content && pendingAttachments.length === 0) ||
      running ||
      (attachmentsLoading && !options?.preserveComposerDraft)
    ) {
      return
    }

    setEvidenceSources([])
    setEvidenceDetails(null)
    setPendingInterrupt(null)

    let currentThreadId = options?.forkThread
      ? null
      : activeThreadId

    const userMessage: UiMessage = options?.userMessage
      ? { ...options.userMessage, content }
      : {
          id: crypto.randomUUID(),
          role: "user",
          content,
          attachments: pendingAttachments,
        }

    const assistantId = crypto.randomUUID()

    stickToBottomRef.current = true

    const assistantMessage: UiMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "Processing",
    }

    streamTextBatcherRef.current?.flush()
    pendingAssistantTextIdRef.current = assistantId

    const nextMessages = [
      ...(options?.history ?? messages),
      userMessage,
    ]

    if (options?.forkThread) {
      threadIdRef.current = null
      setActiveThreadId(null)
    }

    setMessages([
      ...nextMessages,
      assistantMessage,
    ])
    if (!options?.preserveComposerDraft) {
      setDraft("")
      setAttachments([])
    }
    setRunning(true)
    setActivity("正在连接研究服务…")

    const controller = new AbortController()
    controllerRef.current = controller

    try {
      for await (
        const event of streamChat(
          nextMessages.map(
            ({ role, content: messageContent }) => ({
              role,
              content: messageContent,
            }),
          ),
          chatMode,
          controller.signal,
          currentThreadId ?? undefined,
          (threadId) => {
            if (threadIdRef.current !== threadId) {
              if (browserOperationTimerRef.current !== null) {
                window.clearTimeout(browserOperationTimerRef.current)
                browserOperationTimerRef.current = null
              }
              setBrowserOperation(null)
            }
            currentThreadId = threadId
            threadIdRef.current = threadId
            setActiveThreadId(threadId)

            browserEventControllerRef.current?.abort()
            const browserController = new AbortController()
            browserEventControllerRef.current = browserController
            const lastEventId =
              browserEventCursorRef.current.get(threadId)

            void streamThreadEvents(
              threadId,
              browserController.signal,
              (event) => {
                const sequence = Number(
                  event.data.seq,
                )

                if (
                  Number.isInteger(sequence) &&
                  sequence > 0
                ) {
                  browserEventCursorRef.current.set(
                    threadId,
                    sequence,
                  )
                }

                handleBrowserEvent(event, assistantId, threadId)
              },
              lastEventId,
            ).catch((error) => {
              if (!isAbortError(error)) {
                setActivity("浏览器活动连接失败")
              }
            })
          },
          model,
          pendingAttachments.map(({ name, mime, data }) => ({
            name,
            mime,
            data,
          })),
          chatMode === "mcp" ? selectedMcp : undefined,
        )
      ) {
        handleRuntimeEvent(event, assistantId)
      }
    } catch (error) {
      streamTextBatcherRef.current?.flush()
      if (isAbortError(error)) {
        setActivity("研究已停止")

        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content:
                    message.content ||
                    "研究已取消。",
                  status: "Cancelled",
                }
              : message,
          ),
        )
      } else {
        setActivity(
          error instanceof Error
            ? userFacingErrorMessage(error.message)
            : "请求失败，请稍后重试。",
        )

        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  status: "Failed",
                }
              : message,
          ),
        )
      }
    } finally {
      streamTextBatcherRef.current?.flush()
      setRunning(false)
      controllerRef.current = null
      browserEventControllerRef.current?.abort()
      browserEventControllerRef.current = null

      const completedThreadId =
        currentThreadId ?? threadIdRef.current ?? activeThreadId
      if (completedThreadId) {
        void loadEvidence(completedThreadId)
        void (async () => {
          for (let attempt = 0; attempt < 3; attempt += 1) {
            if (attempt > 0) {
              await new Promise((resolve) =>
                window.setTimeout(resolve, attempt * 250),
              )
            }
            const snapshot = await getSessionState(completedThreadId)
            const research = normalizeResearchSessionSnapshot(snapshot)
            const terminal = ["completed", "failed", "cancelled"].includes(
              snapshot.status.toLowerCase(),
            )
            const treeSettled =
              !research.tree ||
              (research.tree.active === 0 && research.tree.pending === 0)
            if ((!research.tree && !research.budget) || !terminal || !treeSettled) {
              if (attempt < 2) continue
            }
            if (!research.tree && !research.budget) return

            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      ...(research.tree
                        ? { researchTree: research.tree }
                        : {}),
                      ...(research.budget
                        ? { researchBudget: research.budget }
                        : {}),
                    }
                  : message,
              ),
            )
            return
          }
        })().catch(() => undefined)
      }

      void refreshSessions()
    }
  }

  function handleRuntimeEvent(
    event: RuntimeEvent,
    assistantId: string,
  ) {
    setPendingInterrupt((current) =>
      pendingInterruptAfterRuntimeEvent(event.type, current),
    )

    if (
      event.type === "completion" ||
      event.type === "done" ||
      event.type === "cancelled" ||
      event.type === "error"
    ) {
      streamTextBatcherRef.current?.flush()
    }

    const content =
      typeof event.data.content === "string"
        ? event.data.content
        : contentToText(event.data.value)

    const text =
      typeof event.data.text === "string"
        ? event.data.text
        : ""

    const updateAssistant = (
      updater: (
        message: UiMessage,
      ) => UiMessage,
    ) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? updater(message)
            : message,
        ),
      )
    }

    const appendGeneratedArtifacts = () => {
      const generatedArtifacts = extractGeneratedArtifacts(
        event,
      )

      if (generatedArtifacts.length === 0) {
        return
      }

      updateAssistant((message) => {
        const currentArtifacts =
          message.generatedArtifacts ?? []
        const existingIds = new Set(
          currentArtifacts.map((item) => item.id),
        )

        return {
          ...message,
          generatedArtifacts: [
            ...currentArtifacts,
            ...generatedArtifacts.filter(
              (item) => !existingIds.has(item.id),
            ),
          ].slice(-12),
        }
      })
    }

    const appendAssistantStep = (step: string) => {
      const normalizedStep = step.trim()

      if (!normalizedStep) {
        return
      }

      updateAssistant((message) => ({
        ...message,
        steps: message.steps?.includes(normalizedStep)
          ? message.steps
          : [
              ...(message.steps ?? []),
              normalizedStep,
            ],
      }))
    }

    const updateToolCall = (
      name: string,
      status: ToolCall["status"],
      detail?: string,
      metadata: ToolCallMetadata = {},
    ) => {
      const normalizedName = name.trim() || "tool"
      const normalizedDetail = detail?.trim()
      const callKey =
        metadata.callCount !== undefined
          ? `${normalizedName}:${metadata.callCount}`
          : undefined

      updateAssistant((message) => {
        const toolCalls = [
          ...(message.toolCalls ?? []),
        ]
        let targetIndex = -1

        if (callKey) {
          targetIndex = toolCalls.findIndex(
            (toolCall) => toolCall.id === callKey,
          )

          if (targetIndex < 0) {
            for (
              let index = toolCalls.length - 1;
              index >= 0;
              index -= 1
            ) {
              if (
                toolCalls[index].name === normalizedName &&
                toolCalls[index].callCount === undefined
              ) {
                targetIndex = index
                break
              }
            }
          }
        }

        if (targetIndex < 0 && !callKey) {
          for (
            let index = toolCalls.length - 1;
            index >= 0;
            index -= 1
          ) {
            if (toolCalls[index].name === normalizedName) {
              if (
                status === "running" &&
                toolCalls[index].status !== "running"
              ) {
                continue
              }

              targetIndex = index
              break
            }
          }
        }

        if (targetIndex < 0) {
          toolCalls.push({
            id: callKey ?? `${normalizedName}-${toolCalls.length}`,
            name: normalizedName,
            status,
            detail: normalizedDetail,
            callCount: metadata.callCount,
            attempt: metadata.attempt,
          })
        } else {
          const current = toolCalls[targetIndex]
          const statusPriority = {
            running: 1,
            completed: 2,
            failed: 3,
          }
          const mergedStatus =
            statusPriority[status] >=
            statusPriority[current.status]
              ? status
              : current.status

          toolCalls[targetIndex] = {
            ...current,
            id: callKey ?? current.id,
            status: mergedStatus,
            detail:
              normalizedDetail ||
              current.detail,
            callCount:
              metadata.callCount ??
              current.callCount,
            attempt:
              metadata.attempt ??
              current.attempt,
          }
        }

        return {
          ...message,
          toolCalls,
        }
      })
    }

    if (
      event.type === "interrupt" ||
      event.type === "tool_approval_required"
    ) {
      const interrupt = normalizePendingInterrupt(
        event.data,
        assistantId,
      )

      if (interrupt) {
        setPendingInterrupt(interrupt)
        updateAssistant((message) => ({
          ...message,
          status: "Paused",
        }))
        const waitingLabel = interruptWaitMessage(interrupt.kind)
        setActivity(waitingLabel)
        appendAssistantStep(waitingLabel)
      }
      return
    }

    if (event.type === "status") {
      setActivity(localizeRuntimeStatus(text))
      return
    }

    if (
      event.type === "thinking" ||
      event.type === "planner_output"
    ) {
      setActivity("正在规划研究步骤…")
      appendAssistantStep("正在规划研究步骤…")
      return
    }

    if (
      event.type === "tool_start" ||
      event.type === "tool_started" ||
      event.type === "tool_end" ||
      event.type === "tool_result" ||
      event.type === "tool_error"
    ) {
      const toolName =
        typeof event.data.tool_name === "string"
          ? event.data.tool_name
        : typeof event.data.name === "string"
            ? event.data.name
            : typeof event.data.tool === "string"
              ? event.data.tool
            : "tool"

      const toolMetadata = {
        callCount: normalizeToolSequence(
          event.data.call_count,
        ),
        attempt: normalizeToolSequence(
          event.data.attempt,
        ),
      }

      setActivity(
        event.type === "tool_error"
          ? `工具执行失败：${toolDisplayName(toolName)}`
          : `正在调用工具：${toolDisplayName(toolName)}`,
      )
      const toolDetail = valueToDisplayText(
        event.data.error ??
          event.data.output ??
          event.data.result,
      )
      updateToolCall(
        toolName,
        event.type === "tool_error"
          ? "failed"
          : event.type === "tool_start" ||
              event.type === "tool_started"
            ? "running"
            : "completed",
        toolDetail,
        toolMetadata,
      )
      appendGeneratedArtifacts()
      appendAssistantStep(
        event.type === "tool_error"
          ? `工具执行失败：${toolDisplayName(toolName)}`
          : `调用工具：${toolDisplayName(toolName)}`,
      )
      return
    }

    if (event.type === "search") {
      const query =
        typeof event.data.query === "string"
          ? event.data.query
          : ""

      setActivity(
        query
          ? `正在搜索：${query}`
          : "正在搜索资料…",
      )
      appendAssistantStep(
        query ? `正在搜索：${query}` : "正在搜索资料…",
      )
      return
    }

    if (event.type === "research_plan") {
      const queries = Array.isArray(event.data.queries)
        ? event.data.queries
            .map((query) => String(query).trim())
            .filter(Boolean)
        : []
      const reasoning =
        typeof event.data.reasoning === "string"
          ? event.data.reasoning.trim()
          : ""
      const mode =
        typeof event.data.search_mode === "string"
          ? event.data.search_mode
          : typeof event.data.mode === "string"
            ? event.data.mode
            : ""

      updateAssistant((message) => ({
        ...message,
        researchPlan: {
          queries,
          reasoning: reasoning || undefined,
          mode: mode || undefined,
        },
      }))

      setActivity(
        queries.length > 0
          ? `研究计划已生成：${queries.length} 个查询`
          : "研究计划已生成",
      )
      appendAssistantStep(
        queries.length > 0
          ? `研究计划：${queries.join("；")}`
          : "研究计划已生成",
      )
      return
    }

    if (event.type === "agent_plan") {
      const pendingCalls = Array.isArray(
        event.data.pending_tool_calls,
      )
        ? event.data.pending_tool_calls
        : []
      const firstCall = pendingCalls[0]

      if (
        firstCall !== null &&
        typeof firstCall === "object" &&
        "tool_name" in firstCall
      ) {
        const toolName = String(
          (firstCall as { tool_name: unknown }).tool_name,
        )
        setActivity(`已规划工具调用：${toolDisplayName(toolName)}`)
        updateToolCall(toolName, "running")
        appendAssistantStep(
          `已规划工具调用：${toolDisplayName(toolName)}`,
        )
      } else {
        setActivity("工具调用计划已生成")
        appendAssistantStep("工具调用计划已生成")
      }
      return
    }

    if (event.type === "tool") {
      const toolName =
        typeof event.data.tool_name === "string"
          ? event.data.tool_name
          : typeof event.data.tool === "string"
            ? event.data.tool
          : "tool"
      const eventName = String(
        event.data.event_type ?? "",
      ).toLowerCase()
      const failed =
        eventName.includes("fail") ||
        eventName.includes("error")
      const running =
        eventName.includes("start") ||
        eventName.includes("retry")
      const toolDetail = valueToDisplayText(
        event.data.error ??
          event.data.output ??
          event.data.result,
      )
      const toolMetadata = {
        callCount: normalizeToolSequence(
          event.data.call_count,
        ),
        attempt: normalizeToolSequence(
          event.data.attempt,
        ),
      }

      setActivity(
        failed
          ? `工具执行失败：${toolDisplayName(toolName)}`
          : `正在调用工具：${toolDisplayName(toolName)}`,
      )

      updateToolCall(
        toolName,
        failed
          ? "failed"
          : running
            ? "running"
            : "completed",
        toolDetail,
        toolMetadata,
      )
      appendGeneratedArtifacts()
      appendAssistantStep(
        failed
          ? `工具执行失败：${toolDisplayName(toolName)}`
          : `调用工具：${toolDisplayName(toolName)}`,
      )
      return
    }

    if (event.type === "research_tree_update") {
      const tree = normalizeResearchTree(
        event.data.tree,
        event.data.quality,
      )

      if (tree) {
        updateAssistant((message) => ({
          ...message,
          researchTree: tree,
        }))
      }

      setActivity("研究计划已更新")
      appendAssistantStep("研究计划已更新")
      return
    }

    if (event.type === "quality_update") {
      const budget = normalizeResearchBudget(event.data)
      updateAssistant((message) => ({
        ...message,
        researchBudget: mergeResearchBudget(
          message.researchBudget,
          budget,
        ),
        researchTree: mergeResearchTreeBudget(
          mergeResearchTreeQuality(message.researchTree, event.data),
          event.data,
        ),
        quality: normalizeResearchQuality(event.data) ?? message.quality,
      }))

      setActivity("正在评估证据质量…")
      appendAssistantStep("正在评估证据质量…")
      return
    }

    if (
      event.type === "text" ||
      event.type === "token"
    ) {
      if (
        !content ||
        !isFinalTextEvent(event, content)
      ) {
        return
      }

      pendingAssistantTextIdRef.current = assistantId
      streamTextBatcherRef.current?.append(content)
      return
    }

    if (
      event.type === "message" ||
      event.type === "content" ||
      event.type === "answer"
    ) {
      if (!content || isPlannerPayload(content)) {
        return
      }

      updateAssistant((message) => ({
        ...message,
        content,
      }))
      return
    }

    if (event.type === "sources") {
      const rawSources =
        event.data.sources ??
        event.data.items

      const normalizedSources = normalizeEvidenceSources(
        rawSources,
      )

      setEvidenceSources((current) => {
        const merged = normalizeEvidenceSources([
          ...current,
          ...normalizedSources,
        ])
        return merged
      })
      setEvidenceDetails((current) => {
        if (!current) {
          return null
        }

        return {
          ...current,
          sources: normalizeEvidenceSources([
            ...current.sources,
            ...normalizedSources,
          ]),
        }
      })

      const count = normalizedSources.length

      setActivity(
        count > 0
          ? `已找到 ${count} 个来源`
          : "来源已整理",
      )
      appendAssistantStep(
        count > 0
          ? `已整理 ${count} 个来源`
          : "来源已整理",
      )
      return
    }

    if (event.type === "completion") {
      const finalContent = isPlannerPayload(content)
        ? ""
        : content

      appendGeneratedArtifacts()

      updateAssistant((message) => ({
        ...message,
        content: finalContent || message.content,
        status: "Completed",
      }))

      setActivity("研究已完成")
      appendAssistantStep("已生成最终答复")
      return
    }

    if (event.type === "cancelled") {
      updateAssistant((message) => ({
        ...message,
        content:
          content ||
          "研究已取消。",
        status: "Cancelled",
      }))

      setActivity("研究已取消")
      appendAssistantStep("研究已取消")
      return
    }

    if (event.type === "done") {
      updateAssistant((message) => ({
        ...message,
        status:
          message.status === "Failed" ||
          message.status === "Cancelled"
            ? message.status
            : "Completed",
      }))

      setActivity("研究已完成")
      return
    }

    if (event.type === "error") {
      const rawErrorMessage =
        typeof event.data.message === "string"
          ? event.data.message
          : "研究失败，请稍后重试。"
      const errorMessage = userFacingErrorMessage(
        rawErrorMessage,
      )

      updateAssistant((message) => ({
        ...message,
        content:
          message.content || errorMessage,
        status: "Failed",
      }))

      setActivity(errorMessage)
      appendAssistantStep(`执行失败：${errorMessage}`)
    }
  }

  async function respondToInterrupt(
    decision: "approve" | "reject" | "edit",
  ) {
    const pending = pendingInterrupt
    const threadId = threadIdRef.current ?? activeThreadId

    if (
      !pending ||
      !threadId ||
      interruptSubmitting ||
      interruptSubmitInFlightRef.current
    ) {
      return
    }

    let resumeValue: InterruptResumeDecision
    if (decision === "edit") {
      if (
        pending.kind === "human_review" ||
        pending.kind === "draft_review"
      ) {
        if (!pending.contentText.trim()) {
          setActivity("审核内容不能为空。")
          return
        }
        resumeValue = { content: pending.contentText }
      } else if (pending.kind === "sources_review") {
        resumeValue = { content: pending.contentText }
      } else if (pending.kind === "plan_review") {
        const queries = pending.contentText
          .split(/\r?\n/)
          .map((line) => line.replace(/^\s*(?:(?:[-*+])|(?:\d+[.)]))\s*/, "").trim())
          .filter(Boolean)
        if (queries.length === 0) {
          setActivity("研究计划至少需要一条查询。")
          return
        }
        resumeValue = { type: "edit_plan", queries }
      } else {
        const toolName = pending.toolName.trim()
        let parsedJson: unknown
        try {
          parsedJson = JSON.parse(pending.argumentsText) as unknown
        } catch {
          setActivity("工具参数不是有效的 JSON。")
          return
        }
        const parsedArguments = asRecord(parsedJson)
        if (!toolName || !parsedArguments) {
          setActivity("工具名称不能为空，参数必须是 JSON 对象。")
          return
        }
        resumeValue = {
          type: "edit",
          edited_action: {
            name: toolName,
            args: parsedArguments,
          },
        }
      }
    } else {
      resumeValue = decision === "approve"
    }

    const subject = interruptSubject(pending.kind)
    interruptSubmitInFlightRef.current = true
    setInterruptSubmitting(true)
    setRunning(true)
    setActivity(
      decision === "approve"
        ? `正在批准${subject}并恢复研究…`
        : decision === "edit"
          ? pending.kind === "human_review"
            ? "正在提交报告修改…"
            : pending.kind === "draft_review"
              ? "正在提交初稿修改…"
              : pending.kind === "sources_review"
                ? "正在提交补充要求…"
                : pending.kind === "plan_review"
                  ? "正在提交修改后的研究计划…"
                  : "正在验证修改并执行工具…"
          : `正在拒绝${subject}…`,
    )

    try {
      const response = await resumeInterrupt(threadId, resumeValue)
      const nextInterrupt = normalizePendingInterrupt(
        response.result ?? (response.status === "interrupted" ? response : null),
        pending.assistantId,
      )

      if (nextInterrupt) {
        setPendingInterrupt(nextInterrupt)
        setActivity(interruptWaitMessage(nextInterrupt.kind))
        return
      }

      const result = asRecord(response.result) ?? asRecord(response)
      const nestedResult = asRecord(result?.result)
      const executionStatus = String(
        response.execution_status ??
          result?.status ??
          nestedResult?.status ??
          "",
      ).toLowerCase()
      const executionFailed = executionStatus === "failed"
      const executionCancelled =
        decision === "reject" || executionStatus === "cancelled"
      const resumedContent = [
        result?.final_report,
        result?.content,
        result?.answer,
        nestedResult?.final_report,
        nestedResult?.content,
      ].find(
        (value): value is string =>
          typeof value === "string" && value.trim().length > 0,
      )

      setMessages((current) =>
        current.map((message) =>
          message.id === pending.assistantId
            ? {
                ...message,
                content:
                  resumedContent?.trim() ||
                  message.content ||
                  (executionFailed
                    ? "工具执行失败，请检查参数和权限。"
                    : executionCancelled
                    ? interruptRejectedMessage(pending.kind)
                    : decision === "edit"
                      ? interruptEditedMessage(pending.kind)
                      : interruptApprovedMessage(pending.kind)),
                status: executionFailed
                  ? "Failed"
                  : executionCancelled
                    ? "Cancelled"
                    : "Completed",
              }
            : message,
        ),
      )
      setPendingInterrupt(null)
      setActivity(
        executionFailed
          ? `工具执行失败：${resumedContent?.trim() || "请检查参数和权限。"}`
          : executionCancelled
          ? `已拒绝${subject}`
          : decision === "edit"
            ? interruptEditedMessage(pending.kind)
            : interruptApprovedMessage(pending.kind),
      )
    } catch (error) {
      setActivity(
        error instanceof Error
          ? `审批操作失败：${error.message}`
          : "审批操作失败，请稍后重试。",
      )
    } finally {
      interruptSubmitInFlightRef.current = false
      setInterruptSubmitting(false)
      setRunning(false)
    }
  }

  function stopMessage() {
    const threadId = threadIdRef.current

    if (threadId) {
      void cancelChat(threadId).catch(() => {
        setActivity("无法取消后端任务")
      })
    }

    controllerRef.current?.abort()
    browserEventControllerRef.current?.abort()
    setActivity("正在停止研究…")
  }

  function resetWorkspace() {
    const threadId = threadIdRef.current

    if (threadId) {
      void cancelChat(threadId).catch(() => {
        // 新建会话不应被取消失败阻塞
      })
    }

    controllerRef.current?.abort()
    browserEventControllerRef.current?.abort()
    browserEventControllerRef.current = null
    threadIdRef.current = null
    browserEventCursorRef.current.clear()
    if (browserOperationTimerRef.current !== null) {
      window.clearTimeout(browserOperationTimerRef.current)
      browserOperationTimerRef.current = null
    }
    setBrowserOperation(null)
    stickToBottomRef.current = true

    setMessages([])
    setDraft("")
    setAttachments([])
    setIsDraggingImages(false)
    setActivity("")
    setRunning(false)
    setActiveThreadId(null)
    setBrowserEventScreenshots({})
    setBrowserPreviewOpen(false)
    setWorkspaceFilesOpen(false)
    setEvidenceSources([])
    setEvidenceDetails(null)
    setPendingInterrupt(null)
    setActiveNav("Dashboard")
  }

  async function downloadReport(
    format: "markdown" | "html" | "json" | "pdf" | "docx",
  ) {
    const threadId = activeThreadId

    if (!threadId || running) {
      return
    }

    try {
      const blob = await requestReportExport(
        threadId,
        format,
        messages,
      )
      const objectUrl = URL.createObjectURL(blob)
      const link = document.createElement("a")
      const extension =
        format === "markdown" ? "md" : format

      link.href = objectUrl
      link.download = `PuzzleResearch-report.${extension}`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(
        () => URL.revokeObjectURL(objectUrl),
        60_000,
      )
    } catch (error) {
      setActivity(
        error instanceof Error
          ? error.message
          : "研究报告导出失败。",
      )
    }
  }

  async function loadEvidence(
    threadId: string,
  ) {
    try {
      const payload =
        await getSessionEvidence(threadId)

      setEvidenceDetails(payload)
      setEvidenceSources(payload.sources)
    } catch {
      setEvidenceDetails(null)
      setEvidenceSources([])
    }
  }

  async function openSession(
    session: SessionSummary,
  ) {
    if (running) {
      return
    }

    try {
      const payload = await getSessionState(
        session.thread_id,
      )

      const archivedConversation =
        payload.state.conversation_history
      const restored = restoreSessionMessages(
        Array.isArray(archivedConversation) &&
          archivedConversation.length > 0
          ? archivedConversation
          : payload.state.messages,
      )
      const artifacts =
        asRecord(payload.state.deepsearch_artifacts) ??
        asRecord(payload.deepsearch_artifacts) ??
        {}
      const restoredTree = normalizeResearchTree(
        payload.state.research_tree ?? artifacts.research_tree,
        artifacts,
      )
      const restoredBudget =
        normalizeResearchBudget(artifacts) ??
        normalizeResearchBudget(payload.deepsearch_artifacts) ??
        normalizeResearchBudget(payload.state)
      const restoredArtifacts = Array.isArray(artifacts.generated_artifacts)
        ? artifacts.generated_artifacts
            .map((item, index) =>
              normalizeGeneratedArtifact(item, `restored-chart:${index}`),
            )
            .filter((item): item is GeneratedArtifact => Boolean(item))
        : []
      const restoredAssistantId = [...restored]
        .reverse()
        .find((message) => message.role === "assistant")?.id
      const restoredWithResearch = restored.map((message) => {
        if (message.id !== restoredAssistantId) return message
        return {
          ...message,
          ...(restoredTree ? { researchTree: restoredTree } : {}),
          ...(restoredBudget ? { researchBudget: restoredBudget } : {}),
          ...(restoredArtifacts.length > 0
            ? { generatedArtifacts: restoredArtifacts }
            : {}),
        }
      })

      setMessages(restoredWithResearch)
      setAttachments([])

      const restoredAssistant = [...restoredWithResearch]
        .reverse()
        .find((message) => message.role === "assistant")
      setPendingInterrupt(
        restoredAssistant
          ? normalizePendingInterrupt(
              { interrupts: payload.interrupts },
              restoredAssistant.id,
            )
          : null,
      )

      threadIdRef.current = session.thread_id
      setActiveThreadId(session.thread_id)
      setActiveNav("Dashboard")
      setActivity("")
      setDraft("")
      setEvidenceSources([])
      setEvidenceDetails(null)
      void loadEvidence(session.thread_id)
    } catch (error) {
      setActivity(
          error instanceof Error
            ? userFacingErrorMessage(error.message)
          : "无法加载会话。",
      )
    }
  }

  function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault()
      void submitMessage()
    }
  }

  function chooseSuggestion(prompt: string) {
    setDraft(prompt)
    textareaRef.current?.focus()
  }

  function selectPromptCommand(command: string) {
    setCommandMenuOpen(false)

    if (command === "clear") {
      resetWorkspace()
      return
    }

    const modeCommands: Record<string, ChatMode> = {
      direct: "direct",
      web: "web",
      agent: "agent",
      deep: "deep",
    }

    if (modeCommands[command]) {
      const nextMode = modeCommands[command]
      setChatMode(nextMode)
      setDraft("")
      setActivity(
        nextMode === "direct"
          ? "已切换为普通对话模式"
          : `${
              nextMode === "web"
                ? "Web 搜索模式"
                : nextMode === "agent"
                  ? "智能体模式"
                  : "深度研究模式"
            }已开启`,
      )
      textareaRef.current?.focus()
      return
    }

    const promptByCommand: Record<string, string> = {
      fix: "请帮我定位并修复以下代码：\n\n",
      explain: "请解释以下概念或代码，并给出一个简短示例：\n\n",
      test: "请为以下代码编写可靠的测试用例：\n\n",
    }
    const prompt = promptByCommand[command]

    if (prompt) {
      setDraft(prompt)
      textareaRef.current?.focus()
    }
  }

  function appendRecognizedText(text: string) {
    const normalized = text.trim()
    if (!normalized) return

    setDraft((current) =>
      current.trim()
        ? `${current.trimEnd()} ${normalized}`
        : normalized,
    )
    textareaRef.current?.focus()
  }

  function startBrowserSpeechRecognition(reason?: string): boolean {
    const speechWindow = window as typeof window & {
      SpeechRecognition?: BrowserSpeechRecognitionConstructor
      webkitSpeechRecognition?: BrowserSpeechRecognitionConstructor
    }
    const SpeechRecognition =
      speechWindow.SpeechRecognition ??
      speechWindow.webkitSpeechRecognition

    if (!SpeechRecognition) {
      setVoiceInputState("idle")
      setActivity(
        reason
          ? `${reason}；当前浏览器不支持语音识别。`
          : "当前浏览器不支持语音识别。",
      )
      return false
    }

    try {
      const recognition = new SpeechRecognition()
      let finalTextReceived = false
      let recognitionError = ""

      recognition.lang = "zh-CN"
      recognition.interimResults = false
      recognition.continuous = false
      recognition.onresult = (event) => {
        const text = Array.from(
          { length: event.results.length - event.resultIndex },
          (_, index) => event.results[event.resultIndex + index],
        )
          .filter((result) => result?.isFinal)
          .map((result) => result[0].transcript)
          .join("")

        if (text.trim()) {
          finalTextReceived = true
          appendRecognizedText(text)
        }
      }
      recognition.onerror = (event) => {
        recognitionError = event.error || "语音识别失败"
      }
      recognition.onend = () => {
        if (speechRecognitionRef.current !== recognition) return

        speechRecognitionRef.current = null
        setVoiceInputState("idle")
        setActivity(
          finalTextReceived
            ? "语音识别完成，文字已填入输入框。"
            : recognitionError
              ? `浏览器语音识别失败：${recognitionError}`
              : "没有识别到语音，请重试。",
        )
      }

      speechRecognitionRef.current = recognition
      setVoiceInputState("recognizing")
      setActivity(
        reason
          ? `${reason}，正在使用浏览器语音识别…`
          : "正在使用浏览器语音识别…",
      )
      recognition.start()
      return true
    } catch {
      speechRecognitionRef.current = null
      setVoiceInputState("idle")
      setActivity("无法启动浏览器语音识别，请检查麦克风权限。")
      return false
    }
  }

  function releaseAudioStream() {
    audioStreamRef.current?.getTracks().forEach((track) => track.stop())
    audioStreamRef.current = null
  }

  async function transcribeRecordedAudio(
    blob: Blob,
    format: string,
    sampleRate: number,
  ) {
    if (!blob.size) {
      setVoiceInputState("idle")
      setActivity("没有录到音频，请重试。")
      return
    }

    setVoiceInputState("processing")
    setActivity("正在识别录音…")

    try {
      const audioData = arrayBufferToBase64(await blob.arrayBuffer())
      const response = await fetch(
        `${getApiBaseUrl()}/api/asr/recognize`,
        {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            audio_data: audioData,
            format,
            sample_rate: sampleRate,
            language_hints: ["zh", "en"],
          }),
        },
      )
      const result = (await response.json()) as {
        success?: boolean
        text?: string
        error?: string
      }

      if (!response.ok || !result.success) {
        throw new Error(result.error || `语音识别请求失败（${response.status}）`)
      }

      const text = result.text?.trim() ?? ""
      if (!text) throw new Error("没有识别到语音内容")

      appendRecognizedText(text)
      setVoiceInputState("idle")
      setActivity("语音识别完成，文字已填入输入框。")
    } catch (error) {
      const reason =
        error instanceof Error ? error.message : "服务器语音识别不可用"
      if (!startBrowserSpeechRecognition(`${reason}，正在回退`)) {
        setVoiceInputState("idle")
      }
    }
  }

  async function startAudioRecording() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      startBrowserSpeechRecognition("当前设备无法录音")
      return
    }

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      startBrowserSpeechRecognition("无法访问麦克风")
      return
    }

    audioStreamRef.current = stream
    audioChunksRef.current = []

    const formats = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/ogg;codecs=opus",
    ]
    const mimeType = formats.find((format) =>
      MediaRecorder.isTypeSupported(format),
    )
    let recorder: MediaRecorder

    try {
      recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      )
    } catch {
      releaseAudioStream()
      startBrowserSpeechRecognition("当前浏览器无法录制音频")
      return
    }

    const sampleRate = Math.min(
      48000,
      Math.max(
        8000,
        stream.getAudioTracks()[0]?.getSettings().sampleRate ?? 16000,
      ),
    )
    mediaRecorderRef.current = recorder
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunksRef.current.push(event.data)
    }
    recorder.onerror = () => {
      recorder.onstop = null
      mediaRecorderRef.current = null
      audioChunksRef.current = []
      releaseAudioStream()
      startBrowserSpeechRecognition("录音失败")
    }
    recorder.onstop = () => {
      const blob = new Blob(audioChunksRef.current, {
        type: recorder.mimeType || mimeType || "audio/webm",
      })
      audioChunksRef.current = []
      mediaRecorderRef.current = null
      releaseAudioStream()
      void transcribeRecordedAudio(
        blob,
        audioFormatFromMimeType(blob.type),
        sampleRate,
      )
    }

    try {
      recorder.start()
      setVoiceInputState("recording")
      setActivity("正在录音，再次点击麦克风结束录音。")
    } catch {
      mediaRecorderRef.current = null
      releaseAudioStream()
      startBrowserSpeechRecognition("无法开始录音")
    }
  }

  async function handleVoiceInput() {
    if (voiceInputState === "recording") {
      setVoiceInputState("processing")
      mediaRecorderRef.current?.stop()
      return
    }

    if (voiceInputState === "recognizing") {
      speechRecognitionRef.current?.stop()
      return
    }

    if (voiceInputState !== "idle") return

    setVoiceInputState("checking")
    setActivity("正在检查语音识别服务…")
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/asr/status`, {
        headers: { Accept: "application/json" },
      })
      if (!response.ok) throw new Error("无法连接语音识别服务")

      const status = (await response.json()) as { enabled?: boolean }
      if (!status.enabled) {
        startBrowserSpeechRecognition("服务器语音识别未启用，正在回退")
        return
      }

      await startAudioRecording()
    } catch (error) {
      startBrowserSpeechRecognition(
        error instanceof Error ? error.message : "服务器语音识别不可用",
      )
    }
  }

  function handleConversationScroll() {
    const conversation = conversationRef.current

    if (!conversation) {
      return
    }

    const distanceFromBottom =
      conversation.scrollHeight -
      conversation.scrollTop -
      conversation.clientHeight

    stickToBottomRef.current = distanceFromBottom < 120
  }

  function useDiscoverPrompt(prompt: string) {
    setDraft(prompt)
    setActivity("")
    focusComposerAfterNavigation.current = true
    setActiveNav("Dashboard")
  }

  function scrollToBottom() {
    stickToBottomRef.current = true
    conversationRef.current?.scrollTo({
      top: conversationRef.current.scrollHeight,
      behavior: "smooth",
    })
  }

  async function removeLibrarySession(
    session: SessionSummary,
  ) {
    if (
      running ||
      !window.confirm(
        "确定删除此会话吗？此操作无法撤销。",
      )
    ) {
      return
    }

    try {
      await deleteSession(session.thread_id)
      setSessions((current) =>
        current.filter(
          (item) =>
            item.thread_id !== session.thread_id,
        ),
      )
      if (sessionPreferencesLoaded) {
        const remainingPreferences = { ...sessionPreferences }
        delete remainingPreferences[session.thread_id]
        if (persistSessionPreferences(remainingPreferences)) {
          setSessionPreferences(remainingPreferences)
        }
      }

      if (activeThreadId === session.thread_id) {
        resetWorkspace()
      }
    } catch (error) {
      setActivity(
        error instanceof Error
          ? userFacingErrorMessage(error.message)
          : "无法删除会话。",
      )
    }
  }

  function removeSavedAnswer(answer: SavedAnswer) {
    if (
      !window.confirm("确定从资料库中删除这条已保存回答吗？")
    ) {
      return
    }

    const nextAnswers = savedAnswers.filter(
      (item) => item.id !== answer.id,
    )
    if (!persistSavedAnswers(nextAnswers)) {
      setActivity("删除失败，请检查浏览器本地存储空间")
      return
    }

    setSavedAnswers(nextAnswers)
  }

  function removeSavedArtifact(artifact: SavedArtifact) {
    if (!window.confirm("确定从资料库中删除这个生成制品吗？")) {
      return
    }

    const nextArtifacts = savedArtifacts.filter(
      (item) => item.savedId !== artifact.savedId,
    )
    if (!persistSavedArtifacts(nextArtifacts)) {
      setActivity("删除失败，请检查浏览器本地存储空间")
      return
    }

    setSavedArtifacts(nextArtifacts)
  }

  function sessionDisplayTitle(session: SessionSummary): string {
    return (
      sessionPreferences[session.thread_id]?.title?.trim() ||
      session.topic.trim() ||
      "未命名研究"
    )
  }

  function sessionIsPinned(threadId: string): boolean {
    return sessionPreferences[threadId]?.pinned === true
  }

  function updateSessionPreference(
    threadId: string,
    updates: Partial<SessionPreferences[string]>,
  ): boolean {
    if (!sessionPreferencesLoaded) return false

    const nextPreferences = {
      ...sessionPreferences,
      [threadId]: {
        title: sessionPreferences[threadId]?.title,
        pinned: sessionIsPinned(threadId),
        ...updates,
      },
    }

    if (!persistSessionPreferences(nextPreferences)) {
      setActivity("保存会话设置失败，请检查浏览器本地存储空间")
      return false
    }

    setSessionPreferences(nextPreferences)
    return true
  }

  function toggleSessionPin(session: SessionSummary) {
    updateSessionPreference(session.thread_id, {
      pinned: !sessionIsPinned(session.thread_id),
    })
  }

  function beginRenameSession(session: SessionSummary) {
    setRenameSessionId(session.thread_id)
    setRenameSessionDraft(sessionDisplayTitle(session))
  }

  function closeRenameSession() {
    setRenameSessionId(null)
    setRenameSessionDraft("")
  }

  function saveRenamedSession() {
    if (!renameSessionId) return
    const title = renameSessionDraft.trim()

    if (!title) return

    if (updateSessionPreference(renameSessionId, { title })) {
      closeRenameSession()
    }
  }

  const normalizedLibraryQuery =
    libraryQuery.trim().toLowerCase()
  const orderedSessions = [...sessions].sort((left, right) => {
    const pinOrder =
      Number(sessionIsPinned(right.thread_id)) -
      Number(sessionIsPinned(left.thread_id))
    if (pinOrder !== 0) return pinOrder

    return (
      new Date(right.updated_at).getTime() -
      new Date(left.updated_at).getTime()
    )
  })
  const librarySessions = orderedSessions.filter((session) => {
    const haystack = [
      sessionDisplayTitle(session),
      session.thread_id,
      session.route,
      session.status,
      sessionRouteLabel(session.route),
      sessionStatusLabel(session.status),
    ]
      .join(" ")
      .toLowerCase()

    const matchesQuery =
      !normalizedLibraryQuery ||
      haystack.includes(normalizedLibraryQuery)

    const normalizedStatus =
      session.status.toLowerCase()
    const matchesFilter =
      libraryFilter === "all" ||
      (libraryFilter === "active" &&
        ["running", "paused"].includes(
          normalizedStatus,
        )) ||
      (libraryFilter === "completed" &&
        normalizedStatus === "completed") ||
      (libraryFilter === "failed" &&
        ["failed", "cancelled"].includes(
          normalizedStatus,
        ))

    return (
      libraryKindFilter !== "saved" &&
      libraryKindFilter !== "artifacts" &&
      libraryKindFilter !== "documents" &&
      (libraryKindFilter !== "pinned" ||
        sessionIsPinned(session.thread_id)) &&
      matchesQuery &&
      matchesFilter
    )
  })

  const librarySavedAnswers = savedAnswers.filter((answer) => {
    if (
      libraryKindFilter === "sessions" ||
      libraryKindFilter === "pinned" ||
      libraryKindFilter === "artifacts" ||
      libraryKindFilter === "documents"
    ) {
      return false
    }

    return (
      !normalizedLibraryQuery ||
      `${answer.title} ${answer.content}`
        .toLowerCase()
        .includes(normalizedLibraryQuery)
    )
  })

  const librarySavedArtifacts = savedArtifacts.filter((artifact) => {
    if (
      libraryKindFilter === "sessions" ||
      libraryKindFilter === "pinned" ||
      libraryKindFilter === "saved" ||
      libraryKindFilter === "documents"
    ) {
      return false
    }

    const searchable = [
      artifact.title,
      artifact.type,
      artifact.content ?? "",
      artifact.sessionId ?? "",
    ]
      .join(" ")
      .toLowerCase()

    return !normalizedLibraryQuery || searchable.includes(normalizedLibraryQuery)
  })

  const savedMessageIds = new Set(
    savedAnswers.map((answer) => answer.sourceMessageId),
  )
  const savedArtifactKeys = new Set(
    savedArtifacts.map((artifact) =>
      savedArtifactSourceKey(
        artifact.sourceArtifactId,
        artifact.sessionId,
      ),
    ),
  )

  const normalizedDiscoverQuery =
    discoverQuery.trim().toLowerCase()
  const visibleDiscoverPrompts = discoverPrompts.filter(
    (item) => {
      const matchesCategory =
        discoverCategory === "All" ||
        item.category === discoverCategory
      const searchableText = [
        item.title,
        item.description,
        item.category,
        discoverCategoryLabels[item.category],
        item.prompt,
      ]
        .join(" ")
        .toLowerCase()
      const matchesQuery =
        !normalizedDiscoverQuery ||
        searchableText.includes(normalizedDiscoverQuery)

      return matchesCategory && matchesQuery
    },
  )

  const hasCompletedReport = messages.some(
    (message) =>
      message.role === "assistant" &&
      (message.status === "Completed" ||
        message.status === "Restored") &&
      Boolean(message.content.trim()),
  )

  const promptCommands = [
    {
      id: "direct",
      label: "普通对话",
      description: "关闭特殊研究模式",
      icon: MessageSquare,
    },
    {
      id: "web",
      label: "Web 搜索",
      description: "使用网页搜索辅助回答",
      icon: Globe2,
    },
    {
      id: "agent",
      label: "智能体模式",
      description: "规划并调用工具完成任务",
      icon: Bot,
    },
    {
      id: "deep",
      label: "深度研究",
      description: "执行多轮检索与证据分析",
      icon: Rocket,
    },
    {
      id: "fix",
      label: "修复代码",
      description: "生成代码问题定位模板",
      icon: Wrench,
    },
    {
      id: "explain",
      label: "解释概念",
      description: "生成概念解释模板",
      icon: BookOpen,
    },
    {
      id: "test",
      label: "编写测试",
      description: "生成测试用例模板",
      icon: ListChecks,
    },
    {
      id: "clear",
      label: "清空会话",
      description: "开始新的研究会话",
      icon: Trash2,
    },
  ]

  const selectedMcpLabel =
    mcpOptions.find((option) => option.id === selectedMcp)?.label ??
    "MCP"
  const generatedArtifacts = messages
    .flatMap((message) => message.generatedArtifacts ?? [])
    .filter(
      (artifact, index, all) =>
        all.findIndex((item) => item.id === artifact.id) === index,
    )
    .slice(-20)

  useEffect(() => {
    if (generatedArtifacts.length === 0) {
      setArtifactsFullscreenOpen(false)
    }
  }, [generatedArtifacts.length])

  return (
    <div
      className={`workspace-shell ${
        sidebarOpen ? "" : "sidebar-hidden"
      }`}
      data-theme={darkMode ? "dark" : "light"}
    >
      <aside className="app-sidebar">
        <div className="sidebar-brand-row">
          <div className="brand-lockup">
            <div className="brand-mark">P</div>

            {sidebarOpen && (
              <span className="brand-name">
                PuzzleResearch
              </span>
            )}
          </div>

          <button
            className="sidebar-collapse"
            type="button"
            title={
              sidebarOpen
                ? "收起侧栏"
                : "展开侧栏"
            }
            onClick={() =>
              setSidebarOpen((value) => !value)
            }
          >
            <PanelLeft size={15} />
          </button>
        </div>

        {sidebarOpen && (
          <>
            <button
              className="new-investigation"
              type="button"
              onClick={resetWorkspace}
            >
              <Plus size={16} />
              <span>新建研究</span>
            </button>

            <nav className="sidebar-navigation">
              <div className="sidebar-heading">
                工作区
              </div>

              {navigationItems.map((item) => {
                const Icon = item.icon
                const active =
                  activeNav === item.key

                return (
                  <button
                    className={`sidebar-nav-item ${
                      active ? "active" : ""
                    }`}
                    type="button"
                    key={item.key}
                    onClick={() => {
                      setActiveNav(item.key)
                      setActivity("")
                    }}
                  >
                    <Icon size={15} />
                    <span>{item.label}</span>
                  </button>
                )
              })}
            </nav>

            <div className="today-section">
              <div className="sidebar-heading">
                最近会话
              </div>

              {sessionsLoading ? (
                <div className="recent-empty">
                  正在加载会话…
                </div>
              ) : sessions.length === 0 ? (
                <div className="recent-empty">暂无历史会话</div>
              ) : (
                orderedSessions.slice(0, 8).map((session) => (
                  <button
                    className={`recent-item ${
                      activeThreadId === session.thread_id
                        ? "active"
                        : ""
                    }`}
                    type="button"
                    key={session.thread_id}
                    title={session.topic}
                    onClick={() => void openSession(session)}
                  >
                    <MessageSquare size={15} />
                    <span>
                      {sessionDisplayTitle(session)}
                    </span>
                    {sessionIsPinned(session.thread_id) && (
                      <Pin size={12} aria-label="已置顶" />
                    )}
                  </button>
                ))
              )}
            </div>
          </>
        )}

        <div className="sidebar-footer">
          <button
            className="sidebar-footer-button"
            type="button"
            title={darkMode ? "切换到浅色模式" : "切换到深色模式"}
            aria-label={darkMode ? "切换到浅色模式" : "切换到深色模式"}
            onClick={() => setDarkMode((value) => !value)}
          >
            {darkMode ? <Sun size={17} /> : <Moon size={17} />}
            {sidebarOpen && <span>外观</span>}
          </button>
          <button
            className="sidebar-footer-button"
            type="button"
            title="设置"
            aria-label="打开设置"
            onClick={() => setSettingsOpen(true)}
          >
            <Settings size={17} />
            {sidebarOpen && <span>设置</span>}
          </button>
        </div>
      </aside>

      <main className="main-panel">
        <header className="app-header">
          <button
            className="header-menu-button"
            type="button"
            title="打开侧栏"
            onClick={() => setSidebarOpen(true)}
          >
            <PanelLeft size={17} />
          </button>

          <div className="header-actions">
            <div className="model-selector">
              <button
                className="model-button"
                type="button"
                onClick={() =>
                  setModelMenuOpen((value) => !value)
                }
              >
                <span>{model}</span>
                <ChevronDown size={14} />
              </button>

              {modelMenuOpen && (
                <div className="model-menu">
                  {modelOptions.map((option) => (
                    <button
                      className={
                        option === model
                          ? "model-option selected"
                          : "model-option"
                      }
                      type="button"
                      key={option}
                      onClick={() => chooseModel(option)}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {activeThreadId && (
              <button
                className={`header-icon-button ${workspaceFilesOpen ? "active" : ""}`}
                type="button"
                title={workspaceFilesOpen ? "关闭工作区文件" : "查看工作区文件"}
                aria-label={workspaceFilesOpen ? "关闭工作区文件" : "查看工作区文件"}
                aria-pressed={workspaceFilesOpen}
                onClick={() => {
                  setBrowserPreviewOpen(false)
                  setWorkspaceFilesOpen((open) => !open)
                }}
              >
                <FolderOpen size={17} />
              </button>
            )}

            {generatedArtifacts.length > 0 && (
              <button
                className={`header-icon-button ${artifactsPanelOpen ? "active" : ""}`}
                type="button"
                title={artifactsPanelOpen ? "关闭生成制品" : "查看生成制品"}
                aria-label={artifactsPanelOpen ? "关闭生成制品" : "查看生成制品"}
                aria-pressed={artifactsPanelOpen}
                onClick={() => {
                  setBrowserPreviewOpen(false)
                  setWorkspaceFilesOpen(false)
                  setArtifactsPanelOpen((open) => !open)
                }}
              >
                <PanelRight size={17} />
              </button>
            )}

            {activeThreadId && (
              <button
                className={`header-icon-button ${
                  browserPreviewOpen ? "active" : ""
                }`}
                type="button"
                title={
                  browserPreviewOpen
                    ? "关闭浏览器预览"
                    : "打开浏览器预览"
                }
                aria-label={
                  browserPreviewOpen
                    ? "关闭浏览器预览"
                    : "打开浏览器预览"
                }
                aria-pressed={browserPreviewOpen}
                onClick={() => {
                  setWorkspaceFilesOpen(false)
                  setBrowserPreviewOpen((open) => !open)
                }}
              >
                <Monitor size={18} />
              </button>
            )}

          </div>
        </header>

        {activeNav === "Discover" ? (
          <section className="discover-view">
            <div className="discover-content">
              <div className="discover-heading">
                <span className="discover-eyebrow">
                  工作区
                </span>
                <h1>
                  <Compass size={24} />
                  发现
                </h1>
                <p>
                  从精选提示开始，快速梳理你的研究问题。
                </p>
              </div>

              <label className="discover-search">
                <Search size={16} />
                <input
                  aria-label="搜索提示词"
                  value={discoverQuery}
                  placeholder="搜索研究提示词…"
                  onChange={(event) =>
                    setDiscoverQuery(event.target.value)
                  }
                />
              </label>

              <div
                className="discover-categories"
                aria-label="提示词类别"
              >
                {discoverCategories.map((category) => (
                  <button
                    className={
                      discoverCategory === category
                        ? "discover-category active"
                        : "discover-category"
                    }
                    type="button"
                    key={category}
                    aria-pressed={
                      discoverCategory === category
                    }
                    onClick={() =>
                      setDiscoverCategory(category)
                    }
                  >
                    {discoverCategoryLabels[category]}
                  </button>
                ))}
              </div>

              <div className="discover-results-heading">
                <span>精选提示词</span>
                <small>
                  共 {visibleDiscoverPrompts.length} 条
                </small>
              </div>

              {visibleDiscoverPrompts.length === 0 ? (
                <div className="discover-empty">
                  <Search size={24} />
                  <h2>没有找到匹配的提示词</h2>
                  <p>试试更换关键词或类别。</p>
                </div>
              ) : (
                <div className="discover-grid">
                  {visibleDiscoverPrompts.map((item) => {
                    const Icon = item.icon

                    return (
                      <article
                        className="discover-card"
                        key={item.title}
                      >
                        <div className="discover-card-heading">
                          <span className="discover-card-icon">
                            <span className={`discover-card-icon-tone ${item.category.toLowerCase()}`}>
                            <Icon size={17} />
                            </span>
                          </span>
                          <span className="discover-card-category">
                            {discoverCategoryLabels[item.category]}
                          </span>
                        </div>
                        <h2>{item.title}</h2>
                        <p>{item.description}</p>
                        <button
                          className="discover-use-button"
                          type="button"
                          onClick={() =>
                            useDiscoverPrompt(item.prompt)
                          }
                        >
                          <span>使用此提示</span>
                          <ArrowUp size={14} />
                        </button>
                      </article>
                    )
                  })}
                </div>
              )}
            </div>
          </section>
        ) : activeNav === "Library" ? (
          <section className="library-view">
            <div className="library-content">
              <div className="library-heading">
                <div>
                  <span className="library-eyebrow">
                    工作区
                  </span>
                  <h1>
                    <FolderOpen size={24} />
                    资料库
                  </h1>
                  <p>
                    管理研究会话、已保存回答、生成制品与知识库文档。
                  </p>
                </div>
                <button
                  className="library-new-button"
                  type="button"
                  onClick={resetWorkspace}
                >
                  <Plus size={15} />
                  <span>新建研究</span>
                </button>
              </div>

              <div className="library-toolbar">
                {libraryKindFilter !== "documents" && (
                  <label className="library-search">
                    <Search size={15} />
                    <input
                      aria-label="搜索资料库"
                      value={libraryQuery}
                      placeholder="搜索会话、已保存回答和制品…"
                      onChange={(event) =>
                        setLibraryQuery(event.target.value)
                      }
                    />
                  </label>
                )}

                <div className="library-filter-groups">
                  <div
                    className="library-filters"
                    role="group"
                    aria-label="资料类型"
                  >
                    {[
                      ["all", "全部"],
                      ["sessions", "研究会话"],
                      ["saved", "已保存回答"],
                      ["artifacts", "生成制品"],
                      ["pinned", "置顶会话"],
                      ["documents", "知识库文档"],
                    ].map(([value, label]) => (
                      <button
                        className={
                          libraryKindFilter === value
                            ? "library-filter active"
                            : "library-filter"
                        }
                        type="button"
                        key={value}
                        aria-pressed={libraryKindFilter === value}
                        onClick={() =>
                          setLibraryKindFilter(
                            value as LibraryKindFilter,
                          )
                        }
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  {libraryKindFilter !== "saved" &&
                    libraryKindFilter !== "artifacts" &&
                    libraryKindFilter !== "documents" && (
                    <div
                      className="library-filters library-status-filters"
                      role="group"
                      aria-label="会话状态"
                    >
                      {[
                        ["all", "全部状态"],
                        ["active", "进行中"],
                        ["completed", "已完成"],
                        ["failed", "失败"],
                      ].map(([value, label]) => (
                        <button
                          className={
                            libraryFilter === value
                              ? "library-filter active"
                              : "library-filter"
                          }
                          type="button"
                          key={value}
                          aria-pressed={libraryFilter === value}
                          onClick={() =>
                            setLibraryFilter(
                              value as LibraryFilter,
                            )
                          }
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="library-summary">
                <span>
                  {libraryKindFilter === "documents"
                    ? `${knowledgeDocuments.length} 份知识库文档`
                    : `${librarySessions.length} 个会话 · ${librarySavedAnswers.length} 条已保存回答 · ${librarySavedArtifacts.length} 个生成制品`}
                </span>
                {libraryKindFilter === "documents" && knowledgeDocumentsLoading ? (
                  <span>正在刷新…</span>
                ) : sessionsLoading ? (
                  <span>正在刷新…</span>
                ) : null}
              </div>

              {libraryKindFilter === "documents" && (
                <section
                  className="knowledge-documents"
                  aria-label="知识库文档管理"
                >
                  <input
                    ref={knowledgeDocumentInputRef}
                    className="knowledge-document-input"
                    type="file"
                    accept=".txt,.md,.csv,.pdf,.docx"
                    multiple
                    onChange={(event) => {
                      const files = event.currentTarget.files
                      void handleKnowledgeDocumentUpload(files)
                      event.currentTarget.value = ""
                    }}
                  />
                  <div className="knowledge-documents-toolbar">
                    <div>
                      <h2>知识库文档</h2>
                      <p>上传文档供研究 Agent 检索引用，单个文件最大 50 MB。</p>
                    </div>
                    <div className="knowledge-document-actions">
                      <button
                        className="library-new-button"
                        type="button"
                        disabled={knowledgeUploadLoading}
                        onClick={() => knowledgeDocumentInputRef.current?.click()}
                      >
                        {knowledgeUploadLoading ? (
                          <LoaderCircle size={14} className="spin" />
                        ) : (
                          <Upload size={14} />
                        )}
                        <span>{knowledgeUploadLoading ? "正在上传…" : "上传文档"}</span>
                      </button>
                      <button
                        className="library-session-action"
                        type="button"
                        title="刷新文档列表"
                        aria-label="刷新文档列表"
                        disabled={knowledgeDocumentsLoading}
                        onClick={() => void refreshKnowledgeDocuments()}
                      >
                        <RefreshCw
                          size={14}
                          className={knowledgeDocumentsLoading ? "spin" : ""}
                        />
                      </button>
                    </div>
                  </div>

                  <form
                    className="knowledge-document-search"
                    onSubmit={(event) => void runKnowledgeSearch(event)}
                  >
                    <label className="library-search">
                      <Search size={15} />
                      <input
                        aria-label="搜索知识库文档"
                        value={knowledgeQuery}
                        placeholder="搜索文档内容…"
                        onChange={(event) => setKnowledgeQuery(event.target.value)}
                      />
                    </label>
                    <button
                      className="library-new-button"
                      type="submit"
                      disabled={knowledgeSearchLoading || !knowledgeQuery.trim()}
                    >
                      {knowledgeSearchLoading ? (
                        <LoaderCircle size={14} className="spin" />
                      ) : (
                        <Search size={14} />
                      )}
                      <span>搜索</span>
                    </button>
                  </form>

                  {knowledgeDocumentsError && (
                    <p className="knowledge-document-message error" role="alert">
                      {knowledgeDocumentsError}
                    </p>
                  )}
                  {knowledgeDocumentsNotice && (
                    <p className="knowledge-document-message" role="status">
                      {knowledgeDocumentsNotice}
                    </p>
                  )}

                  {knowledgeSearchResults !== null && (
                    <section className="knowledge-search-results" aria-label="文档搜索结果">
                      <div className="library-section-heading">
                        <h2>搜索结果</h2>
                        <small>{knowledgeSearchResults.length} 条</small>
                      </div>
                      {knowledgeSearchResults.length === 0 ? (
                        <p className="knowledge-document-empty">没有找到匹配的文档片段。</p>
                      ) : (
                        knowledgeSearchResults.map((result, index) => (
                          <article
                            className="knowledge-search-result"
                            key={`${result.source}-${result.chunk_index}-${index}`}
                          >
                            <div>
                              <strong>{result.source}</strong>
                              <small>片段 {result.chunk_index + 1} · 相关度 {result.score.toFixed(2)}</small>
                            </div>
                            <p>{result.text.slice(0, 700)}</p>
                          </article>
                        ))
                      )}
                    </section>
                  )}

                  {knowledgeDocumentsLoading ? (
                    <div className="knowledge-document-empty">正在读取文档列表…</div>
                  ) : knowledgeDocuments.length === 0 ? (
                    !knowledgeDocumentsError && (
                      <div className="knowledge-document-empty">
                        <FileText size={22} />
                        <strong>知识库中还没有文档</strong>
                        <span>上传 TXT、Markdown、CSV、PDF 或 DOCX 文件开始使用。</span>
                      </div>
                    )
                  ) : (
                    <div className="knowledge-document-table-wrap">
                      <table className="knowledge-document-table">
                        <thead>
                          <tr>
                            <th scope="col">文件</th>
                            <th scope="col">文本块</th>
                            <th scope="col">大小</th>
                            <th scope="col">上传时间</th>
                            <th scope="col" aria-label="操作"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {knowledgeDocuments.map((document) => (
                            <tr key={document.source}>
                              <td>
                                <span className="knowledge-document-name">
                                  <FileText size={15} />
                                  <strong title={document.filename}>{document.filename}</strong>
                                </span>
                              </td>
                              <td>{document.chunks}</td>
                              <td>{(document.size / (1024 * 1024)).toFixed(2)} MB</td>
                              <td>
                                {new Date(document.uploaded_at).toLocaleString("zh-CN", {
                                  dateStyle: "medium",
                                  timeStyle: "short",
                                })}
                              </td>
                              <td>
                                <button
                                  className="library-delete-button"
                                  type="button"
                                  title={`删除 ${document.filename}`}
                                  aria-label={`删除知识库文档：${document.filename}`}
                                  onClick={() => void removeKnowledgeDocument(document)}
                                >
                                  <Trash2 size={14} />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              )}

              {libraryKindFilter !== "documents" && librarySavedAnswers.length > 0 && (
                <section
                  className="library-saved-section"
                  aria-label="已保存回答"
                >
                  <div className="library-section-heading">
                    <h2>已保存回答</h2>
                    <small>{librarySavedAnswers.length} 条</small>
                  </div>
                  <div className="library-saved-grid">
                    {librarySavedAnswers.map((answer) => (
                      <article
                        className="library-saved-card"
                        key={answer.id}
                      >
                        <div className="library-saved-heading">
                          <span className="library-saved-title">
                            <FileText size={15} />
                            <strong>{answer.title}</strong>
                          </span>
                          <button
                            className="library-delete-button"
                            type="button"
                            title="删除已保存回答"
                            aria-label={`删除已保存回答：${answer.title}`}
                            onClick={() => removeSavedAnswer(answer)}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <small className="library-saved-date">
                          {new Date(answer.createdAt).toLocaleString(
                            "zh-CN",
                            {
                              dateStyle: "medium",
                              timeStyle: "short",
                            },
                          )}
                        </small>
                        <details className="library-saved-details">
                          <summary>展开回答</summary>
                          <AssistantMarkdown
                            content={answer.content}
                            sourceCount={0}
                          />
                        </details>
                      </article>
                    ))}
                  </div>
                </section>
              )}

              {libraryKindFilter !== "documents" && librarySavedArtifacts.length > 0 && (
                <section
                  className="library-saved-section"
                  aria-label="生成制品"
                >
                  <div className="library-section-heading">
                    <h2>生成制品</h2>
                    <small>{librarySavedArtifacts.length} 项</small>
                  </div>
                  <div className="library-saved-grid">
                    {librarySavedArtifacts.map((artifact) => {
                      const isVisual =
                        (artifact.type === "image" ||
                          artifact.type === "chart") &&
                        Boolean(artifact.url)

                      return (
                        <article
                          className="library-saved-card"
                          key={artifact.savedId}
                        >
                          <div className="library-saved-heading">
                            <span className="library-saved-title">
                              {artifact.type === "code" ? (
                                <Code2 size={15} />
                              ) : artifact.type === "chart" ? (
                                <BarChart3 size={15} />
                              ) : (
                                <FileText size={15} />
                              )}
                              <strong>{artifact.title}</strong>
                            </span>
                            <button
                              className="library-delete-button"
                              type="button"
                              title="删除生成制品"
                              aria-label={`删除生成制品：${artifact.title}`}
                              onClick={() => removeSavedArtifact(artifact)}
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                          <small className="library-saved-date">
                            {generatedArtifactTypeLabel(artifact.type)} · {new Date(artifact.createdAt).toLocaleString("zh-CN")}
                          </small>
                          {isVisual && artifact.url && (
                            <a
                              className="generated-artifact-preview library-artifact-preview"
                              href={resolveApiUrl(artifact.url)}
                              target="_blank"
                              rel="noreferrer"
                            >
                              <img
                                src={resolveApiUrl(artifact.url)}
                                alt={artifact.title}
                                loading="lazy"
                              />
                            </a>
                          )}
                          {artifact.content && (
                            <pre className="library-artifact-content">
                              {artifact.content}
                            </pre>
                          )}
                          {artifact.url && !isVisual && (
                            <a
                              className="generated-artifact-download"
                              href={resolveApiUrl(artifact.url)}
                              download
                              target="_blank"
                              rel="noreferrer"
                            >
                              <Download size={13} />
                              下载制品
                            </a>
                          )}
                        </article>
                      )
                    })}
                  </div>
                </section>
              )}

              {libraryKindFilter !== "documents" && (librarySessions.length === 0 &&
              librarySavedAnswers.length === 0 &&
              librarySavedArtifacts.length === 0 ? (
                <div className="library-empty">
                  <History size={28} />
                  <h2>
                    {libraryKindFilter === "saved"
                      ? "没有找到已保存回答"
                      : libraryKindFilter === "artifacts"
                        ? "没有找到生成制品"
                      : libraryKindFilter === "pinned"
                        ? "没有置顶会话"
                      : libraryKindFilter === "sessions"
                        ? "没有找到研究会话"
                        : "资料库中没有匹配内容"}
                  </h2>
                  <p>
                    {libraryQuery.trim()
                      ? "换个关键词再试试。"
                      : libraryKindFilter === "saved"
                        ? "在回答下方点击收藏按钮，保存后可在这里查看。"
                        : libraryKindFilter === "artifacts"
                          ? "在生成内容旁点击收藏按钮，制品就会保存在这里。"
                        : libraryKindFilter === "pinned"
                          ? "点击会话上的图钉按钮即可置顶。"
                          : "开始一项研究后，会话会显示在这里。"}
                  </p>
                </div>
              ) : librarySessions.length > 0 ? (
                <div className="library-session-grid">
                  {librarySessions.map((session) => (
                    <article
                      className={`library-session-card${
                        sessionIsPinned(session.thread_id)
                          ? " pinned"
                          : ""
                      }`}
                      key={session.thread_id}
                    >
                      <button
                        className="library-session-open"
                        type="button"
                        onClick={() =>
                          void openSession(session)
                        }
                      >
                        <span className="library-session-title">
                          <MessageSquare size={16} />
                          <strong>
                            {sessionDisplayTitle(session)}
                          </strong>
                        </span>
                        <span className="library-session-preview">
                          {session.message_count} 条消息
                          {session.route
                            ? ` · ${sessionRouteLabel(session.route)}`
                            : ""}
                        </span>
                        <span className="library-session-date">
                          {session.updated_at
                            .replace("T", " ")
                            .slice(0, 16)}
                        </span>
                      </button>

                      <div className="library-session-footer">
                        <span
                          className={`library-status ${session.status.toLowerCase()}`}
                        >
                          {sessionStatusLabel(session.status)}
                        </span>
                        <div className="library-session-actions">
                          <button
                            className="library-session-action"
                            type="button"
                            title={
                              sessionIsPinned(session.thread_id)
                                ? "取消置顶"
                                : "置顶会话"
                            }
                            aria-label={
                              sessionIsPinned(session.thread_id)
                                ? "取消置顶会话"
                                : "置顶会话"
                            }
                            disabled={!sessionPreferencesLoaded}
                            onClick={() => toggleSessionPin(session)}
                          >
                            {sessionIsPinned(session.thread_id) ? (
                              <PinOff size={14} />
                            ) : (
                              <Pin size={14} />
                            )}
                          </button>
                          <button
                            className="library-session-action"
                            type="button"
                            title="重命名会话"
                            aria-label="重命名会话"
                            onClick={() => beginRenameSession(session)}
                          >
                            <Pencil size={14} />
                          </button>
                          <button
                            className="library-delete-button"
                            type="button"
                            title="删除会话"
                            aria-label="删除会话"
                            onClick={() =>
                              void removeLibrarySession(session)
                            }
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              ) : null)}

              {renameSessionId && (
                <div className="library-dialog-backdrop">
                  <form
                    className="library-rename-dialog"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby="library-rename-title"
                    onSubmit={(event) => {
                      event.preventDefault()
                      saveRenamedSession()
                    }}
                  >
                    <h2 id="library-rename-title">重命名会话</h2>
                    <label className="library-rename-field">
                      <span>会话名称</span>
                      <input
                        autoFocus
                        maxLength={80}
                        value={renameSessionDraft}
                        onChange={(event) =>
                          setRenameSessionDraft(event.target.value)
                        }
                      />
                    </label>
                    <p>名称和置顶状态仅保存在此浏览器中。</p>
                    <div className="library-rename-actions">
                      <button
                        className="settings-secondary-button"
                        type="button"
                        onClick={closeRenameSession}
                      >
                        取消
                      </button>
                      <button
                        className="settings-primary-button"
                        type="submit"
                        disabled={
                          !sessionPreferencesLoaded ||
                          !renameSessionDraft.trim()
                        }
                      >
                        保存名称
                      </button>
                    </div>
                  </form>
                </div>
              )}
            </div>
          </section>
        ) : activeNav === "Triggers" ? (
          <TriggersPanel />
        ) : (
          <>
        <section
          className="conversation"
          ref={conversationRef}
          onScroll={handleConversationScroll}
        >
          {messages.length === 0 ? (
            <div className="welcome-state">
              <div className="welcome-logo">
                <Network size={46} strokeWidth={1.25} />
              </div>

              <h1>PuzzleResearch AI</h1>

              <p>
                你的深度研究伙伴。
                <br />
                输入问题，开始一项全面的研究。
              </p>

              <div className="suggestion-grid">
                {suggestions.map((suggestion) => {
                  const Icon = suggestion.icon

                  return (
                    <button
                      className="suggestion-card"
                      type="button"
                      key={suggestion.label}
                      onClick={() =>
                        chooseSuggestion(suggestion.prompt)
                      }
                    >
                      <span className="suggestion-icon">
                        <Icon size={17} />
                      </span>

                      <span>{suggestion.label}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message ${message.role}`}
                  key={message.id}
                >
                  <div className="message-label">
                    <span>
                      {message.role === "user"
                        ? "你"
                        : "PuzzleResearch"}
                    </span>

                    {message.status && (
                      <small>
                        {messageStatusLabel(message.status)}
                      </small>
                    )}
                  </div>

                  {message.role === "user" &&
                    editingMessageId === message.id && (
                      <form
                        className="message-edit-form"
                        onSubmit={(event) => {
                          event.preventDefault()
                          void resubmitEditedMessage(message)
                        }}
                      >
                        <textarea
                          aria-label="编辑提问内容"
                          autoFocus
                          rows={3}
                          value={editingMessageContent}
                          onChange={(event) =>
                            setEditingMessageContent(event.target.value)
                          }
                        />
                        <div className="message-edit-actions">
                          <button
                            type="button"
                            className="message-action-button"
                            aria-label="取消编辑"
                            title="取消编辑"
                            onClick={cancelEditingMessage}
                          >
                            <X size={15} />
                          </button>
                          <button
                            type="submit"
                            className="message-edit-submit"
                            disabled={running || !editingMessageContent.trim()}
                          >
                            <Check size={14} />
                            重新提交
                          </button>
                        </div>
                      </form>
                    )}

                  {!(
                    message.role === "user" &&
                    editingMessageId === message.id
                  ) && (
                    <>
                  <div className="message-content">
                    {message.role === "user" &&
                      message.attachments &&
                      message.attachments.length > 0 && (
                        <div className="message-image-attachments">
                          {message.attachments.map((image) => (
                            <img
                              key={image.id}
                              src={image.preview}
                              alt={image.name}
                              title={image.name}
                            />
                          ))}
                        </div>
                      )}
                    {message.role === "assistant" ? (
                      <>
                        {message.researchPlan && (
                          <details
                            className="research-plan"
                            open={message.status === "Processing"}
                          >
                            <summary>
                              <span className="research-plan-title">
                                <ListChecks size={14} />
                                <span>研究计划</span>
                              </span>
                              <small>
                                {message.researchPlan.queries.length} 个查询
                              </small>
                            </summary>
                            <div className="research-plan-body">
                              {message.researchPlan.reasoning && (
                                <p className="research-plan-reasoning">
                                  {message.researchPlan.reasoning}
                                </p>
                              )}
                              <ol>
                                {message.researchPlan.queries.map(
                                  (query) => (
                                    <li key={query}>{query}</li>
                                  ),
                                )}
                              </ol>
                            </div>
                          </details>
                        )}

                        {message.taskBoard && (
                          <section className="task-board">
                            <div className="task-board-heading">
                              <span>
                                <ListChecks size={14} />
                                任务清单
                              </span>
                              <small>
                                {message.taskBoard.progress.completed}/
                                {message.taskBoard.progress.total} 已完成
                              </small>
                            </div>
                            <div
                              className="task-board-progress"
                              role="progressbar"
                              aria-label="任务完成进度"
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-valuenow={
                                message.taskBoard.progress.progressPercent
                              }
                            >
                              <span
                                style={{
                                  width: `${message.taskBoard.progress.progressPercent}%`,
                                }}
                              />
                            </div>
                            <div className="task-board-stats">
                              <span>待办 {message.taskBoard.progress.pending}</span>
                              <span>进行中 {message.taskBoard.progress.running}</span>
                              <span>失败 {message.taskBoard.progress.failed}</span>
                            </div>
                            <div className="task-board-sections">
                              {message.taskBoard.sections.map((section) => (
                                <div className="task-board-section" key={section.id}>
                                  <strong>{section.title}</strong>
                                  <ul>
                                    {section.tasks.map((task) => (
                                      <li
                                        className={`task-board-item ${task.status}`}
                                        key={task.id}
                                      >
                                        <span className="task-board-status" />
                                        <span>{task.content}</span>
                                        <small>
                                          {task.status === "completed"
                                            ? "已完成"
                                            : task.status === "running"
                                              ? `${task.progress}%`
                                              : task.status === "failed"
                                                ? "失败"
                                                : task.status === "cancelled"
                                                  ? "已取消"
                                                  : "待办"}
                                        </small>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              ))}
                            </div>
                          </section>
                        )}

                        {message.researchTree && (
                          <div className="research-tree-summary">
                            {(() => {
                              const branchCounts = countResearchTreeNodes(
                                message.researchTree,
                              )
                              return (
                                <>
                            <div className="research-tree-heading">
                              <span>
                                <Network size={14} />
                                研究分支
                              </span>
                              {message.researchTree.epoch !== undefined && (
                                <small>
                                  第 {message.researchTree.epoch} 轮
                                </small>
                              )}
                            </div>
                            <div className="research-tree-stats">
                              <span>
                                <strong>{branchCounts.total}</strong>
                                总分支
                              </span>
                              <span className="completed">
                                <strong>{branchCounts.completed}</strong>
                                已完成
                              </span>
                              <span className="active">
                                <strong>{branchCounts.active}</strong>
                                处理中
                              </span>
                              {branchCounts.pending > 0 && (
                                <span>
                                  <strong>{branchCounts.pending}</strong>
                                  待开始
                                </span>
                              )}
                              {branchCounts.failed > 0 && (
                                <span className="failed">
                                  <strong>{branchCounts.failed}</strong>
                                  失败
                                </span>
                              )}
                            </div>
                                </>
                              )
                            })()}
                            {message.researchTree.nodes.length > 0 && (
                              <details className="research-tree-branches">
                                <summary>
                                  <span>分支详情</span>
                                  <small>{message.researchTree.nodes.length}</small>
                                </summary>
                                <ul>
                                  {message.researchTree.nodes.map((node) => (
                                    <li
                                      className={`research-tree-node ${node.status}`}
                                      key={node.id}
                                      style={{
                                        paddingLeft: `${Math.min(node.depth, 4) * 14}px`,
                                      }}
                                    >
                                      <span className="research-tree-node-dot" />
                                      <span className="research-tree-node-content">
                                        <strong>{node.topic || node.query || "研究分支"}</strong>
                                        <small>
                                          {node.query && node.query !== node.topic
                                            ? `检索：${node.query} · `
                                            : ""}
                                          第 {node.depth} 层 · {node.resultCount} 条发现 · {node.sourceCount} 个来源
                                        </small>
                                        {node.findings.length > 0 && (
                                          <ul className="research-tree-node-findings">
                                            {node.findings.map((finding, index) => (
                                              <li key={`${finding.url || finding.title}-${index}`}>
                                                {finding.url ? (
                                                  <a href={finding.url} target="_blank" rel="noreferrer">
                                                    {finding.title || finding.url}
                                                  </a>
                                                ) : (
                                                  <strong>{finding.title || "研究结果"}</strong>
                                                )}
                                                {finding.snippet && <small>{finding.snippet}</small>}
                                              </li>
                                            ))}
                                          </ul>
                                        )}
                                      </span>
                                      <span className="research-tree-node-status">
                                        {researchNodeStatusLabel(node.status)}
                                      </span>
                                    </li>
                                  ))}
                                </ul>
                              </details>
                            )}
                            {message.researchTree.quality && (
                              <div className="research-tree-quality">
                                {message.researchTree.quality.queryCoverageScore !== undefined && (
                                  <div>
                                    <strong>查询覆盖</strong>
                                    <span>
                                      {Math.round(
                                        message.researchTree.quality.queryCoverageScore * 100,
                                      )}%
                                    </span>
                                    {message.researchTree.quality.dimensionsCovered.length > 0 && (
                                      <small>
                                        已覆盖：
                                        {message.researchTree.quality.dimensionsCovered
                                          .map(researchDimensionLabel)
                                          .join("、")}
                                      </small>
                                    )}
                                    {message.researchTree.quality.dimensionsMissing.length > 0 && (
                                      <small>
                                        待补充：
                                        {message.researchTree.quality.dimensionsMissing
                                          .map(researchDimensionLabel)
                                          .join("、")}
                                      </small>
                                    )}
                                  </div>
                                )}
                                {message.researchTree.quality.freshness && (
                                  <div>
                                    <strong>来源时效</strong>
                                    <span>
                                      近 30 天 {
                                        message.researchTree.quality.freshness.knownCount === 0
                                          ? "—"
                                          : `${Math.round(
                                              (message.researchTree.quality.freshness.fresh30Ratio ?? 0) * 100,
                                            )}%`
                                      }
                                    </span>
                                    <small>
                                      已知日期 {message.researchTree.quality.freshness.knownCount ?? 0}
                                      {" / "}
                                      {message.researchTree.quality.freshness.totalResults ?? 0} 条
                                      {message.researchTree.quality.freshness.unknownCount
                                        ? ` · 日期未知 ${message.researchTree.quality.freshness.unknownCount} 条`
                                        : ""}
                                    </small>
                                  </div>
                                )}
                                {message.researchTree.quality.freshnessWarning && (
                                  <p>
                                    时间敏感主题的近 30 天来源占比较低，建议补充最新来源。
                                  </p>
                                )}
                              </div>
                            )}
                            {message.researchTree.budget && (
                              <div
                                className={`research-tree-budget${
                                  message.researchTree.budget.stopReason
                                    ? " warning"
                                    : ""
                                }`}
                              >
                                <strong>
                                  {researchBudgetReasonLabel(
                                    message.researchTree.budget.stopReason,
                                  )}
                                </strong>
                                <span>
                                  {message.researchTree.budget.tokensUsed !== undefined &&
                                    `令牌 ${message.researchTree.budget.tokensUsed.toLocaleString("zh-CN")}${
                                      message.researchTree.budget.maxTokens
                                        ? ` / ${message.researchTree.budget.maxTokens.toLocaleString("zh-CN")}`
                                        : ""
                                    }`}
                                  {message.researchTree.budget.elapsedSeconds !== undefined &&
                                    ` · ${message.researchTree.budget.elapsedSeconds.toFixed(1)}${
                                      message.researchTree.budget.maxSeconds
                                        ? ` / ${message.researchTree.budget.maxSeconds} 秒`
                                        : " 秒"
                                    }`}
                                  {message.researchTree.budget.searchesUsed !== undefined &&
                                    ` · 检索 ${message.researchTree.budget.searchesUsed} 次`}
                                </span>
                              </div>
                            )}
                          </div>
                        )}

                        {message.researchBudget &&
                          !message.researchTree?.budget && (
                            <div
                              className={`research-tree-budget${
                                message.researchBudget.stopReason
                                  ? " warning"
                                  : ""
                              }`}
                            >
                              <strong>
                                {researchBudgetReasonLabel(
                                  message.researchBudget.stopReason,
                                )}
                              </strong>
                              <span>
                                {message.researchBudget.tokensUsed !== undefined &&
                                  `令牌 ${message.researchBudget.tokensUsed.toLocaleString("zh-CN")}${
                                    message.researchBudget.maxTokens
                                      ? ` / ${message.researchBudget.maxTokens.toLocaleString("zh-CN")}`
                                      : ""
                                  }`}
                                {message.researchBudget.elapsedSeconds !== undefined &&
                                  ` · ${message.researchBudget.elapsedSeconds.toFixed(1)}${
                                    message.researchBudget.maxSeconds
                                      ? ` / ${message.researchBudget.maxSeconds} 秒`
                                      : " 秒"
                                  }`}
                                {message.researchBudget.searchesUsed !== undefined &&
                                  ` · 检索 ${message.researchBudget.searchesUsed} 次`}
                              </span>
                            </div>
                          )}

                        {message.quality && (
                          <div className="research-quality">
                            <div className="research-quality-heading">
                              <span>
                                <BarChart3 size={14} />
                                证据质量
                              </span>
                              <span
                                className={
                                  message.quality.sufficient === false
                                    ? "research-quality-badge warning"
                                    : "research-quality-badge"
                                }
                              >
                                {message.quality.sufficient === false
                                  ? "需要补充"
                                  : "评估完成"}
                              </span>
                            </div>
                            <div className="research-quality-metrics">
                              {message.quality.overallScore !== undefined && (
                                <span>
                                  <strong>
                                    {Math.round(
                                      message.quality.overallScore * 100,
                                    )}%
                                  </strong>
                                  综合质量
                                </span>
                              )}
                              {message.quality.citationCoverage !== undefined && (
                                <span>
                                  <strong>
                                    {Math.round(
                                      message.quality.citationCoverage * 100,
                                    )}%
                                  </strong>
                                  引用覆盖
                                </span>
                              )}
                              {message.quality.sourceCount !== undefined && (
                                <span>
                                  <strong>{message.quality.sourceCount}</strong>
                                  个来源
                                </span>
                              )}
                            </div>
                            {message.quality.summary && (
                              <p>{message.quality.summary}</p>
                            )}
                            {message.quality.missingTopics.length > 0 && (
                              <div className="research-quality-gaps">
                                待补充：{message.quality.missingTopics.join("、")}
                              </div>
                            )}
                          </div>
                        )}

                        {message.steps &&
                          message.steps.length > 0 && (
                            <details
                              className="research-steps"
                              open={
                                message.status ===
                                "Processing"
                              }
                            >
                              <summary>
                                <span className="research-steps-title">
                                  <ListChecks size={14} />
                                  <span>研究过程</span>
                                </span>
                                <small>
                                  {message.steps.length}
                                </small>
                              </summary>
                              <ol>
                                {message.steps.map(
                                  (step) => (
                                    <li key={step}>
                                      {step}
                                    </li>
                                  ),
                                )}
                              </ol>
                            </details>
                          )}
                        {message.toolCalls &&
                          message.toolCalls.length > 0 && (
                            <div className="tool-invocations">
                              <div className="tool-invocations-title">
                                <Wrench size={14} />
                                <span>工具调用</span>
                              </div>
                              {summarizeToolCalls(
                                message.toolCalls,
                              ).map(
                                (toolCall) => (
                                  <div
                                    className="tool-invocation"
                                    key={toolCall.id}
                                  >
                                    <span
                                      className={`tool-status ${toolCall.status}`}
                                    />
                                    <span className="tool-name">
                                      {toolDisplayName(toolCall.name)}
                                      {toolCall.count > 1 &&
                                        ` ×${toolCall.count}`}
                                    </span>
                                    <small>
                                      {toolCall.status ===
                                      "running"
                                        ? "运行中"
                                        : toolCall.status ===
                                            "failed"
                                          ? "失败"
                                          : "已完成"}
                                    </small>
                                    {toolCall.detail && (
                                      <span className="tool-detail">
                                        {toolCall.detail}
                                      </span>
                                    )}
                                  </div>
                                ),
                          )}
                        </div>
                      )}

                        {message.browserActivities &&
                          message.browserActivities.length > 0 && (
                            <div
                              className="browser-activity"
                              aria-label="浏览器活动"
                            >
                              <div className="browser-activity-heading">
                                <Globe2 size={14} />
                                <span>浏览器活动</span>
                                <small>
                                  {message.browserActivities.length}
                                </small>
                              </div>
                              <div className="browser-activity-list">
                                {message.browserActivities.map((activityItem) => (
                                  <div
                                    className="browser-activity-item"
                                    key={activityItem.id}
                                  >
                                    <span className="browser-activity-dot" />
                                    <div className="browser-activity-copy">
                                      <strong>
                                        {toolDisplayName(activityItem.tool)}
                                      </strong>
                                      <span>
                                        {browserActionLabel(activityItem.action)}
                                        {activityItem.title
                                          ? `：${activityItem.title}`
                                          : ""}
                                      </span>
                                      {activityItem.url && (
                                        <a
                                          href={activityItem.url}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          {activityItem.url}
                                        </a>
                                      )}
                                    </div>
                                    {activityItem.screenshotUrl && (
                                      <a
                                        className="browser-activity-preview"
                                        href={resolveApiUrl(
                                          activityItem.screenshotUrl,
                                        )}
                                        target="_blank"
                                        rel="noreferrer"
                                      >
                                        <img
                                          src={resolveApiUrl(
                                            activityItem.screenshotUrl,
                                          )}
                                          alt="浏览器页面截图"
                                          loading="lazy"
                                        />
                                      </a>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                        {!artifactsPanelOpen &&
                          message.generatedArtifacts &&
                          message.generatedArtifacts.length > 0 && (
                            <section
                              className="message-artifacts"
                              aria-label="本轮生成内容"
                            >
                              <div className="message-artifacts-heading">
                                <span>
                                  <FileText size={14} />
                                  本轮生成内容
                                </span>
                                <small>
                                  {message.generatedArtifacts.length} 项
                                </small>
                              </div>
                              <GeneratedArtifactList
                                artifacts={message.generatedArtifacts}
                                canSave={savedArtifactsLoaded}
                                isSaved={(artifact) =>
                                  savedArtifactKeys.has(
                                    savedArtifactSourceKey(
                                      artifact.id,
                                      activeThreadId,
                                    ),
                                  )
                                }
                                onSave={saveGeneratedArtifact}
                              />
                            </section>
                          )}

                        <AssistantMarkdown
                          content={
                            message.content ||
                            (message.status
                              ? "正在生成研究结果…"
                              : "")
                          }
                          sourceCount={
                            evidenceSources.length
                          }
                        />
                      </>
                    ) : (
                      message.content
                    )}
                  </div>
                  {message.role === "assistant" &&
                    message.content.trim() &&
                    message.status !== "Processing" && (
                      <div className="message-actions" aria-label="回答操作">
                        <button
                          type="button"
                          className="message-action-button"
                          aria-label={
                            copiedMessageId === message.id
                              ? "已复制回答"
                              : "复制回答"
                          }
                          title={
                            copiedMessageId === message.id
                              ? "已复制"
                              : "复制回答"
                          }
                          onClick={() => void copyMessage(message)}
                        >
                          {copiedMessageId === message.id ? (
                            <Check size={15} />
                          ) : (
                            <Copy size={15} />
                          )}
                        </button>
                        <button
                          type="button"
                          className="message-action-button"
                          aria-label={
                            savedMessageIds.has(message.id)
                              ? "已保存到资料库"
                              : "保存到资料库"
                          }
                          title={
                            savedMessageIds.has(message.id)
                              ? "已保存到资料库"
                              : "保存到资料库"
                          }
                          disabled={
                            !savedAnswersLoaded ||
                            savedMessageIds.has(message.id)
                          }
                          onClick={() => saveAssistantMessage(message)}
                        >
                          {savedMessageIds.has(message.id) ? (
                            <BookmarkCheck size={15} />
                          ) : (
                            <BookmarkPlus size={15} />
                          )}
                        </button>
                        <button
                          type="button"
                          className="message-action-button"
                          aria-label={
                            speechLoadingMessageId === message.id
                              ? "正在准备朗读"
                              : speakingMessageId === message.id
                                ? "停止朗读"
                                : "朗读回答"
                          }
                          title={
                            speechLoadingMessageId === message.id
                              ? "正在准备朗读"
                              : speakingMessageId === message.id
                                ? "停止朗读"
                                : "朗读回答"
                          }
                          onClick={() => void speakAssistantMessage(message)}
                        >
                          {speechLoadingMessageId === message.id ? (
                            <LoaderCircle
                              className="message-action-spinning"
                              size={15}
                            />
                          ) : speakingMessageId === message.id ? (
                            <VolumeX size={15} />
                          ) : (
                            <Volume2 size={15} />
                          )}
                        </button>
                      </div>
                    )}
                    {message.role === "user" &&
                      message.content.trim() && (
                        <div className="message-actions" aria-label="问题操作">
                          <button
                            type="button"
                            className="message-action-button"
                            aria-label={
                              copiedMessageId === message.id
                                ? "已复制问题"
                                : "复制问题"
                            }
                            title={
                              copiedMessageId === message.id
                                ? "已复制"
                                : "复制问题"
                            }
                            onClick={() => void copyMessage(message)}
                          >
                            {copiedMessageId === message.id ? (
                              <Check size={15} />
                            ) : (
                              <Copy size={15} />
                            )}
                          </button>
                          <button
                            type="button"
                            className="message-action-button"
                            aria-label="编辑并重新提交问题"
                            title="编辑问题"
                            disabled={running}
                            onClick={() => beginEditingMessage(message)}
                          >
                            <Pencil size={15} />
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </article>
              ))}
            </div>
          )}

          {activeThreadId && hasCompletedReport && (
            <section
              className="generated-content"
              aria-label="研究报告"
            >
              <div className="generated-content-heading">
                <span>
                  <FileText size={14} />
                  研究报告
                </span>
                <small>会话级导出</small>
              </div>
              <p>
                当前会话已生成完整报告，可以导出为不同格式保存。
              </p>
              <div className="generated-content-actions">
                <button
                  type="button"
                  className="generated-content-action"
                  onClick={() => void downloadReport("markdown")}
                  disabled={running}
                >
                  <Download size={14} />
                  下载 Markdown
                </button>
                <button
                  type="button"
                  className="generated-content-action"
                  onClick={() => void downloadReport("json")}
                  disabled={running}
                >
                  <Download size={14} />
                  下载 JSON
                </button>
                <button
                  type="button"
                  className="generated-content-action"
                  onClick={() => void downloadReport("html")}
                  disabled={running}
                >
                  <Download size={14} />
                  下载 HTML
                </button>
                <button
                  type="button"
                  className="generated-content-action"
                  onClick={() => void downloadReport("pdf")}
                  disabled={running}
                >
                  <Download size={14} />
                  下载 PDF
                </button>
                <button
                  type="button"
                  className="generated-content-action"
                  onClick={() => void downloadReport("docx")}
                  disabled={running}
                >
                  <Download size={14} />
                  下载 DOCX
                </button>
              </div>
            </section>
          )}

          <EvidencePanel
            evidence={evidenceDetails}
            fallbackSources={evidenceSources}
          />
        </section>

        <section className="prompt-panel">
          {activity && (
            <div className="activity-line">
              <span className="activity-dot" />
              <span>{activity}</span>
            </div>
          )}

          {pendingInterrupt && (
            <section
              className="interrupt-panel"
              aria-label={
                pendingInterrupt.kind === "human_review"
                  ? "最终报告审核"
                  : pendingInterrupt.kind === "draft_review"
                    ? "初稿审核"
                  : pendingInterrupt.kind === "sources_review"
                    ? "来源审核"
                  : pendingInterrupt.kind === "plan_review"
                    ? "研究计划审核"
                  : pendingInterrupt.kind === "tool_approval"
                    ? "工具审批"
                    : "操作确认"
              }
            >
              <div className="interrupt-panel-heading">
                <span>
                  {pendingInterrupt.kind === "tool_approval"
                    ? <Wrench size={15} />
                    : <FileText size={15} />}
                  {pendingInterrupt.kind === "human_review"
                    ? "审核最终报告"
                    : pendingInterrupt.kind === "draft_review"
                      ? "审核研究初稿"
                    : pendingInterrupt.kind === "sources_review"
                      ? "审核研究来源"
                    : pendingInterrupt.kind === "plan_review"
                      ? "审核研究计划"
                    : pendingInterrupt.kind === "tool_approval"
                      ? "需要审批工具调用"
                      : "需要确认操作"}
                </span>
                <small>研究已暂停</small>
              </div>
              <p>{pendingInterrupt.message}</p>
              {pendingInterrupt.kind === "sources_review" && (
                <div
                  className="interrupt-source-snapshot"
                  aria-label="已收集的研究来源"
                >
                  <strong>已收集的研究来源</strong>
                  <pre>{pendingInterrupt.snapshotText || "暂无来源详情"}</pre>
                </div>
              )}
              {pendingInterrupt.kind === "plan_review" && (
                <div className="interrupt-edit-fields">
                  <label>
                    <span>研究查询（每行一条）</span>
                    <textarea
                      className="interrupt-report-editor"
                      aria-label="研究查询，每行一条"
                      rows={Math.min(Math.max(pendingInterrupt.contentText.split(/\r?\n/).length, 4), 10)}
                      value={pendingInterrupt.contentText}
                      disabled={interruptSubmitting}
                      onChange={(event) =>
                        setPendingInterrupt((current) =>
                          current
                            ? { ...current, contentText: event.target.value }
                            : current,
                        )
                      }
                    />
                  </label>
                </div>
              )}
              {pendingInterrupt.kind === "human_review" && (
                <div className="interrupt-edit-fields">
                  <label>
                    <span>审核后的报告</span>
                    <textarea
                      className="interrupt-report-editor"
                      aria-label="审核后的报告"
                      rows={9}
                      value={pendingInterrupt.contentText}
                      disabled={interruptSubmitting}
                      onChange={(event) =>
                        setPendingInterrupt((current) =>
                          current
                            ? { ...current, contentText: event.target.value }
                            : current,
                        )
                      }
                    />
                  </label>
                </div>
              )}
              {pendingInterrupt.kind === "draft_review" && (
                <div className="interrupt-edit-fields">
                  <label>
                    <span>审核后的初稿</span>
                    <textarea
                      className="interrupt-report-editor"
                      aria-label="审核后的初稿"
                      rows={9}
                      value={pendingInterrupt.contentText}
                      disabled={interruptSubmitting}
                      onChange={(event) =>
                        setPendingInterrupt((current) =>
                          current
                            ? { ...current, contentText: event.target.value }
                            : current,
                        )
                      }
                    />
                  </label>
                </div>
              )}
              {pendingInterrupt.kind === "sources_review" && (
                <div className="interrupt-edit-fields">
                  <label>
                    <span>补充写作要求（选填）</span>
                    <textarea
                      aria-label="补充写作要求"
                      rows={4}
                      value={pendingInterrupt.contentText}
                      disabled={interruptSubmitting}
                      onChange={(event) =>
                        setPendingInterrupt((current) =>
                          current
                            ? { ...current, contentText: event.target.value }
                            : current,
                        )
                      }
                    />
                  </label>
                </div>
              )}
              {pendingInterrupt.toolName && (
                <div className="interrupt-edit-fields">
                  <label>
                    <span>工具名称</span>
                    <input
                      value={pendingInterrupt.toolName}
                      list="interrupt-allowed-tools"
                      disabled={interruptSubmitting}
                      onChange={(event) =>
                        setPendingInterrupt((current) =>
                          current
                            ? { ...current, toolName: event.target.value }
                            : current,
                        )
                      }
                    />
                    <datalist id="interrupt-allowed-tools">
                      {pendingInterrupt.allowedTools.map((toolName) => (
                        <option key={toolName} value={toolName} />
                      ))}
                    </datalist>
                    <small>
                      可选工具：{pendingInterrupt.allowedTools.join("、") || "仅限当前工具"}
                    </small>
                  </label>
                  <label>
                    <span>JSON 参数</span>
                    <textarea
                      rows={5}
                      spellCheck={false}
                      value={pendingInterrupt.argumentsText}
                      disabled={interruptSubmitting}
                      onChange={(event) =>
                        setPendingInterrupt((current) =>
                          current
                            ? { ...current, argumentsText: event.target.value }
                            : current,
                        )
                      }
                    />
                  </label>
                </div>
              )}
              <div className="interrupt-panel-actions">
                {pendingInterrupt.availableActions.includes("approve") && (
                  <button
                    className="interrupt-approve-button"
                    type="button"
                    disabled={interruptSubmitting}
                    onClick={() => void respondToInterrupt("approve")}
                  >
                    {pendingInterrupt.kind === "human_review"
                      ? "批准报告"
                      : pendingInterrupt.kind === "draft_review"
                        ? "批准初稿并评估"
                      : pendingInterrupt.kind === "sources_review"
                        ? "来源确认并继续"
                      : pendingInterrupt.kind === "plan_review"
                        ? "批准计划并开始搜索"
                      : "批准并继续"}
                  </button>
                )}
                {pendingInterrupt.availableActions.includes("edit") &&
                  (pendingInterrupt.kind === "human_review" ||
                    pendingInterrupt.kind === "draft_review" ||
                    pendingInterrupt.kind === "sources_review" ||
                    pendingInterrupt.kind === "plan_review" ||
                    pendingInterrupt.toolName) && (
                    <button
                      className="interrupt-edit-button"
                      type="button"
                      disabled={interruptSubmitting}
                      onClick={() => void respondToInterrupt("edit")}
                    >
                      {pendingInterrupt.kind === "human_review"
                        ? "保存修改并继续"
                        : pendingInterrupt.kind === "draft_review"
                          ? "提交修改并评估"
                        : pendingInterrupt.kind === "sources_review"
                          ? "提交要求并继续"
                        : pendingInterrupt.kind === "plan_review"
                          ? "修改计划并开始搜索"
                        : "编辑并执行"}
                    </button>
                  )}
                {pendingInterrupt.availableActions.includes("reject") && (
                  <button
                    className="interrupt-reject-button"
                    type="button"
                    disabled={interruptSubmitting}
                    onClick={() => void respondToInterrupt("reject")}
                  >
                    {pendingInterrupt.kind === "human_review"
                      ? "拒绝报告"
                      : pendingInterrupt.kind === "draft_review"
                        ? "取消本次研究"
                      : pendingInterrupt.kind === "sources_review"
                        ? "取消本次研究"
                      : pendingInterrupt.kind === "plan_review"
                        ? "取消研究"
                      : pendingInterrupt.kind === "tool_approval"
                        ? "拒绝工具"
                        : "拒绝"}
                  </button>
                )}
              </div>
            </section>
          )}

          {commandMenuOpen && (
            <div
              className="prompt-command-menu"
              role="menu"
              aria-label="快捷命令"
            >
              <div className="prompt-command-menu-header">
                <span>快捷命令</span>
                <small>选择后立即执行</small>
              </div>
              <div className="prompt-command-list">
                {promptCommands.map((command) => {
                  const Icon = command.icon

                  return (
                    <button
                      key={command.id}
                      className="prompt-command-item"
                      type="button"
                      role="menuitem"
                      onClick={() => selectPromptCommand(command.id)}
                    >
                      <span className="prompt-command-icon">
                        <Icon size={15} />
                      </span>
                      <span className="prompt-command-copy">
                        <strong>{command.label}</strong>
                        <small>{command.description}</small>
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {mcpMenuOpen && (
            <div
              className="mcp-command-menu"
              role="menu"
              aria-label="MCP 工具服务器"
              style={
                mcpMenuPosition
                  ? {
                      top: mcpMenuPosition.top,
                      left: mcpMenuPosition.left,
                    }
                  : { visibility: "hidden" }
              }
            >
              <div className="mcp-command-menu-header">
                MCP 工具服务器
              </div>
              <div className="mcp-command-list">
                {mcpOptions.length === 0 && (
                  <div className="mcp-command-empty">
                    尚无已启用的服务器，请在设置中配置 MCP。
                  </div>
                )}
                {mcpOptions.map((option) => (
                  <button
                    key={option.id}
                    className={
                      selectedMcp === option.id && chatMode === "mcp"
                        ? "mcp-command-item selected"
                        : "mcp-command-item"
                    }
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setSelectedMcp(option.id)
                      setChatMode("mcp")
                      setMcpMenuOpen(false)
                      setActivity(`MCP 已选择：${option.label}`)
                    }}
                  >
                    <span>{option.label}</span>
                    {selectedMcp === option.id && chatMode === "mcp" && (
                      <Check size={13} />
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="capability-bar">
            <button
              className={
                "capability-button" +
                (chatMode === "web" ? " active" : "")
              }
              type="button"
                aria-pressed={chatMode === "web"}
                onClick={() => {
                  const nextMode =
                    chatMode === "web" ? "direct" : "web"
                  setChatMode(nextMode)
                  setMcpMenuOpen(false)
                  setActivity(
                  nextMode === "web"
                    ? "Web 搜索模式已开启"
                    : "已切换为普通对话模式",
                )
              }}
            >
              <Globe2 size={14} />
              <span>Web</span>
            </button>

            <button
            className={`capability-button ${
                agentMode ? "active" : ""
              }`}
              type="button"
              aria-pressed={agentMode}
              onClick={() => {
                const nextMode =
                  chatMode === "agent" ? "direct" : "agent"
                setChatMode(nextMode)
                setMcpMenuOpen(false)
                setActivity(
                  nextMode === "agent"
                    ? "智能体模式已开启"
                    : "已切换为普通对话模式",
                )
              }}
            >
              <Bot size={14} />
              <span>Agent</span>
            </button>

            <button
            className={`capability-button ${
                deepSearch ? "active" : ""
              }`}
              type="button"
              aria-pressed={deepSearch}
              onClick={() => {
                const nextMode =
                  chatMode === "deep" ? "direct" : "deep"
                setChatMode(nextMode)
                setMcpMenuOpen(false)
                setActivity(
                  nextMode === "deep"
                    ? "深度研究模式已开启"
                    : "已切换为普通对话模式",
                )
              }}
            >
              <Rocket size={14} />
              <span>Deep Research</span>
            </button>

            <button
              className={`capability-button${
                chatMode === "mcp" ? " active mcp-active" : ""
              }`}
              ref={mcpButtonRef}
              type="button"
              aria-pressed={chatMode === "mcp"}
              aria-expanded={mcpMenuOpen}
              onClick={() => {
                if (chatMode === "mcp") {
                  setChatMode("direct")
                  setMcpMenuOpen(false)
                  setActivity("已切换为普通对话模式")
                  return
                }

                setChatMode("mcp")
                setCommandMenuOpen(false)
                setMcpMenuOpen((open) => !open)
                setActivity(`MCP 已选择：${selectedMcpLabel}`)
              }}
            >
              <Plug size={14} />
              <span>{chatMode === "mcp" ? selectedMcpLabel : "MCP"}</span>
              <ChevronDown size={12} />
            </button>
          </div>

          <input
            ref={imageInputRef}
            className="image-file-input"
            type="file"
            accept="image/*"
            multiple
            aria-label="选择图片"
            onChange={(event) => {
              const selected = Array.from(event.target.files ?? [])
              event.target.value = ""
              void addImageFiles(selected)
            }}
          />

          {(attachments.length > 0 || attachmentsLoading) && (
            <div className="prompt-attachments" aria-label="待发送图片">
              {attachments.map((image) => (
                <div className="prompt-attachment" key={image.id}>
                  <img src={image.preview} alt={image.name} />
                  <span title={image.name}>{image.name}</span>
                  <button
                    className="prompt-attachment-remove"
                    type="button"
                    title={`移除 ${image.name}`}
                    aria-label={`移除图片 ${image.name}`}
                    onClick={() =>
                      setAttachments((current) =>
                        current.filter((item) => item.id !== image.id),
                      )
                    }
                  >
                    <X size={13} />
                  </button>
                </div>
              ))}
              {attachmentsLoading && (
                <span className="prompt-attachment-loading">
                  <RefreshCw size={13} className="spin" />
                  正在读取图片
                </span>
              )}
            </div>
          )}

          <form
            className={`prompt-form${
              isDraggingImages ? " file-drop-active" : ""
            }`}
            onDragOver={(event) => {
              if (event.dataTransfer.types.includes("Files")) {
                event.preventDefault()
                setIsDraggingImages(true)
              }
            }}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node)) {
                setIsDraggingImages(false)
              }
            }}
            onDrop={(event) => {
              event.preventDefault()
              setIsDraggingImages(false)
              void addImageFiles(Array.from(event.dataTransfer.files))
            }}
            onSubmit={(event) => {
              event.preventDefault()
              void submitMessage()
            }}
          >
            <button
              className="prompt-icon-button"
              type="button"
              title="添加附件"
              aria-label="添加图片"
              onClick={() => imageInputRef.current?.click()}
            >
              <Paperclip size={17} />
            </button>

            <textarea
              ref={textareaRef}
              className="prompt-input"
              value={draft}
              placeholder="输入你的研究问题…"
              rows={1}
              onChange={(event) => {
                const value = event.target.value
                setDraft(value)
                setCommandMenuOpen(value === "/")
                setMcpMenuOpen(false)
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setCommandMenuOpen(false)
                  setMcpMenuOpen(false)
                  return
                }

                handleKeyDown(event)
              }}
            />

            <div className="prompt-controls">
              <button
                className={`prompt-icon-button${
                  voiceInputState === "recording" ? " voice-recording" : ""
                }`}
                type="button"
                title={
                  voiceInputState === "recording" ||
                  voiceInputState === "recognizing"
                    ? "停止语音输入"
                    : voiceInputState === "processing" ||
                        voiceInputState === "checking"
                      ? "正在识别"
                      : "语音输入"
                }
                aria-label={
                  voiceInputState === "recording" ||
                  voiceInputState === "recognizing"
                    ? "停止语音输入"
                    : "开始语音输入"
                }
                aria-pressed={
                  voiceInputState === "recording" ||
                  voiceInputState === "recognizing"
                }
                disabled={
                  voiceInputState === "processing" ||
                  voiceInputState === "checking"
                }
                onClick={() => void handleVoiceInput()}
              >
                {voiceInputState === "recording" ||
                voiceInputState === "recognizing" ? (
                  <MicOff size={16} />
                ) : voiceInputState === "processing" ||
                  voiceInputState === "checking" ? (
                  <RefreshCw size={15} className="spin" />
                ) : (
                  <Mic size={16} />
                )}
              </button>

              {running ? (
                <button
                  className="prompt-submit stop"
                  type="button"
                  title="停止研究"
                  onClick={stopMessage}
                >
                  <span />
                </button>
              ) : (
                <button
                  className="prompt-submit"
                  type="submit"
                  title="发送消息"
                  disabled={
                    (!draft.trim() && attachments.length === 0) ||
                    attachmentsLoading
                  }
                >
                  <ArrowUp size={16} />
                </button>
              )}
            </div>
          </form>

          <div className="prompt-footer">
            <span>输入 / 查看可用命令</span>
            <span>人工智能也可能出错，重要信息请核实。</span>
          </div>
        </section>

        <button
          className="scroll-button"
          type="button"
          title="滚动到最新消息"
          onClick={scrollToBottom}
        >
          <ArrowDown size={17} />
        </button>
          </>
        )}
        </main>

      {artifactsPanelOpen && generatedArtifacts.length > 0 && (
        <aside
          className="artifacts-panel"
          aria-label="生成制品面板"
        >
          <header className="artifacts-panel-header">
            <div className="artifacts-panel-title">
              <PanelRight size={16} />
              <div>
                <strong>生成制品</strong>
                <small>{generatedArtifacts.length} 项</small>
              </div>
            </div>
            <div className="artifacts-panel-controls">
              <button
                ref={artifactsFullscreenTriggerRef}
                className="artifacts-panel-close"
                type="button"
                title="全屏查看生成制品"
                aria-label="全屏查看生成制品"
                aria-haspopup="dialog"
                onClick={() => setArtifactsFullscreenOpen(true)}
              >
                <Maximize2 size={15} />
              </button>
              <button
                className="artifacts-panel-close"
                type="button"
                title="关闭生成制品"
                aria-label="关闭生成制品"
                onClick={() => setArtifactsPanelOpen(false)}
              >
                <X size={15} />
              </button>
            </div>
          </header>
          <div className="artifacts-panel-list">
            <GeneratedArtifactList
              artifacts={generatedArtifacts}
              canSave={savedArtifactsLoaded}
              isSaved={(artifact) =>
                savedArtifactKeys.has(
                  savedArtifactSourceKey(
                    artifact.id,
                    activeThreadId,
                  ),
                )
              }
              onSave={saveGeneratedArtifact}
            />
          </div>
        </aside>
      )}

      {artifactsFullscreenOpen && generatedArtifacts.length > 0 && (
        <section
          className="artifacts-fullscreen"
          role="dialog"
          aria-modal="true"
          aria-label="生成制品全屏查看"
        >
          <header className="artifacts-fullscreen-header">
            <div className="artifacts-panel-title">
              <PanelRight size={17} />
              <div>
                <strong>生成制品</strong>
                <small>{generatedArtifacts.length} 项</small>
              </div>
            </div>
            <button
              ref={artifactsFullscreenCloseRef}
              className="artifacts-panel-close"
              type="button"
              title="退出全屏"
              aria-label="退出全屏"
              onClick={() => {
                setArtifactsFullscreenOpen(false)
                window.requestAnimationFrame(() =>
                  artifactsFullscreenTriggerRef.current?.focus(),
                )
              }}
            >
              <Minimize2 size={16} />
            </button>
          </header>
          <div className="artifacts-fullscreen-list">
            <GeneratedArtifactList
              artifacts={generatedArtifacts}
              canSave={savedArtifactsLoaded}
              isSaved={(artifact) =>
                savedArtifactKeys.has(
                  savedArtifactSourceKey(
                    artifact.id,
                    activeThreadId,
                  ),
                )
              }
              onSave={saveGeneratedArtifact}
            />
          </div>
        </section>
      )}

      {browserPreviewOpen && activeThreadId && (
        <BrowserPreviewPanel
          key={activeThreadId}
          threadId={activeThreadId}
          eventScreenshots={browserEventScreenshots[activeThreadId] ?? []}
          operation={
            browserOperation?.threadId === activeThreadId
              ? browserOperation
              : null
          }
          onClose={() => setBrowserPreviewOpen(false)}
        />
      )}

      {workspaceFilesOpen && activeThreadId && (
        <WorkspaceFilesPanel
          key={activeThreadId}
          threadId={activeThreadId}
          onClose={() => setWorkspaceFilesOpen(false)}
        />
      )}

      {settingsOpen && (
        <div
          className="settings-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSettingsOpen(false)
            }
          }}
        >
          <section
            className="settings-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <header className="settings-dialog-header">
              <div>
                <span className="settings-eyebrow">控制面板</span>
                <h2 id="settings-title">设置</h2>
              </div>
              <button
                className="settings-close"
                type="button"
                title="关闭设置"
                aria-label="关闭设置"
                onClick={() => setSettingsOpen(false)}
              >
                <X size={17} />
              </button>
            </header>

            <div className="settings-dialog-body">
              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <h3>默认模型</h3>
                    <p>模型列表来自后端公开配置。</p>
                  </div>
                </div>

                <div className="settings-model-list">
                  {modelOptions.map((option) => (
                    <button
                      className={
                        option === model
                          ? "settings-model-option selected"
                          : "settings-model-option"
                      }
                      type="button"
                      key={option}
                      onClick={() => chooseModel(option)}
                    >
                      <span>
                        <strong>{option}</strong>
                        {option === defaultModel && (
                          <small>后端默认</small>
                        )}
                      </span>
                      {option === model && <Check size={16} />}
                    </button>
                  ))}
                </div>

                {modelConfigStatus && (
                  <p className="settings-notice">
                    {modelConfigStatus}
                  </p>
                )}
              </section>

              <SearchHealthPanel />

              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <h3>MCP 扩展工具</h3>
                    <p>
                      配置外部工具服务器。当前只保存配置，不会自动执行未知工具。
                    </p>
                  </div>
                  <button
                    className={`settings-switch ${
                      mcpEnabled ? "active" : ""
                    }`}
                    type="button"
                    role="switch"
                    aria-label="启用 MCP 扩展工具"
                    aria-checked={mcpEnabled}
                    onClick={() => setMcpEnabled((value) => !value)}
                  >
                    <span className="settings-switch-track">
                      <span />
                    </span>
                    <strong>{mcpEnabled ? "已启用" : "已停用"}</strong>
                  </button>
                </div>

                <label className="settings-field">
                  <span>服务器配置（JSON）</span>
                  <textarea
                    className="settings-textarea"
                    value={mcpConfigDraft}
                    spellCheck={false}
                    onChange={(event) =>
                      setMcpConfigDraft(event.target.value)
                    }
                    placeholder={'{\n  "示例服务器": {}\n}'}
                  />
                </label>

                <div className="settings-status">
                  {settingsLoading ? (
                    <RefreshCw size={14} className="spin" />
                  ) : (
                    <CheckCircle2 size={14} />
                  )}
                  <span>已加载工具：{mcpLoadedTools}</span>
                </div>

                <McpDiagnosticsPanel
                  refreshKey={mcpDiagnosticsVersion}
                />

                {settingsNotice && (
                  <p className="settings-notice">{settingsNotice}</p>
                )}
              </section>
            </div>

            <footer className="settings-dialog-footer">
              <button
                className="settings-secondary-button"
                type="button"
                onClick={() => setSettingsOpen(false)}
              >
                关闭
              </button>
              <button
                className="settings-primary-button"
                type="button"
                disabled={settingsLoading}
                onClick={() => void saveMcpSettings()}
              >
                保存配置
              </button>
            </footer>
          </section>
        </div>
      )}
    </div>
  )
}
