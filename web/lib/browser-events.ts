export type BrowserOperationUpdate =
  | {
      kind: "running"
      label: string
      progress?: number
    }
  | { kind: "finished" | "reset" }
  | {
      kind: "error"
      label: string
      message: string
    }

const TOOL_LABELS: Record<string, string> = {
  browser_use: "网页自动化",
  browser_search: "浏览器搜索",
  browser_navigate: "浏览器导航",
  browser_click: "浏览器点击",
  browser_back: "返回上一页",
  browser_extract_text: "提取网页文本",
  browser_list_links: "列出网页链接",
  browser_reset: "重置浏览器",
  browser_screenshot: "网页截图",
  sandbox_web_search: "网页搜索",
}

const ACTION_LABELS: Record<string, string> = {
  go_to_url: "正在打开网页",
  navigate: "正在打开网页",
  navigate_completed: "网页已打开",
  go_back: "正在返回上一页",
  input_text: "正在填写网页内容",
  scroll_to_text: "正在定位网页内容",
  search: "正在搜索网页",
  search_completed: "网页搜索已完成",
  click: "正在点击网页元素",
  click_completed: "网页点击已完成",
  screenshot: "正在截取网页",
  open_tab: "正在打开新标签页",
  close_tab: "正在关闭标签页",
  switch_tab: "正在切换标签页",
  scroll_down: "正在向下浏览",
  scroll_up: "正在向上浏览",
}

function eventToolName(data: Record<string, unknown>): string {
  return String(
    data.tool_name ?? data.tool ?? data.name ?? "",
  )
    .trim()
    .toLowerCase()
}

function isBrowserRelatedTool(tool: string): boolean {
  return (
    tool === "browser_use" ||
    tool === "browser" ||
    tool.startsWith("browser_") ||
    tool.startsWith("sb_browser") ||
    tool === "sandbox_web_search" ||
    tool.startsWith("sandbox_search") ||
    tool.startsWith("sandbox_extract_search")
  )
}

function toolLabel(tool: string): string {
  return TOOL_LABELS[tool] ?? tool.replace(/[_-]+/g, " ")
}

export function getBrowserOperationUpdate(
  eventType: string,
  data: Record<string, unknown>,
): BrowserOperationUpdate | null {
  if (["done", "cancelled", "error"].includes(eventType)) {
    return { kind: "reset" }
  }

  const tool = eventToolName(data)
  if (!isBrowserRelatedTool(tool)) return null

  if (["tool_end", "tool_result"].includes(eventType)) {
    return { kind: "finished" }
  }

  if (eventType === "tool_error") {
    return {
      kind: "error",
      label: toolLabel(tool),
      message: String(data.error ?? data.message ?? "").trim(),
    }
  }

  if (
    !["tool_start", "tool_started", "tool_progress"].includes(eventType)
  ) {
    return null
  }

  const action = typeof data.action === "string" ? data.action.trim() : ""
  const message = [data.message, data.info]
    .find((value) => typeof value === "string" && value.trim())
  const progressValue = Number(data.progress)

  return {
    kind: "running",
    label:
      (typeof message === "string" ? message.trim() : "") ||
      ACTION_LABELS[action.toLowerCase()] ||
      toolLabel(tool),
    ...(Number.isFinite(progressValue)
      ? { progress: Math.max(0, Math.min(100, progressValue)) }
      : {}),
  }
}
