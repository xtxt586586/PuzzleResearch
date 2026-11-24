"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { FormEvent, PointerEvent, WheelEvent, KeyboardEvent } from "react"
import {
  ArrowRight,
  Camera,
  Maximize2,
  Minimize2,
  Globe2,
  History,
  LoaderCircle,
  Pause,
  Play,
  X,
} from "lucide-react"

import {
  getApiBaseUrl,
  listThreadScreenshots,
  type StoredScreenshot,
} from "../lib/api"

type BrowserFrameMetadata = {
  url?: string
  page_url?: string
  title?: string
}

type BrowserFrame = {
  data: string
  metadata: BrowserFrameMetadata
}

type BrowserPreviewPanelProps = {
  threadId: string
  eventScreenshots: BrowserEventScreenshot[]
  operation: BrowserPreviewOperation | null
  onClose: () => void
}

export type BrowserPreviewOperation = {
  status: "running" | "error"
  label: string
  progress?: number
  message?: string
}

type BrowserEventScreenshot = {
  id: string
  url: string
  action: string
  pageUrl?: string
  title?: string
  timestamp: number
}

type BrowserHistoryItem = BrowserEventScreenshot & {
  filename?: string
}

type BrowserStreamPacket = {
  type?: unknown
  action?: unknown
  data?: unknown
  metadata?: unknown
  active?: unknown
  message?: unknown
  ok?: unknown
  error?: unknown
}

type RemotePointerButton = "left" | "middle" | "right"

type ActivePointer = {
  pointerId: number
  button: RemotePointerButton
}

const REMOTE_KEYS: Record<string, string> = {
  Backspace: "Backspace",
  Delete: "Delete",
  Enter: "Enter",
  Tab: "Tab",
  ArrowUp: "ArrowUp",
  ArrowDown: "ArrowDown",
  ArrowLeft: "ArrowLeft",
  ArrowRight: "ArrowRight",
  Home: "Home",
  End: "End",
  PageUp: "PageUp",
  PageDown: "PageDown",
  " ": "Space",
}

function getBrowserStreamUrl(threadId: string): string {
  const base = new URL(getApiBaseUrl())
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:"
  base.pathname = `${base.pathname.replace(/\/+$/, "")}/api/browser/${encodeURIComponent(threadId)}/stream`
  base.search = ""
  base.hash = ""
  return base.toString()
}

function mapStoredScreenshot(item: StoredScreenshot): BrowserHistoryItem {
  const timestamp = Date.parse(item.created_at)
  return {
    id: item.filename,
    filename: item.filename,
    url: new URL(item.url, getApiBaseUrl()).toString(),
    action: screenshotActionLabel(item.action || ""),
    pageUrl: item.page_url || undefined,
    title: item.title || undefined,
    timestamp: Number.isFinite(timestamp) ? timestamp : Date.now(),
  }
}

function screenshotActionLabel(action: string): string {
  const normalized = action
    .toLowerCase()
    .replace(/^browser_(?:use_)?/, "")
  const labels: Record<string, string> = {
    manual: "手动截图",
    screenshot: "页面截图",
    navigate: "导航后截图",
    click: "点击后截图",
    type: "输入后截图",
    scroll: "滚动后截图",
    press: "操作后截图",
    capture: "手动截图",
  }
  return labels[normalized] || "浏览器截图"
}

export function BrowserPreviewPanel({
  threadId,
  eventScreenshots,
  operation,
  onClose,
}: BrowserPreviewPanelProps) {
  const [frame, setFrame] = useState<BrowserFrame | null>(null)
  const [viewMode, setViewMode] = useState<"live" | "history">("live")
  const [selectedScreenshotId, setSelectedScreenshotId] = useState<string | null>(null)
  const [storedScreenshots, setStoredScreenshots] = useState<BrowserHistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState("")
  const [expanded, setExpanded] = useState(true)
  const [status, setStatus] = useState("正在连接浏览器…")
  const [error, setError] = useState("")
  const [actionError, setActionError] = useState("")
  const [connected, setConnected] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [interacting, setInteracting] = useState(false)
  const [urlDraft, setUrlDraft] = useState("")
  const [editingUrl, setEditingUrl] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const screenRef = useRef<HTMLDivElement | null>(null)
  const activePointerRef = useRef<ActivePointer | null>(null)
  const lastMoveAtRef = useRef(0)
  const requestIdRef = useRef(0)

  useEffect(() => {
    const controller = new AbortController()
    setHistoryLoading(true)
    setHistoryError("")

    void listThreadScreenshots(threadId, 100, controller.signal)
      .then((items) => {
        const browserItems = items
          .filter((item) => item.action?.toLowerCase().startsWith("browser"))
          .map((item) => mapStoredScreenshot(item))
        setStoredScreenshots(browserItems)
      })
      .catch((loadError) => {
        if (!controller.signal.aborted) {
          setHistoryError(
            loadError instanceof Error
              ? loadError.message
              : "截图记录加载失败。",
          )
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setHistoryLoading(false)
      })

    return () => controller.abort()
  }, [threadId])

  const screenshots = useMemo(() => {
    const byUrl = new Map<string, BrowserHistoryItem>()
    for (const screenshot of [
      ...storedScreenshots,
      ...eventScreenshots.map((item) => ({
        ...item,
        action: screenshotActionLabel(item.action),
      })),
    ]) {
      const existing = byUrl.get(screenshot.url)
      byUrl.set(
        screenshot.url,
        existing
          ? {
              ...existing,
              ...screenshot,
              pageUrl: screenshot.pageUrl || existing.pageUrl,
              title: screenshot.title || existing.title,
            }
          : screenshot,
      )
    }
    return [...byUrl.values()].sort((left, right) => left.timestamp - right.timestamp)
  }, [eventScreenshots, storedScreenshots])

  const selectedScreenshot =
    screenshots.find((screenshot) => screenshot.id === selectedScreenshotId) ??
    screenshots[screenshots.length - 1] ??
    null

  const sendAction = useCallback((payload: Record<string, unknown>) => {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) return false

    requestIdRef.current += 1
    socket.send(
      JSON.stringify({
        ...payload,
        id: `browser-${requestIdRef.current}`,
      }),
    )
    return true
  }, [])

  useEffect(() => {
    let disposed = false
    let reconnectTimer: number | null = null

    setFrame(null)
    setError("")
    setActionError("")
    setConnected(false)
    setStreaming(false)
    setInteracting(false)
    setStatus("正在连接浏览器…")

    const scheduleReconnect = () => {
      if (disposed || reconnectTimer !== null) return
      setConnected(false)
      setStreaming(false)
      setStatus("连接中断，3 秒后重试")
      setError("浏览器连接中断，正在自动重连。")
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        connect()
      }, 3000)
    }

    const connect = () => {
      if (disposed) return
      setStatus("正在连接浏览器…")

      let socket: WebSocket
      try {
        socket = new WebSocket(getBrowserStreamUrl(threadId))
      } catch {
        setError("无法连接浏览器预览服务。")
        scheduleReconnect()
        return
      }
      socketRef.current = socket

      socket.onopen = () => {
        if (disposed || socketRef.current !== socket) return

        setConnected(true)
        setStatus("浏览器已连接")
        setError("")
      }

      socket.onmessage = (event) => {
        if (
          disposed ||
          socketRef.current !== socket ||
          typeof event.data !== "string"
        ) {
          return
        }

        let packet: BrowserStreamPacket

        try {
          packet = JSON.parse(event.data) as BrowserStreamPacket
        } catch {
          return
        }

        if (packet.type === "frame" && typeof packet.data === "string") {
          const metadata =
            packet.metadata !== null &&
            typeof packet.metadata === "object" &&
            !Array.isArray(packet.metadata)
              ? (packet.metadata as BrowserFrameMetadata)
              : {}

          setFrame({ data: packet.data, metadata })
          setStatus((current) =>
            current === "预览已暂停" ? current : "实时预览中",
          )
          setError("")
          return
        }

        if (packet.type === "status") {
          if (packet.message === "started") {
            setStreaming(true)
            setStatus("正在捕获页面")
          } else if (packet.message === "stopped") {
            setStreaming(false)
            setStatus("预览已暂停")
          } else if (packet.message === "浏览器帧流已经运行。") {
            setStreaming(true)
            setStatus("实时预览中")
          } else if (packet.active === false) {
            setStreaming(false)
            setStatus("等待活动浏览器")
          }
          return
        }

        if (packet.type === "ack") {
          if (packet.ok === false) {
            setActionError(String(packet.error ?? "浏览器操作失败。"))
          } else {
            setActionError("")
            if (
              packet.action === "navigate" &&
              packet.metadata !== null &&
              typeof packet.metadata === "object" &&
              !Array.isArray(packet.metadata)
            ) {
              const metadata = packet.metadata as BrowserFrameMetadata
              if (typeof metadata.url === "string") setUrlDraft(metadata.url)
              setEditingUrl(false)
            }
          }
          return
        }

        if (packet.type === "error") {
          setStreaming(false)
          const message = String(packet.message ?? "")
          setError(
            message === "No active browser session"
              ? "当前会话没有可预览的浏览器页面。"
              : message || "浏览器预览启动失败。",
          )
          setStatus("预览不可用")
        }
      }

      socket.onerror = () => {
        if (disposed || socketRef.current !== socket) return
        setConnected(false)
        setStreaming(false)
        setStatus("连接失败")
        setError("浏览器预览连接失败，正在尝试重连。")
      }

      socket.onclose = () => {
        if (disposed || socketRef.current !== socket) return
        socketRef.current = null
        scheduleReconnect()
      }
    }

    connect()

    return () => {
      disposed = true
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer)
      const socket = socketRef.current
      socketRef.current = null
      socket?.close(1000, "Preview closed")
    }
  }, [threadId])

  useEffect(() => {
    if (!connected) return
    sendAction(
      viewMode === "live"
        ? { action: "start", quality: 55, max_fps: 2 }
        : { action: "stop" },
    )
  }, [connected, sendAction, viewMode])

  useEffect(() => {
    if (editingUrl) return
    setUrlDraft(frame?.metadata.url || frame?.metadata.page_url || "")
  }, [editingUrl, frame?.metadata.page_url, frame?.metadata.url])

  const normalizePoint = useCallback((event: PointerEvent<HTMLImageElement>) => {
    const image = event.currentTarget
    const rect = image.getBoundingClientRect()
    if (!rect.width || !rect.height) return null

    const naturalWidth = image.naturalWidth || rect.width
    const naturalHeight = image.naturalHeight || rect.height
    const scale = Math.min(rect.width / naturalWidth, rect.height / naturalHeight)
    const renderedWidth = naturalWidth * scale
    const renderedHeight = naturalHeight * scale
    const left = rect.left + (rect.width - renderedWidth) / 2
    const top = rect.top + (rect.height - renderedHeight) / 2
    const x = (event.clientX - left) / renderedWidth
    const y = (event.clientY - top) / renderedHeight

    if (x < 0 || x > 1 || y < 0 || y > 1) return null
    return { x, y }
  }, [])

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLImageElement>) => {
      if (!connected || !frame || ![0, 1, 2].includes(event.button)) return
      const point = normalizePoint(event)
      if (!point) return

      event.preventDefault()
      const button: RemotePointerButton =
        event.button === 1 ? "middle" : event.button === 2 ? "right" : "left"
      activePointerRef.current = { pointerId: event.pointerId, button }
      event.currentTarget.setPointerCapture(event.pointerId)
      screenRef.current?.focus()
      setInteracting(true)
      sendAction({ action: "mouse", type: "move", ...point })
      sendAction({ action: "mouse", type: "down", button })
    },
    [connected, frame, normalizePoint, sendAction],
  )

  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLImageElement>) => {
      if (!interacting || !connected) return
      const point = normalizePoint(event)
      if (!point) return

      const now = performance.now()
      if (now - lastMoveAtRef.current < 50) return
      lastMoveAtRef.current = now
      sendAction({ action: "mouse", type: "move", ...point })
    },
    [connected, interacting, normalizePoint, sendAction],
  )

  const handlePointerUp = useCallback(
    (event: PointerEvent<HTMLImageElement>) => {
      const activePointer = activePointerRef.current
      if (!activePointer || activePointer.pointerId !== event.pointerId) return
      activePointerRef.current = null
      sendAction({ action: "mouse", type: "up", button: activePointer.button })
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }
    },
    [sendAction],
  )

  const handleWheel = useCallback(
    (event: WheelEvent<HTMLDivElement>) => {
      if (!connected || !interacting) return
      event.preventDefault()
      event.stopPropagation()
      const dx = Math.round(event.deltaX)
      const dy = Math.round(event.deltaY)
      if (dx || dy) sendAction({ action: "scroll", dx, dy })
    },
    [connected, interacting, sendAction],
  )

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (!connected || !interacting || event.nativeEvent.isComposing) return
      if (event.key === "Escape") {
        const activePointer = activePointerRef.current
        if (activePointer) {
          sendAction({ action: "mouse", type: "up", button: activePointer.button })
          activePointerRef.current = null
        }
        setInteracting(false)
        return
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return

      const remoteKey = REMOTE_KEYS[event.key]
      if (remoteKey) {
        event.preventDefault()
        sendAction({ action: "keyboard", type: "press", key: remoteKey })
      } else if (event.key.length === 1) {
        event.preventDefault()
        sendAction({ action: "keyboard", type: "type", text: event.key })
      }
    },
    [connected, interacting, sendAction],
  )

  const handleNavigate = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      const value = urlDraft.trim()
      if (!value || !connected) return
      const url = /^[a-z][a-z\d+.-]*:\/\//i.test(value)
        ? value
        : `https://${value}`
      if (sendAction({ action: "navigate", url })) {
        setEditingUrl(false)
        setViewMode("live")
      }
    },
    [connected, sendAction, urlDraft],
  )

  const pageTitle =
    (viewMode === "history"
      ? selectedScreenshot?.title || selectedScreenshot?.action
      : frame?.metadata.title) || "浏览器页面"
  const pageUrl = viewMode === "history"
    ? selectedScreenshot?.pageUrl || ""
    : frame?.metadata.url || frame?.metadata.page_url || ""
  const safePageUrl = /^https?:\/\//i.test(pageUrl) ? pageUrl : ""
  const operationError =
    operation?.status === "error" ? operation.message : ""
  const visibleError = error || actionError || operationError
  const headerStatus = operation
    ? operation.status === "error"
      ? operation.message || `浏览器操作失败：${operation.label}`
      : `${operation.label}${operation.progress !== undefined ? ` · ${Math.round(operation.progress)}%` : ""}`
    : actionError || status
  const collapsedImage = viewMode === "history"
    ? selectedScreenshot?.url
    : frame
      ? `data:image/jpeg;base64,${frame.data}`
      : undefined

  return (
    <aside
      className={`browser-preview-panel ${viewMode === "history" ? "has-history" : ""} ${expanded ? "" : "is-collapsed"}`}
      aria-label={viewMode === "live" ? "浏览器实时预览" : "浏览器截图记录"}
    >
      <header className="browser-preview-header">
        <div className="browser-preview-heading">
          <Globe2 size={16} />
          <strong>浏览器预览</strong>
          <span
            className={`browser-preview-indicator ${
              visibleError ? "error" : frame ? "live" : "waiting"
            }`}
          />
          {operation?.status === "running" && (
            <LoaderCircle
              size={12}
              className="browser-preview-spinner browser-preview-operation-spinner"
              aria-hidden="true"
            />
          )}
          <small
            className={operation?.status === "error" ? "error" : ""}
            title={headerStatus}
            aria-live="polite"
          >
            {headerStatus}
          </small>
        </div>
        {expanded && (
          <>
            <form
              className="browser-preview-address"
              onSubmit={handleNavigate}
              onFocus={() => setEditingUrl(true)}
              onBlur={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                  setEditingUrl(false)
                }
              }}
            >
              <input
                aria-label="浏览器地址"
                value={urlDraft}
                onChange={(event) => setUrlDraft(event.target.value)}
                placeholder="输入网址"
                disabled={!connected}
              />
              <button
                type="submit"
                title="打开网址"
                aria-label="打开网址"
                disabled={!connected || !urlDraft.trim()}
              >
                <ArrowRight size={14} />
              </button>
            </form>
            <button
              className="browser-preview-mode"
              type="button"
              title={viewMode === "live" ? "查看截图记录" : "返回实时预览"}
              aria-label={viewMode === "live" ? "查看截图记录" : "返回实时预览"}
              aria-pressed={viewMode === "history"}
              onClick={() => setViewMode((mode) => mode === "live" ? "history" : "live")}
            >
              {viewMode === "live" ? <History size={14} /> : <Globe2 size={14} />}
            </button>
            {viewMode === "live" && (
              <div className="browser-preview-controls">
                <button
                  type="button"
                  title={streaming ? "暂停预览" : "恢复预览"}
                  aria-label={streaming ? "暂停预览" : "恢复预览"}
                  disabled={!connected}
                  onClick={() =>
                    sendAction(
                      streaming
                        ? { action: "stop" }
                        : { action: "start", quality: 55, max_fps: 2 },
                    )
                  }
                >
                  {streaming ? <Pause size={14} /> : <Play size={14} />}
                </button>
                <button
                  type="button"
                  title="截取当前画面"
                  aria-label="截取当前画面"
                  disabled={!connected}
                  onClick={() => sendAction({ action: "capture", quality: 85 })}
                >
                  <Camera size={14} />
                </button>
              </div>
            )}
          </>
        )}
        <button
          className="browser-preview-collapse"
          type="button"
          title={expanded ? "收起预览" : "展开预览"}
          aria-label={expanded ? "收起预览" : "展开预览"}
          aria-expanded={expanded}
          onClick={() => {
            const activePointer = activePointerRef.current
            if (expanded && activePointer) {
              sendAction({
                action: "mouse",
                type: "up",
                button: activePointer.button,
              })
              activePointerRef.current = null
            }
            setInteracting(false)
            setExpanded((current) => !current)
          }}
        >
          {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </button>
        <button
          className="browser-preview-close"
          type="button"
          title="关闭浏览器预览"
          aria-label="关闭浏览器预览"
          onClick={onClose}
        >
          <X size={16} />
        </button>
      </header>

      <div
        ref={screenRef}
        className="browser-preview-screen"
        data-interacting={expanded && viewMode === "live" && interacting}
        tabIndex={expanded ? 0 : -1}
        aria-label={viewMode === "live" ? "实时浏览器页面" : "所选浏览器截图"}
        onClick={() => {
          if (!expanded) setExpanded(true)
        }}
        onKeyDown={expanded ? handleKeyDown : undefined}
        onWheel={expanded ? handleWheel : undefined}
      >
        {!expanded ? (
          collapsedImage ? (
            <img src={collapsedImage} alt={pageTitle} draggable={false} />
          ) : (
            <div className="browser-preview-empty">
              <Globe2 size={18} />
              <span>{status}</span>
            </div>
          )
        ) : viewMode === "history" && selectedScreenshot ? (
          <img
            src={selectedScreenshot.url}
            alt={selectedScreenshot.title || selectedScreenshot.action}
            draggable={false}
          />
        ) : viewMode === "live" && frame ? (
          <img
            src={`data:image/jpeg;base64,${frame.data}`}
            alt={pageTitle}
            draggable={false}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onContextMenu={(event) => event.preventDefault()}
          />
        ) : viewMode === "live" && visibleError ? (
          <div className="browser-preview-empty">
            <Globe2 size={22} />
            <span>{visibleError}</span>
          </div>
        ) : viewMode === "history" ? (
          <div className="browser-preview-empty">
            <Camera size={22} />
            <span>
              {historyLoading
                ? "正在加载截图记录…"
                : historyError || "当前会话还没有浏览器截图。"}
            </span>
          </div>
        ) : (
          <div className="browser-preview-empty">
            <LoaderCircle
              size={20}
              className="browser-preview-spinner"
            />
            <span>{status}</span>
          </div>
        )}
      </div>

      {expanded && viewMode === "history" && (
        <div className="browser-preview-history" aria-label="截图时间线">
          {screenshots.map((screenshot) => (
            <button
              key={screenshot.id}
              type="button"
              className={screenshot.id === selectedScreenshot?.id ? "selected" : ""}
              title={`${screenshot.action} · ${new Date(screenshot.timestamp).toLocaleString()}`}
              aria-label={`查看${screenshot.action}截图，${new Date(screenshot.timestamp).toLocaleString()}`}
              aria-pressed={screenshot.id === selectedScreenshot?.id}
              onClick={() => setSelectedScreenshotId(screenshot.id)}
            >
              <img src={screenshot.url} alt="" loading="lazy" />
            </button>
          ))}
          <span>
            {screenshots.length > 0
              ? `${screenshots.length} 张截图`
              : historyLoading
                ? "正在读取…"
                : "暂无记录"}
          </span>
        </div>
      )}

      {expanded && (
        <footer className="browser-preview-page">
          <strong title={pageTitle}>{pageTitle}</strong>
          {safePageUrl && (
            <a href={safePageUrl} target="_blank" rel="noreferrer">
              {safePageUrl}
            </a>
          )}
        </footer>
      )}
    </aside>
  )
}
