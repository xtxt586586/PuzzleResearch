"use client"

import { RefreshCw, Server } from "lucide-react"
import { useCallback, useEffect, useState } from "react"

import {
  getMcpDiagnostics,
  type McpDiagnostics,
} from "../lib/api"

const overallLabels: Record<string, string> = {
  disabled: "MCP 已停用",
  not_configured: "尚未配置服务器",
  ready: "全部服务器已连接",
  degraded: "部分服务器需要处理",
}

const serverLabels: Record<string, string> = {
  connected: "已连接",
  failed: "连接失败",
  disconnected: "已断开",
  not_loaded: "未加载",
  disabled: "已停用",
}

const transportLabels: Record<string, string> = {
  stdio: "本地进程",
  sse: "SSE",
  unknown: "未知传输方式",
}

export function McpDiagnosticsPanel({
  refreshKey,
}: {
  refreshKey: number
}) {
  const [diagnostics, setDiagnostics] =
    useState<McpDiagnostics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const refresh = useCallback(async () => {
    setLoading(true)
    setError("")
    try {
      setDiagnostics(await getMcpDiagnostics())
    } catch (cause) {
      setError(
        cause instanceof TypeError
          ? "无法连接后端服务，请检查服务状态。"
          : cause instanceof Error
            ? cause.message
            : "MCP 诊断读取失败。",
      )
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh, refreshKey])

  return (
    <section className="mcp-diagnostics" aria-label="MCP 服务器诊断">
      <header className="mcp-diagnostics-heading">
        <div>
          <h4>服务器连接状态</h4>
          <p>状态由后端连接和工具发现结果生成。</p>
        </div>
        <button
          className="mcp-diagnostics-refresh"
          type="button"
          title="刷新连接状态"
          aria-label="刷新 MCP 连接状态"
          onClick={() => void refresh()}
          disabled={loading}
        >
          <RefreshCw size={14} className={loading ? "spin" : ""} />
        </button>
      </header>

      {error && <p className="mcp-diagnostics-error" role="alert">{error}</p>}
      {loading && !diagnostics && (
        <p className="mcp-diagnostics-loading" role="status">正在读取连接状态…</p>
      )}

      {diagnostics && (
        <>
          <div className={`mcp-diagnostics-summary ${diagnostics.status}`} role="status">
            <span className="mcp-status-dot" />
            <strong>{overallLabels[diagnostics.status] ?? "状态未知"}</strong>
            <span>{diagnostics.connected_count}/{diagnostics.server_count} 台已连接</span>
          </div>

          {diagnostics.servers.length === 0 ? (
            <p className="mcp-diagnostics-empty">
              <Server size={14} />
              配置并保存服务器后，可在这里查看连接状态。
            </p>
          ) : (
            <ul className="mcp-diagnostics-list">
              {diagnostics.servers.map((server) => (
                <li className="mcp-diagnostics-row" key={server.server_id}>
                  <div className="mcp-diagnostics-server">
                    <strong title={server.server_id}>{server.server_id}</strong>
                    <span>{transportLabels[server.transport] ?? server.transport}</span>
                  </div>
                  <div className="mcp-diagnostics-state">
                    <span className={`mcp-server-status ${server.status}`}>
                      {serverLabels[server.status] ?? "状态未知"}
                    </span>
                    <span>工具 {server.loaded_tools}</span>
                  </div>
                  {server.message && (
                    <p className="mcp-diagnostics-message">{server.message}</p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  )
}
