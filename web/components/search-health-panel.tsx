"use client"

import { useCallback, useEffect, useState } from "react"
import { CheckCircle2, CircleAlert, RefreshCw, RotateCcw, Trash2 } from "lucide-react"

import {
  clearSearchCache,
  getAgentHealthStatus,
  getSearchCacheStats,
  getSearchProviderStatus,
  resetSearchProviderStatus,
  type RagEmbeddingStatus,
  type SearchCacheStats,
  type SearchProviderStatus,
} from "../lib/api"

const emptyCache: SearchCacheStats = {
  entries: 0,
  hits: 0,
  misses: 0,
  hit_rate: 0,
}

export function SearchHealthPanel() {
  const [providers, setProviders] = useState<SearchProviderStatus[]>([])
  const [embedding, setEmbedding] = useState<RagEmbeddingStatus | null>(null)
  const [cache, setCache] = useState(emptyCache)
  const [loading, setLoading] = useState(true)
  const [clearing, setClearing] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [notice, setNotice] = useState("")

  const refresh = useCallback(async () => {
    setLoading(true)
    setNotice("")
    try {
      const [nextProviders, nextCache, nextHealth] = await Promise.all([
        getSearchProviderStatus(),
        getSearchCacheStats(),
        getAgentHealthStatus(),
      ])
      setProviders(nextProviders)
      setCache(nextCache)
      setEmbedding(nextHealth.rag_embedding)
    } catch {
      setNotice("搜索状态读取失败，请确认后端服务正常。")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  async function handleClearCache() {
    setClearing(true)
    setNotice("")
    try {
      const cleared = await clearSearchCache()
      setNotice(`已清理 ${cleared} 条缓存。`)
      await refresh()
    } catch {
      setNotice("缓存清理失败，请稍后重试。")
    } finally {
      setClearing(false)
    }
  }

  async function handleResetProviders() {
    setResetting(true)
    setNotice("")
    try {
      await resetSearchProviderStatus()
      setNotice("Provider 状态已重置。")
      await refresh()
    } catch {
      setNotice("Provider 状态重置失败，请稍后重试。")
    } finally {
      setResetting(false)
    }
  }

  return (
    <section className="settings-section search-health-section">
      <div className="settings-section-heading">
        <div>
          <h3>搜索服务</h3>
          <p>查看 Provider 可用状态与运行统计。</p>
        </div>
        <button
          className="settings-icon-button"
          type="button"
          title="刷新搜索状态"
          aria-label="刷新搜索状态"
          disabled={loading}
          onClick={() => void refresh()}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} />
        </button>
      </div>

      <div className="search-provider-list" aria-live="polite">
        {providers.map((provider) => (
          <div
            className={`search-provider-row ${provider.healthy ? "" : "unavailable"}`}
            key={provider.name}
          >
            <div className="search-provider-name">
              {provider.healthy ? (
                <CheckCircle2 size={15} aria-hidden="true" />
              ) : (
                <CircleAlert size={15} aria-hidden="true" />
              )}
              <strong>{provider.name}</strong>
              <span>{provider.configured ? "已配置" : "未启用"}</span>
            </div>
            <div className="search-provider-stats">
              <span>{provider.healthy ? "可用" : "不可用"}</span>
              <span>调用 {provider.calls}</span>
              <span>成功 {provider.successes}</span>
              <span>失败 {provider.failures}</span>
              {typeof provider.success_rate === "number" && (
                <span>成功率 {Math.round(provider.success_rate * 100)}%</span>
              )}
              {typeof provider.avg_latency_ms === "number" && provider.avg_latency_ms > 0 && (
                <span>延迟 {Math.round(provider.avg_latency_ms)}ms</span>
              )}
              {provider.circuit?.is_open && <span>已熔断</span>}
            </div>
            {!provider.healthy && provider.reason && (
              <p className="search-provider-reason">{provider.reason}</p>
            )}
            {provider.last_error && (
              <p className="search-provider-reason">最近错误：{provider.last_error}</p>
            )}
          </div>
        ))}
        {!loading && providers.length === 0 && (
          <p className="search-provider-reason">没有可显示的搜索 Provider。</p>
        )}
      </div>

      <div className="search-embedding-row" aria-live="polite">
        <div
          className={`search-embedding-heading ${
            embedding && !embedding.available ? "unavailable" : ""
          }`}
        >
          {embedding?.available ? (
            <CheckCircle2 size={15} aria-hidden="true" />
          ) : (
            <CircleAlert size={15} aria-hidden="true" />
          )}
          <strong>知识库嵌入</strong>
          <span>
            {!embedding ? "读取中" : embedding.available ? "可用" : "不可用"}
          </span>
        </div>
        <div className="search-provider-stats">
          <span>
            提供商：{embedding ? embeddingProviderLabel(embedding.provider) : "读取中"}
          </span>
          {embedding?.dimensions !== null && embedding?.dimensions !== undefined && (
            <span>向量维度 {embedding.dimensions}</span>
          )}
        </div>
        {!embedding?.available && embedding?.reason && (
          <p className="search-provider-reason">{embedding.reason}</p>
        )}
      </div>

      <div className="search-cache-row">
        <div>
          <strong>搜索缓存</strong>
          <span>
            {cache.entries} 条 · 命中 {cache.hits} · 未命中 {cache.misses} · 命中率 {Math.round(cache.hit_rate * 100)}%
          </span>
        </div>
        <button
          className="settings-secondary-button search-cache-clear"
          type="button"
          title="清理搜索缓存"
          disabled={clearing || loading || cache.entries === 0}
          onClick={() => void handleClearCache()}
        >
          {clearing ? <RefreshCw size={14} className="spin" /> : <Trash2 size={14} />}
          清理
        </button>
      </div>

      <div className="search-health-actions">
        <button
          className="settings-secondary-button"
          type="button"
          title="重置 Provider 统计与熔断状态"
          disabled={resetting || loading}
          onClick={() => void handleResetProviders()}
        >
          {resetting ? <RefreshCw size={14} className="spin" /> : <RotateCcw size={14} />}
          重置状态
        </button>
      </div>

      {notice && <p className="settings-notice">{notice}</p>}
    </section>
  )
}

function embeddingProviderLabel(provider: string): string {
  if (provider === "local-hash") {
    return "本地哈希嵌入"
  }
  if (provider === "openai-compatible") {
    return "OpenAI 兼容服务"
  }
  return provider || "未知"
}
