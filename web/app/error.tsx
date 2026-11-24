"use client"

import { useEffect } from "react"
import { AlertTriangle, RefreshCw, RotateCw } from "lucide-react"

type AppError = Error & { digest?: string }

export default function ErrorPage({
  error,
  retry,
}: {
  error: AppError
  retry: () => void
}) {
  useEffect(() => {
    console.error("PuzzleResearch 页面渲染失败", error)
  }, [error])

  return (
    <main className="app-error-page" role="alert">
      <div className="app-error-icon" aria-hidden="true">
        <AlertTriangle size={24} strokeWidth={1.8} />
      </div>
      <h1>页面暂时无法显示</h1>
      <p>发生了意外错误。你可以重试，或刷新页面后继续使用。</p>
      {error.digest ? (
        <p className="app-error-digest">诊断编号：{error.digest}</p>
      ) : null}
      <div className="app-error-actions">
        <button
          className="app-error-button app-error-button-primary"
          onClick={retry}
          type="button"
        >
          <RotateCw size={15} aria-hidden="true" />
          重试
        </button>
        <button
          className="app-error-button app-error-button-secondary"
          onClick={() => window.location.reload()}
          type="button"
        >
          <RefreshCw size={15} aria-hidden="true" />
          刷新页面
        </button>
      </div>
    </main>
  )
}
