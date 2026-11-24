"use client"

import { useEffect, useRef, useState } from "react"
import { Download, FileText, FolderOpen, LoaderCircle, RefreshCw, Trash2, Upload, X } from "lucide-react"

import {
  deleteWorkspaceFile,
  downloadWorkspaceFile,
  listWorkspaceFiles,
  uploadWorkspaceFile,
} from "../lib/workspace-api"


type WorkspaceFilesPanelProps = {
  threadId: string
  onClose: () => void
}


export function WorkspaceFilesPanel({
  threadId,
  onClose,
}: WorkspaceFilesPanelProps) {
  const [files, setFiles] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [refreshToken, setRefreshToken] = useState(0)
  const [downloadingPath, setDownloadingPath] = useState<string | null>(null)
  const [deletingPath, setDeletingPath] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [notice, setNotice] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let disposed = false
    setLoading(true)
    setError("")

    void listWorkspaceFiles(threadId)
      .then((result) => {
        if (!disposed) setFiles(result.files)
      })
      .catch((reason: unknown) => {
        if (!disposed) {
          setError(
            reason instanceof Error
              ? reason.message
              : "无法读取会话文件。",
          )
        }
      })
      .finally(() => {
        if (!disposed) setLoading(false)
      })

    return () => {
      disposed = true
    }
  }, [threadId, refreshToken])

  async function download(path: string) {
    setDownloadingPath(path)
    setError("")
    try {
      const blob = await downloadWorkspaceFile(threadId, path)
      const objectUrl = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = objectUrl
      link.download = path.split("/").pop() || "workspace-file"
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000)
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "文件下载失败。",
      )
    } finally {
      setDownloadingPath(null)
    }
  }

  async function upload(file: File) {
    setUploading(true)
    setError("")
    setNotice("")
    try {
      const result = await uploadWorkspaceFile(threadId, file)
      setNotice(`已上传：${result.path}`)
      setRefreshToken((value) => value + 1)
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "文件上传失败。",
      )
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  async function remove(path: string) {
    if (!window.confirm(`确定删除工作区文件“${path}”？此操作无法撤销。`)) return
    setDeletingPath(path)
    setError("")
    setNotice("")
    try {
      await deleteWorkspaceFile(threadId, path)
      setNotice(`已删除：${path}`)
      setRefreshToken((value) => value + 1)
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "文件删除失败。",
      )
    } finally {
      setDeletingPath(null)
    }
  }

  return (
    <aside className="workspace-files-panel" aria-label="会话工作区文件">
      <header className="workspace-files-header">
        <div className="workspace-files-title">
          <FolderOpen size={16} />
          <strong>工作区文件</strong>
          {!loading && <small>{files.length}</small>}
        </div>
        <div className="workspace-files-actions">
          <input
            ref={fileInputRef}
            className="workspace-files-input"
            type="file"
            aria-label="选择要上传的工作区文件"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0]
              if (file) void upload(file)
            }}
          />
          <button
            className="workspace-files-icon-button"
            type="button"
            title="上传工作区文件"
            aria-label="上传工作区文件"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {uploading ? <LoaderCircle size={14} className="spin" /> : <Upload size={14} />}
          </button>
          <button
            className="workspace-files-icon-button"
            type="button"
            title="刷新文件列表"
            aria-label="刷新文件列表"
            disabled={loading}
            onClick={() => setRefreshToken((value) => value + 1)}
          >
            <RefreshCw size={14} className={loading ? "spin" : ""} />
          </button>
          <button
            className="workspace-files-icon-button"
            type="button"
            title="关闭工作区文件"
            aria-label="关闭工作区文件"
            onClick={onClose}
          >
            <X size={15} />
          </button>
        </div>
      </header>

      {error && <p className="workspace-files-error" role="alert">{error}</p>}
      {notice && <p className="workspace-files-success" role="status">{notice}</p>}

      <div className="workspace-files-list" aria-live="polite">
        {loading ? (
          <div className="workspace-files-empty">
            <LoaderCircle size={18} className="spin" />
            <span>正在读取文件…</span>
          </div>
        ) : files.length === 0 ? (
          <div className="workspace-files-empty">
            <FolderOpen size={20} />
            <span>暂无工作区文件</span>
          </div>
        ) : (
          files.map((path) => (
            <div className="workspace-file-row" key={path}>
              <FileText size={15} />
              <span className="workspace-file-path" title={path}>{path}</span>
              <button
                className="workspace-files-icon-button"
                type="button"
                title={`下载 ${path}`}
                aria-label={`下载 ${path}`}
                disabled={downloadingPath !== null || deletingPath !== null}
                onClick={() => void download(path)}
              >
                {downloadingPath === path ? (
                  <LoaderCircle size={14} className="spin" />
                ) : (
                  <Download size={14} />
                )}
              </button>
              <button
                className="workspace-files-icon-button workspace-files-delete-button"
                type="button"
                title={`删除 ${path}`}
                aria-label={`删除 ${path}`}
                disabled={downloadingPath !== null || deletingPath !== null}
                onClick={() => void remove(path)}
              >
                {deletingPath === path ? (
                  <LoaderCircle size={14} className="spin" />
                ) : (
                  <Trash2 size={14} />
                )}
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  )
}
