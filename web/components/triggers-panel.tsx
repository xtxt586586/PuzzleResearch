"use client"

import {
  CalendarClock,
  Copy,
  History,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Trash2,
  Webhook,
  Workflow,
  X,
} from "lucide-react"
import {
  useCallback,
  useEffect,
  useState,
  type FormEvent,
} from "react"

import {
  createEventTrigger,
  createScheduledTrigger,
  createWebhookTrigger,
  deleteTrigger,
  getTriggerExecutions,
  getWebhookUrl,
  listTriggers,
  pauseTrigger,
  resumeTrigger,
  type Trigger,
  type TriggerExecution,
  type TriggerStatus,
  type TriggerType,
} from "../lib/triggers-api"

type TriggerKindFilter = TriggerType | "all"
type TriggerStatusFilter = TriggerStatus | "all"

const typeLabels: Record<TriggerType, string> = {
  scheduled: "定时",
  webhook: "Webhook",
  event: "事件",
}

const statusLabels: Record<TriggerStatus, string> = {
  active: "运行中",
  paused: "已暂停",
  disabled: "已停用",
  error: "异常",
}

function formatDate(
  value: string | null | undefined,
  timezone = "Asia/Shanghai",
): string {
  if (!value) return "暂无"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return "暂无"
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: timezone,
    }).format(date)
  } catch {
    return date.toLocaleString("zh-CN")
  }
}

function triggerSummary(trigger: Trigger): string {
  if (trigger.trigger_type === "scheduled") {
    return `Cron ${trigger.schedule ?? "未设置"} · ${trigger.timezone ?? "UTC"} · 下次 ${formatDate(trigger.next_run_at, trigger.timezone)}`
  }
  if (trigger.trigger_type === "webhook") {
    return `${trigger.endpoint_path ?? `/api/webhook/${trigger.id}`} · ${trigger.require_auth ? "需要令牌" : "无需令牌"}`
  }
  return `${trigger.event_type ?? "未设置事件"}${trigger.event_source ? ` · 来源 ${trigger.event_source}` : ""}`
}

export function TriggersPanel() {
  const [triggers, setTriggers] = useState<Trigger[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [kindFilter, setKindFilter] = useState<TriggerKindFilter>("all")
  const [statusFilter, setStatusFilter] = useState<TriggerStatusFilter>("all")
  const [formKind, setFormKind] = useState<TriggerType>("scheduled")
  const [name, setName] = useState("")
  const [task, setTask] = useState("")
  const [schedule, setSchedule] = useState("0 9 * * 1-5")
  const [timezone, setTimezone] = useState("Asia/Shanghai")
  const [runImmediately, setRunImmediately] = useState(false)
  const [requireAuth, setRequireAuth] = useState(true)
  const [eventType, setEventType] = useState("")
  const [eventSource, setEventSource] = useState("")
  const [eventFilters, setEventFilters] = useState("{}")
  const [error, setError] = useState("")
  const [notice, setNotice] = useState("")
  const [oneTimeWebhook, setOneTimeWebhook] = useState<{
    url: string
    token?: string
  } | null>(null)
  const [pendingAction, setPendingAction] = useState("")
  const [expandedTrigger, setExpandedTrigger] = useState<string | null>(null)
  const [executions, setExecutions] = useState<
    Record<string, TriggerExecution[]>
  >({})

  const loadTriggers = useCallback(async () => {
    setLoading(true)
    setError("")
    try {
      setTriggers(await listTriggers({
        triggerType: kindFilter,
        status: statusFilter,
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "自动化列表读取失败。")
    } finally {
      setLoading(false)
    }
  }, [kindFilter, statusFilter])

  useEffect(() => {
    void loadTriggers()
  }, [loadTriggers])

  async function handleCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSaving(true)
    setError("")
    setNotice("")
    setOneTimeWebhook(null)

    try {
      const shared = {
        name: name.trim(),
        task: task.trim(),
        agent_id: "default",
        task_params: {},
      }
      if (formKind === "scheduled") {
        await createScheduledTrigger({
          ...shared,
          schedule: schedule.trim(),
          timezone,
          run_immediately: runImmediately,
        })
      } else if (formKind === "webhook") {
        const created = await createWebhookTrigger({
          ...shared,
          require_auth: requireAuth,
          http_methods: ["POST"],
        })
        setOneTimeWebhook({
          url: getWebhookUrl(created.trigger.endpoint_path ?? `/api/webhook/${created.trigger_id}`),
          token: created.auth_token,
        })
      } else {
        let filters: Record<string, unknown>
        try {
          const parsed: unknown = JSON.parse(eventFilters)
          if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("事件过滤条件必须是 JSON 对象。")
          }
          filters = parsed as Record<string, unknown>
        } catch (cause) {
          throw cause instanceof Error
            ? cause
            : new Error("事件过滤条件不是有效 JSON。")
        }
        await createEventTrigger({
          ...shared,
          event_type: eventType.trim(),
          ...(eventSource.trim() ? { event_source: eventSource.trim() } : {}),
          event_filters: filters,
        })
      }
      setName("")
      setTask("")
      setEventType("")
      setEventSource("")
      setEventFilters("{}")
      setNotice("自动化已创建。")
      await loadTriggers()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "自动化创建失败。")
    } finally {
      setSaving(false)
    }
  }

  async function handleAction(
    trigger: Trigger,
    action: "pause" | "resume" | "delete" | "executions",
  ) {
    const key = `${action}:${trigger.id}`
    if (action === "delete" && !window.confirm(`删除自动化“${trigger.name}”？此操作会同时删除其管理记录。`)) {
      return
    }
    setPendingAction(key)
    setError("")
    setNotice("")
    try {
      if (action === "pause") {
        await pauseTrigger(trigger.id)
        setNotice("自动化已暂停。")
        await loadTriggers()
      } else if (action === "resume") {
        await resumeTrigger(trigger.id)
        setNotice("自动化已恢复。")
        await loadTriggers()
      } else if (action === "delete") {
        await deleteTrigger(trigger.id)
        setExpandedTrigger(null)
        setNotice("自动化已删除。")
        await loadTriggers()
      } else if (expandedTrigger === trigger.id) {
        setExpandedTrigger(null)
      } else {
        setExpandedTrigger(trigger.id)
        setExecutions((current) => ({ ...current, [trigger.id]: [] }))
        const records = await getTriggerExecutions(trigger.id)
        setExecutions((current) => ({ ...current, [trigger.id]: records }))
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "操作未能完成。")
    } finally {
      setPendingAction("")
    }
  }

  async function copyValue(label: string, value: string) {
    try {
      await navigator.clipboard.writeText(value)
      setNotice(`${label}已复制。`)
    } catch {
      setError("复制失败，请手动选择并复制内容。")
    }
  }

  return (
    <section className="triggers-view">
      <div className="triggers-content">
        <header className="triggers-heading">
          <div>
            <span className="triggers-eyebrow">工作区</span>
            <h1><Workflow size={24} />自动化</h1>
            <p>管理定时任务、Webhook 与事件触发流程。</p>
          </div>
          <button
            className="trigger-refresh-button"
            type="button"
            onClick={() => void loadTriggers()}
            disabled={loading}
            aria-label="刷新自动化列表"
            title="刷新"
          >
            <RefreshCw size={16} />
          </button>
        </header>

        {error && <div className="trigger-message error" role="alert">{error}</div>}
        {notice && <div className="trigger-message notice" role="status">{notice}</div>}

        {oneTimeWebhook && (
          <section className="trigger-secret" aria-label="Webhook 调用信息">
            <div className="trigger-secret-heading">
              <div>
                <strong>Webhook 已创建</strong>
                <p>令牌只在本次页面中显示。请立即保存，关闭此页面后无法再次读取。</p>
              </div>
              <button
                className="trigger-icon-button"
                type="button"
                aria-label="关闭 Webhook 信息"
                title="关闭"
                onClick={() => setOneTimeWebhook(null)}
              ><X size={16} /></button>
            </div>
            <div className="trigger-secret-value">
              <span>调用地址</span>
              <code>{oneTimeWebhook.url}</code>
              <button type="button" className="trigger-icon-button" aria-label="复制 Webhook 地址" title="复制地址" onClick={() => void copyValue("Webhook 地址", oneTimeWebhook.url)}><Copy size={15} /></button>
            </div>
            {oneTimeWebhook.token && (
              <div className="trigger-secret-value">
                <span>认证令牌</span>
                <code>{oneTimeWebhook.token}</code>
                <button type="button" className="trigger-icon-button" aria-label="复制 Webhook 令牌" title="复制令牌" onClick={() => void copyValue("Webhook 令牌", oneTimeWebhook.token ?? "")}><Copy size={15} /></button>
              </div>
            )}
          </section>
        )}

        <div className="triggers-layout">
          <form className="trigger-create-panel" onSubmit={(event) => void handleCreate(event)}>
            <div className="trigger-panel-heading">
              <div>
                <h2>新建自动化</h2>
                <p>选择触发方式并填写要执行的任务。</p>
              </div>
            </div>
            <div className="trigger-kind-tabs" role="group" aria-label="触发方式">
              {([
                ["scheduled", "定时", CalendarClock],
                ["webhook", "Webhook", Webhook],
                ["event", "事件", Workflow],
              ] as const).map(([kind, label, Icon]) => (
                <button
                  className={formKind === kind ? "active" : ""}
                  type="button"
                  key={kind}
                  aria-pressed={formKind === kind}
                  onClick={() => setFormKind(kind)}
                ><Icon size={15} /><span>{label}</span></button>
              ))}
            </div>

            <label className="trigger-field">
              <span>名称</span>
              <input required maxLength={100} value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：工作日研究简报" />
            </label>
            <label className="trigger-field">
              <span>执行任务</span>
              <textarea required rows={4} value={task} onChange={(event) => setTask(event.target.value)} placeholder="描述触发后需要完成的研究任务" />
            </label>

            {formKind === "scheduled" && (
              <>
                <label className="trigger-field">
                  <span>Cron 表达式</span>
                  <input required value={schedule} onChange={(event) => setSchedule(event.target.value)} placeholder="分 时 日 月 周" />
                </label>
                <label className="trigger-field">
                  <span>时区</span>
                  <select value={timezone} onChange={(event) => setTimezone(event.target.value)}>
                    <option value="Asia/Shanghai">中国标准时间（上海）</option>
                    <option value="UTC">协调世界时（UTC）</option>
                    <option value="America/New_York">纽约</option>
                    <option value="Europe/London">伦敦</option>
                    <option value="Asia/Tokyo">东京</option>
                  </select>
                </label>
                <label className="trigger-checkbox">
                  <input type="checkbox" checked={runImmediately} onChange={(event) => setRunImmediately(event.target.checked)} />
                  <span>创建后立即执行一次</span>
                </label>
              </>
            )}

            {formKind === "webhook" && (
              <label className="trigger-checkbox">
                <input type="checkbox" checked={requireAuth} onChange={(event) => setRequireAuth(event.target.checked)} />
                <span>启用 Bearer 令牌认证</span>
              </label>
            )}

            {formKind === "event" && (
              <>
                <label className="trigger-field">
                  <span>事件类型</span>
                  <input required value={eventType} onChange={(event) => setEventType(event.target.value)} placeholder="例如：research.completed" />
                </label>
                <label className="trigger-field">
                  <span>事件来源（可选）</span>
                  <input value={eventSource} onChange={(event) => setEventSource(event.target.value)} placeholder="例如：agent" />
                </label>
                <label className="trigger-field">
                  <span>匹配条件（JSON 对象）</span>
                  <textarea rows={3} value={eventFilters} onChange={(event) => setEventFilters(event.target.value)} spellCheck={false} />
                </label>
              </>
            )}

            <button className="trigger-create-button" type="submit" disabled={saving}>
              {saving ? <RefreshCw size={15} className="trigger-spin" /> : <Plus size={15} />}
              <span>{saving ? "正在创建" : "创建自动化"}</span>
            </button>
          </form>

          <div className="trigger-list-panel">
            <div className="trigger-list-heading">
              <div>
                <h2>自动化列表</h2>
                <span>{triggers.length} 项</span>
              </div>
              <div className="trigger-filters">
                <label>
                  <span className="sr-only">按触发方式筛选</span>
                  <select value={kindFilter} onChange={(event) => setKindFilter(event.target.value as TriggerKindFilter)}>
                    <option value="all">全部类型</option>
                    <option value="scheduled">定时</option>
                    <option value="webhook">Webhook</option>
                    <option value="event">事件</option>
                  </select>
                </label>
                <label>
                  <span className="sr-only">按状态筛选</span>
                  <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as TriggerStatusFilter)}>
                    <option value="all">全部状态</option>
                    <option value="active">运行中</option>
                    <option value="paused">已暂停</option>
                    <option value="disabled">已停用</option>
                    <option value="error">异常</option>
                  </select>
                </label>
              </div>
            </div>

            {loading ? (
              <div className="trigger-list-state"><RefreshCw size={17} className="trigger-spin" />正在加载自动化…</div>
            ) : triggers.length === 0 ? (
              <div className="trigger-list-state">
                <Workflow size={25} />
                <strong>暂无自动化</strong>
                <span>创建定时任务、Webhook 或事件触发流程后，会显示在这里。</span>
              </div>
            ) : (
              <div className="trigger-list">
                {triggers.map((trigger) => (
                  <article className="trigger-item" key={trigger.id}>
                    <div className="trigger-item-main">
                      <div className="trigger-item-heading">
                        <h3>{trigger.name}</h3>
                        <span className={`trigger-type ${trigger.trigger_type}`}>{typeLabels[trigger.trigger_type]}</span>
                        <span className={`trigger-status ${trigger.status}`}>{statusLabels[trigger.status]}</span>
                      </div>
                      <p className="trigger-task">{trigger.task}</p>
                      <code className="trigger-summary">{triggerSummary(trigger)}</code>
                      <div className="trigger-metrics">
                        <span>执行 {trigger.execution_count} 次</span>
                        <span>失败 {trigger.failure_count} 次</span>
                        <span>最近执行 {formatDate(trigger.last_executed_at)}</span>
                      </div>
                    </div>
                    <div className="trigger-item-actions">
                      <button type="button" className="trigger-icon-button" title={expandedTrigger === trigger.id ? "收起执行记录" : "查看执行记录"} aria-label={`${expandedTrigger === trigger.id ? "收起" : "查看"} ${trigger.name} 的执行记录`} disabled={pendingAction === `executions:${trigger.id}`} onClick={() => void handleAction(trigger, "executions")}>
                        <History size={16} />
                      </button>
                      {trigger.status === "paused" ? (
                        <button type="button" className="trigger-icon-button" title="恢复" aria-label={`恢复 ${trigger.name}`} disabled={pendingAction === `resume:${trigger.id}`} onClick={() => void handleAction(trigger, "resume")}><Play size={15} /></button>
                      ) : trigger.status === "active" ? (
                        <button type="button" className="trigger-icon-button" title="暂停" aria-label={`暂停 ${trigger.name}`} disabled={pendingAction === `pause:${trigger.id}`} onClick={() => void handleAction(trigger, "pause")}><Pause size={15} /></button>
                      ) : null}
                      <button type="button" className="trigger-icon-button danger" title="删除" aria-label={`删除 ${trigger.name}`} disabled={pendingAction === `delete:${trigger.id}`} onClick={() => void handleAction(trigger, "delete")}><Trash2 size={15} /></button>
                    </div>
                    {expandedTrigger === trigger.id && (
                      <section className="trigger-execution-list" aria-label={`${trigger.name} 执行记录`}>
                        <h4>最近执行记录</h4>
                        {(executions[trigger.id] ?? []).length === 0 ? (
                          <p>暂无执行记录。</p>
                        ) : (executions[trigger.id] ?? []).map((execution) => (
                          <article className="trigger-execution" key={execution.id}>
                            <div>
                              <strong className={`execution-status ${execution.status}`}>{execution.status === "success" ? "成功" : execution.status === "failed" ? "失败" : "运行中"}</strong>
                              <time dateTime={execution.started_at}>{formatDate(execution.started_at)}</time>
                            </div>
                            {execution.error_message && <p>{execution.error_message}</p>}
                            {execution.output_text && <pre>{execution.output_text}</pre>}
                          </article>
                        ))}
                      </section>
                    )}
                  </article>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
