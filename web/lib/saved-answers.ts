const STORAGE_KEY = "puzzleresearch.saved-answers.v1"

export type SavedAnswer = {
  id: string
  type: "text"
  title: string
  content: string
  sourceMessageId: string
  sessionId?: string
  createdAt: number
  updatedAt: number
}

function isSavedAnswer(value: unknown): value is SavedAnswer {
  if (!value || typeof value !== "object") return false

  const item = value as Partial<SavedAnswer>
  return (
    typeof item.id === "string" &&
    item.type === "text" &&
    typeof item.title === "string" &&
    typeof item.content === "string" &&
    typeof item.sourceMessageId === "string" &&
    typeof item.createdAt === "number" &&
    Number.isFinite(item.createdAt) &&
    typeof item.updatedAt === "number" &&
    Number.isFinite(item.updatedAt)
  )
}

export function loadSavedAnswers(): SavedAnswer[] {
  if (typeof window === "undefined") return []

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed)
      ? parsed.filter(isSavedAnswer).sort(
          (left, right) => right.createdAt - left.createdAt,
        )
      : []
  } catch {
    return []
  }
}

export function persistSavedAnswers(
  answers: SavedAnswer[],
): boolean {
  if (typeof window === "undefined") return false

  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(answers),
    )
    return true
  } catch {
    return false
  }
}

export function createSavedAnswer(
  content: string,
  sourceMessageId: string,
  sessionId?: string | null,
): SavedAnswer {
  const now = Date.now()
  const firstLine = content
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^#{1,6}\s*/, ""))
    .find(Boolean)

  return {
    id: crypto.randomUUID(),
    type: "text",
    title: firstLine?.slice(0, 72) || "已保存的回答",
    content,
    sourceMessageId,
    ...(sessionId ? { sessionId } : {}),
    createdAt: now,
    updatedAt: now,
  }
}
