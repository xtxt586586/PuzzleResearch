const STORAGE_KEY = "puzzleresearch.saved-artifacts.v1"

export type ArtifactType = "code" | "image" | "text" | "file" | "report" | "chart"

export type GeneratedArtifact = {
  id: string
  type: ArtifactType
  title: string
  content?: string
  url?: string
  mimeType?: string
}

export type SavedArtifact = GeneratedArtifact & {
  savedId: string
  sourceArtifactId: string
  sessionId?: string
  createdAt: number
}

function isArtifactType(value: unknown): value is ArtifactType {
  return ["code", "image", "text", "file", "report", "chart"].includes(
    String(value),
  )
}

function isSavedArtifact(value: unknown): value is SavedArtifact {
  if (!value || typeof value !== "object") return false

  const item = value as Partial<SavedArtifact>
  return (
    typeof item.savedId === "string" &&
    typeof item.sourceArtifactId === "string" &&
    typeof item.id === "string" &&
    isArtifactType(item.type) &&
    typeof item.title === "string" &&
    (item.content === undefined || typeof item.content === "string") &&
    (item.url === undefined || typeof item.url === "string") &&
    (item.mimeType === undefined || typeof item.mimeType === "string") &&
    (item.sessionId === undefined || typeof item.sessionId === "string") &&
    typeof item.createdAt === "number" &&
    Number.isFinite(item.createdAt)
  )
}

export function savedArtifactSourceKey(
  sourceArtifactId: string,
  sessionId?: string | null,
): string {
  return `${sessionId ?? "workspace"}::${sourceArtifactId}`
}

export function loadSavedArtifacts(): SavedArtifact[] {
  if (typeof window === "undefined") return []

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed)
      ? parsed.filter(isSavedArtifact).sort(
          (left, right) => right.createdAt - left.createdAt,
        )
      : []
  } catch {
    return []
  }
}

export function persistSavedArtifacts(artifacts: SavedArtifact[]): boolean {
  if (typeof window === "undefined") return false

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(artifacts))
    return true
  } catch {
    return false
  }
}

export function createSavedArtifact(
  artifact: GeneratedArtifact,
  sessionId?: string | null,
): SavedArtifact {
  return {
    ...artifact,
    savedId: crypto.randomUUID(),
    sourceArtifactId: artifact.id,
    ...(sessionId ? { sessionId } : {}),
    createdAt: Date.now(),
  }
}
