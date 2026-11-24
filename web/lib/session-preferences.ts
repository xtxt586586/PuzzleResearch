const STORAGE_KEY = "puzzleresearch.session-preferences.v1"

export type SessionPreference = {
  title?: string
  pinned: boolean
}

export type SessionPreferences = Record<
  string,
  SessionPreference
>

export function loadSessionPreferences(): SessionPreferences {
  if (typeof window === "undefined") return {}

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : {}
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed).flatMap(([id, value]) => {
        if (!value || typeof value !== "object") return []

        const preference = value as Partial<SessionPreference>
        return [[
          id,
          {
            ...(typeof preference.title === "string"
              ? { title: preference.title }
              : {}),
            pinned: preference.pinned === true,
          },
        ]]
      }),
    )
  } catch {
    return {}
  }
}

export function persistSessionPreferences(
  preferences: SessionPreferences,
): boolean {
  if (typeof window === "undefined") return false

  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(preferences),
    )
    return true
  } catch {
    return false
  }
}
