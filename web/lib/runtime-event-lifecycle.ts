const TERMINAL_RUNTIME_EVENTS = new Set([
  "completion",
  "cancelled",
  "error",
])

export function isTerminalRuntimeEvent(type: string): boolean {
  return TERMINAL_RUNTIME_EVENTS.has(type)
}

export function pendingInterruptAfterRuntimeEvent<T>(
  type: string,
  pendingInterrupt: T | null,
): T | null {
  return isTerminalRuntimeEvent(type) ? null : pendingInterrupt
}

export function firstPendingInterrupt(payload: unknown): unknown | null {
  const root =
    payload !== null && typeof payload === "object" && !Array.isArray(payload)
      ? (payload as Record<string, unknown>)
      : null
  const interrupts = root?.interrupts ?? root?.__interrupt__ ?? payload

  if (Array.isArray(interrupts)) return interrupts[0] ?? null
  return interrupts ?? null
}
