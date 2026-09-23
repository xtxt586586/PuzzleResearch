import type {
  SseEvent,
  StreamEvent,
} from './types.js'

export function parseSseFrame(
  frame: string,
): SseEvent | null {
  const text = String(frame ?? '')

  if (!text.trim()) {
    return null
  }

  const lines = text
    .replace(/\r\n/g, '\n')
    .split('\n')

  let eventName = ''
  let eventId = ''
  const dataLines: string[] = []

  for (const line of lines) {
    if (!line || line.startsWith(':')) {
      continue
    }

    if (line.startsWith('event:')) {
      eventName = line
        .slice('event:'.length)
        .trim()
      continue
    }

    if (line.startsWith('id:')) {
      eventId = line
        .slice('id:'.length)
        .trim()
      continue
    }

    if (line.startsWith('data:')) {
      dataLines.push(
        line
          .slice('data:'.length)
          .replace(/^ /, ''),
      )
    }
  }

  if (dataLines.length === 0) {
    return null
  }

  let data: unknown

  try {
    data = JSON.parse(
      dataLines.join('\n'),
    )
  } catch {
    return null
  }

  const result: SseEvent = {
    data,
  }

  if (eventName) {
    result.event = eventName
  }

  if (eventId) {
    const numericId = Number(eventId)

    if (Number.isFinite(numericId)) {
      result.id = numericId
    }
  }

  return result
}


export function normalizeSseEvent(
  event: SseEvent,
): StreamEvent | null {
  const data = event.data

  if (
    data !== null
    && typeof data === 'object'
    && 'type' in data
    && 'data' in data
  ) {
    const envelope = data as {
      type: unknown
      data: unknown
    }

    return {
      type: String(envelope.type),
      data: envelope.data,
    }
  }

  if (event.event) {
    return {
      type: event.event,
      data,
    }
  }

  return null
}


export async function* readSseEvents(
  response: Response,
): AsyncGenerator<SseEvent> {
  if (!response.body) {
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const chunk = await reader.read()

    if (chunk.done) {
      break
    }

    buffer += decoder.decode(
      chunk.value,
      { stream: true },
    )

    buffer = buffer.replace(
      /\r\n/g,
      '\n',
    )

    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const parsed = parseSseFrame(frame)

      if (parsed) {
        yield parsed
      }
    }
  }

  buffer += decoder.decode()

  const tail = buffer.trim()

  if (tail) {
    const parsed = parseSseFrame(tail)

    if (parsed) {
      yield parsed
    }
  }
}


export async function* readDataStreamEvents(
  response: Response,
): AsyncGenerator<StreamEvent> {
  if (!response.body) {
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const chunk = await reader.read()

    if (chunk.done) {
      break
    }

    buffer += decoder.decode(
      chunk.value,
      { stream: true },
    )

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const rawLine of lines) {
      const line = rawLine.trim()

      if (!line.startsWith('0:')) {
        continue
      }

      try {
        const value = JSON.parse(
          line.slice(2),
        ) as unknown

        if (
          value !== null
          && typeof value === 'object'
          && 'type' in value
          && 'data' in value
        ) {
          yield value as StreamEvent
        }
      } catch {
        continue
      }
    }
  }

  buffer += decoder.decode()

  const tail = buffer.trim()

  if (tail.startsWith('0:')) {
    try {
      const value = JSON.parse(
        tail.slice(2),
      ) as unknown

      if (
        value !== null
        && typeof value === 'object'
        && 'type' in value
        && 'data' in value
      ) {
        yield value as StreamEvent
      }
    } catch {
      // 忽略不完整的尾部数据。
    }
  }
}