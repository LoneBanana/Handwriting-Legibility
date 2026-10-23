import { apiFetch } from './client'

export type SseEvent = { event: string; data: string; id: string | null }

/**
 * Incremental parser for the text/event-stream format
 * (https://html.spec.whatwg.org/multipage/server-sent-events.html).
 * Chunks may split lines (including a CRLF pair) anywhere.
 */
export class SseParser {
  private buffer = ''
  private pendingCR = false
  private eventType = ''
  private dataLines: string[] = []
  private lastId: string | null = null

  push(chunk: string): SseEvent[] {
    const events: SseEvent[] = []
    let text = chunk
    // A CR at the end of the previous chunk may be the first half of CRLF.
    if (this.pendingCR && text.startsWith('\n')) text = text.slice(1)
    this.pendingCR = false

    let start = 0
    for (let i = 0; i < text.length; i += 1) {
      const ch = text[i]
      if (ch !== '\n' && ch !== '\r') continue
      const line = this.buffer + text.slice(start, i)
      this.buffer = ''
      if (ch === '\r') {
        if (i + 1 < text.length) {
          if (text[i + 1] === '\n') i += 1
        } else {
          this.pendingCR = true
        }
      }
      start = i + 1
      const event = this.processLine(line)
      if (event) events.push(event)
    }
    this.buffer += text.slice(start)
    return events
  }

  /** Dispatches a final event if the stream ended without a trailing blank line. */
  flush(): SseEvent[] {
    const events: SseEvent[] = []
    if (this.buffer) {
      const event = this.processLine(this.buffer)
      this.buffer = ''
      if (event) events.push(event)
    }
    const last = this.dispatch()
    if (last) events.push(last)
    return events
  }

  private processLine(line: string): SseEvent | null {
    if (line === '') return this.dispatch()
    if (line.startsWith(':')) return null // comment / keep-alive ping
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    switch (field) {
      case 'event':
        this.eventType = value
        break
      case 'data':
        this.dataLines.push(value)
        break
      case 'id':
        if (!value.includes('\0')) this.lastId = value
        break
      default:
        break // "retry" and unknown fields are ignored
    }
    return null
  }

  private dispatch(): SseEvent | null {
    if (this.dataLines.length === 0) {
      this.eventType = ''
      return null
    }
    const event: SseEvent = {
      event: this.eventType || 'message',
      data: this.dataLines.join('\n'),
      id: this.lastId,
    }
    this.eventType = ''
    this.dataLines = []
    return event
  }
}

/** POSTs JSON and yields server-sent events as they arrive. */
export async function* postEventStream(
  path: string,
  json: unknown,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const response = await apiFetch(path, {
    method: 'POST',
    json,
    signal,
    headers: { Accept: 'text/event-stream' },
  })
  if (!response.body) throw new Error('This browser cannot read streamed responses.')
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  const parser = new SseParser()
  let finished = false
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      for (const event of parser.push(value)) yield event
    }
    finished = true
    for (const event of parser.flush()) yield event
  } finally {
    // If the consumer stopped early, cancel so the request is torn down.
    if (!finished) await reader.cancel().catch(() => undefined)
    else reader.releaseLock()
  }
}
