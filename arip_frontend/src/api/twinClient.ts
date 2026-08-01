import type { TwinFrame } from '@/store/appStore'

function wsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/ws/v1/twin/stream`
}

export type StreamHandlers = {
  onFrame: (frame: TwinFrame) => void
  onHello?: (msg: Record<string, unknown>) => void
  onState?: (state: 'connecting' | 'live' | 'offline') => void
  onLog?: (level: 'info' | 'warn' | 'err' | 'ok', message: string) => void
}

export class TwinStreamClient {
  private ws: WebSocket | null = null
  private handlers: StreamHandlers
  private shouldRun = true

  constructor(handlers: StreamHandlers) {
    this.handlers = handlers
  }

  connect() {
    this.handlers.onState?.('connecting')
    this.ws = new WebSocket(wsUrl())
    this.ws.onopen = () => {
      this.handlers.onState?.('live')
      this.handlers.onLog?.('ok', 'WebSocket connected to /ws/v1/twin/stream')
      this.send({
        action: 'start',
        hz: 10,
        dt_s: 0.5,
        include_ai_advisory: false,
      })
    }
    this.ws.onclose = () => {
      this.handlers.onState?.('offline')
      this.handlers.onLog?.('warn', 'WebSocket closed — reconnecting in 1.2s')
      if (this.shouldRun) setTimeout(() => this.connect(), 1200)
    }
    this.ws.onerror = () => {
      this.handlers.onLog?.('err', 'WebSocket error')
    }
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'hello') this.handlers.onHello?.(msg)
      if (msg.type === 'frame') this.handlers.onFrame(msg.payload as TwinFrame)
      if (msg.type === 'error') this.handlers.onLog?.('err', String(msg.detail))
    }
  }

  send(payload: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload))
    }
  }

  disconnect() {
    this.shouldRun = false
    this.ws?.close()
  }
}

export async function postSetpoints(body: Record<string, number>) {
  await fetch('/api/v1/twin/setpoints', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function postReset() {
  await fetch('/api/v1/twin/reset', { method: 'POST' })
}

export async function fetchAdvisory() {
  const r = await fetch('/api/v1/twin/advisory', { method: 'POST' })
  return r.json()
}
