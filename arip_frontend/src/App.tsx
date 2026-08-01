import { useEffect, useRef } from 'react'
import { MenuBar } from '@/components/shell/MenuBar'
import { AssetTree, ToolBar } from '@/components/shell/ToolBar'
import { DockWorkspace } from '@/components/workspace/DockWorkspace'
import { AiAdvisoryPanel, TelemetryPanel } from '@/components/panels/Panels'
import { InputExpertDialog } from '@/components/expert/InputExpertDialog'
import { TwinStreamClient, fetchAdvisory, postReset } from '@/api/twinClient'
import { useAppStore } from '@/store/appStore'
import { fmt } from '@/lib/utils'

export function AppShell() {
  const clientRef = useRef<TwinStreamClient | null>(null)
  const setTwin = useAppStore((s) => s.setTwin)
  const setWsState = useAppStore((s) => s.setWsState)
  const pushLog = useAppStore((s) => s.pushLog)
  const setRunning = useAppStore((s) => s.setRunning)
  const clearHistory = useAppStore((s) => s.clearHistory)
  const wsState = useAppStore((s) => s.wsState)
  const twin = useAppStore((s) => s.twin)

  useEffect(() => {
    const client = new TwinStreamClient({
      onFrame: (frame) => setTwin(frame),
      onState: setWsState,
      onLog: pushLog,
      onHello: (msg) => pushLog('info', `Twin hello · batch=${String(msg.batch_id)} · hz=${String(msg.default_hz)}`),
    })
    clientRef.current = client
    client.connect()
    pushLog('info', 'ARIP engineering workbench started')

    const aiTimer = window.setInterval(() => {
      void fetchAdvisory()
        .then((d) => {
          const cur = useAppStore.getState().twin
          if (!cur) return
          setTwin({
            ...cur,
            ai_advisory: d.ai_advisory,
          })
        })
        .catch(() => undefined)
    }, 6000)

    return () => {
      client.disconnect()
      window.clearInterval(aiTimer)
    }
  }, [pushLog, setTwin, setWsState])

  const onRun = () => {
    setRunning(true)
    clientRef.current?.send({ action: 'start', hz: 10, dt_s: 0.5 })
    pushLog('ok', 'Simulation RUN')
  }
  const onPause = () => {
    setRunning(false)
    clientRef.current?.send({ action: 'pause' })
    pushLog('warn', 'Simulation PAUSE')
  }
  const onReset = () => {
    void postReset()
    clearHistory()
    clientRef.current?.send({ action: 'reset' })
    clientRef.current?.send({ action: 'start', hz: 10, dt_s: 0.5 })
    setRunning(true)
    pushLog('ok', 'Batch RESET — runtime cleared')
  }

  return (
    <div className="arip-app">
      <MenuBar onRun={onRun} onPause={onPause} onReset={onReset} />
      <ToolBar onRun={onRun} onPause={onPause} onReset={onReset} />

      <div className="arip-body">
        <aside className="arip-explorer">
          <div className="panel-card" style={{ margin: 0, borderRadius: 0, height: '100%' }}>
            <div className="hd">Asset Tree Explorer</div>
            <div className="bd" style={{ padding: 0 }}>
              <AssetTree />
            </div>
          </div>
        </aside>

        <main className="arip-workspace">
          <DockWorkspace />
        </main>

        <aside className="arip-side">
          <div style={{ height: '55%' }}>
            <TelemetryPanel />
          </div>
          <AiAdvisoryPanel />
        </aside>
      </div>

      <footer className="arip-statusbar">
        <span>WS: {wsState.toUpperCase()}</span>
        <span>t = {fmt(twin?.t_s, 1)} s</span>
        <span>Stage: {twin?.stage ?? '—'}</span>
        <span>
          Safety:{' '}
          {twin?.safety_status === 'CRITICAL'
            ? '🔴 CRITICAL'
            : twin?.safety_status === 'WARNING'
              ? '🟡 WARNING'
              : '🟢 NOMINAL'}
        </span>
        <span style={{ marginLeft: 'auto' }}>STBR-5000L Digital Twin Workbench</span>
      </footer>

      <InputExpertDialog />
    </div>
  )
}
