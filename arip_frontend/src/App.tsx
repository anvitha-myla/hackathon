import { useEffect, useRef } from 'react'
import { MenuBar } from '@/components/shell/MenuBar'
import { AssetTree, ToolBar } from '@/components/shell/ToolBar'
import { PfdCanvas } from '@/components/pfd/PfdCanvas'
import {
  AiAdvisoryPanel,
  PlotGridPanel,
  SystemConsole,
  TelemetryPanel,
  SafetyPanel,
} from '@/components/panels/Panels'
import { InputExpertDialog } from '@/components/expert/InputExpertDialog'
import {
  TwinStreamClient,
  fetchAdvisory,
  fetchPhysicsMode,
  postReset,
  postTimeJump,
} from '@/api/twinClient'
import { useAppStore, type SpeedX } from '@/store/appStore'
import { fmt } from '@/lib/utils'

export function AppShell() {
  const clientRef = useRef<TwinStreamClient | null>(null)
  const setTwin = useAppStore((s) => s.setTwin)
  const hydrateTwin = useAppStore((s) => s.hydrateTwin)
  const setWsState = useAppStore((s) => s.setWsState)
  const pushLog = useAppStore((s) => s.pushLog)
  const setSimPhase = useAppStore((s) => s.setSimPhase)
  const clearHistory = useAppStore((s) => s.clearHistory)
  const replaceHistoryFromKeyframes = useAppStore((s) => s.replaceHistoryFromKeyframes)
  const setPurePhysics = useAppStore((s) => s.setPurePhysics)
  const setSpeedX = useAppStore((s) => s.setSpeedX)
  const wsState = useAppStore((s) => s.wsState)
  const twin = useAppStore((s) => s.twin)
  const primaryView = useAppStore((s) => s.primaryView)
  const bottomPanel = useAppStore((s) => s.bottomPanel)
  const simPhase = useAppStore((s) => s.simPhase)
  const speedX = useAppStore((s) => s.speedX)
  const jumpMinutes = useAppStore((s) => s.jumpMinutes)
  const purePhysics = useAppStore((s) => s.purePhysics)

  useEffect(() => {
    const client = new TwinStreamClient({
      onFrame: (frame) => {
        setTwin(frame)
        // If we were starting, first frame confirms running
        const phase = useAppStore.getState().simPhase
        if (phase === 'starting') setSimPhase('running')
      },
      onState: setWsState,
      onLog: pushLog,
      onHello: (msg) => {
        pushLog(
          'info',
          `Twin hello · batch=${String(msg.batch_id)} · pure_physics=${String(msg.is_pure_physics)}`,
        )
        if (typeof msg.is_pure_physics === 'boolean') setPurePhysics(msg.is_pure_physics)
      },
      onStartAck: () => setSimPhase('running'),
      onPauseAck: () => setSimPhase('paused'),
      onResetAck: () => {
        clearHistory()
        setSimPhase('idle')
      },
    })
    clientRef.current = client
    client.connect()
    pushLog('info', 'ARIP engineering workbench ready — press Start for live Mode A')

    void fetchPhysicsMode().then((d) => {
      setPurePhysics(Boolean(d.is_pure_physics))
      pushLog(
        d.is_pure_physics ? 'ok' : 'info',
        d.is_pure_physics
          ? 'Strict pure-physics mode: no lab dataset in /data — δ_ML = 0'
          : `Hybrid residual enabled (${d.model_name ?? 'model'})`,
      )
    })

    const aiTimer = window.setInterval(() => {
      void fetchAdvisory()
        .then((d) => {
          const cur = useAppStore.getState().twin
          if (!cur) return
          setTwin({ ...cur, ai_advisory: d.ai_advisory })
        })
        .catch(() => undefined)
    }, 8000)

    return () => {
      client.disconnect()
      window.clearInterval(aiTimer)
    }
  }, [clearHistory, pushLog, setPurePhysics, setSimPhase, setTwin, setWsState])

  const onRun = () => {
    if (simPhase !== 'idle' && simPhase !== 'paused') return
    setSimPhase('starting')
    clientRef.current?.setSpeed(speedX)
    clientRef.current?.start(speedX)
    pushLog('ok', `Mode A START at ${speedX}× (awaiting start_ack)`)
  }

  const onPause = () => {
    if (simPhase !== 'running') return
    setSimPhase('pausing')
    clientRef.current?.pause()
    pushLog('warn', 'Mode A PAUSE requested')
  }

  const onReset = async () => {
    setSimPhase('resetting')
    clientRef.current?.pause()
    try {
      const res = await postReset()
      clearHistory()
      if (res.frame) hydrateTwin(res.frame)
      clientRef.current?.resetStream()
      pushLog('ok', 'Batch RESET complete — idle (not auto-started)')
      setSimPhase('idle')
    } catch (e) {
      pushLog('err', `Reset failed: ${String(e)}`)
      setSimPhase('idle')
    }
  }

  const onSpeed = (s: SpeedX) => {
    setSpeedX(s)
    clientRef.current?.setSpeed(s)
    clientRef.current?.send({
      action: 'configure',
      hz: 10,
      dt_s: 0.1,
      speed_x: s,
      include_ai_advisory: false,
    })
    pushLog('info', `Live speed set to ${s}×`)
  }

  const onJump = async () => {
    const tMin = Number(jumpMinutes)
    if (!Number.isFinite(tMin) || tMin < 0) {
      pushLog('err', 'Time jump requires a non-negative minutes value')
      return
    }
    setSimPhase('jumping')
    clientRef.current?.pause()
    try {
      const res = await postTimeJump(tMin)
      clearHistory()
      replaceHistoryFromKeyframes(res.keyframes || [])
      if (res.frame) hydrateTwin(res.frame)
      setPurePhysics(Boolean(res.is_pure_physics))
      pushLog(
        'ok',
        `Mode B TIME JUMP → t=${fmt(res.t_min, 2)} min (${fmt(res.t_s, 1)} s) · keyframes=${res.n_keyframes}`,
      )
      setSimPhase('paused')
    } catch (e) {
      pushLog('err', `Time jump failed: ${String(e)}`)
      setSimPhase('paused')
    }
  }

  return (
    <div className="arip-app">
      <MenuBar onRun={onRun} onPause={onPause} onReset={onReset} />
      <ToolBar
        onRun={onRun}
        onPause={onPause}
        onReset={onReset}
        onJump={onJump}
        onSpeed={onSpeed}
      />

      <div className={`arip-body view-${primaryView}`}>
        {primaryView !== 'analytics' && (
          <aside className="arip-explorer">
            <div className="panel-card" style={{ margin: 0, borderRadius: 0, height: '100%' }}>
              <div className="hd">Asset Tree Explorer</div>
              <div className="bd" style={{ padding: 0 }}>
                <AssetTree />
              </div>
            </div>
          </aside>
        )}

        <main className="arip-workspace">
          {primaryView === 'process' && (
            <div style={{ height: '100%', display: 'grid', gridTemplateRows: bottomPanel === 'hidden' ? '1fr' : '1fr 160px' }}>
              <PfdCanvas />
              {bottomPanel !== 'hidden' && (
                <div style={{ borderTop: '1px solid #3d444d', minHeight: 0 }}>
                  {bottomPanel === 'console' ? <SystemConsole /> : <PlotGridPanel />}
                </div>
              )}
            </div>
          )}

          {primaryView === 'analytics' && (
            <div style={{ height: '100%', display: 'grid', gridTemplateRows: bottomPanel === 'hidden' ? '1fr' : '1fr 140px' }}>
              <PlotGridPanel />
              {bottomPanel === 'console' && <SystemConsole />}
              {bottomPanel === 'plots' && (
                <div className="console">
                  <div className="line info">
                    LabPlot mode — physics traces are dashed; EKF fused traces are solid.
                  </div>
                </div>
              )}
            </div>
          )}

          {primaryView === 'safety' && (
            <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1.1fr 1fr', gridTemplateRows: bottomPanel === 'hidden' ? '1fr' : '1fr 150px' }}>
              <SafetyPanel />
              <AiAdvisoryPanel />
              {bottomPanel !== 'hidden' && (
                <div style={{ gridColumn: '1 / -1', borderTop: '1px solid #3d444d', minHeight: 0 }}>
                  {bottomPanel === 'console' ? <SystemConsole /> : <PlotGridPanel />}
                </div>
              )}
            </div>
          )}
        </main>

        {primaryView === 'process' && (
          <aside className="arip-side">
            <div style={{ height: '100%' }}>
              <TelemetryPanel />
            </div>
          </aside>
        )}
      </div>

      <footer className="arip-statusbar">
        <span>WS: {wsState.toUpperCase()}</span>
        <span>Control: {simPhase.toUpperCase()}</span>
        <span>Speed: {speedX}×</span>
        <span>t = {fmt(twin?.t_s, 1)} s ({fmt(twin?.t_min, 2)} min)</span>
        <span>Stage: {twin?.stage ?? '—'}</span>
        <span>{purePhysics ? 'PURE PHYSICS' : 'HYBRID ML'}</span>
        <span>
          Safety:{' '}
          {twin?.safety_status === 'CRITICAL'
            ? '🔴 CRITICAL INTERLOCK'
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
