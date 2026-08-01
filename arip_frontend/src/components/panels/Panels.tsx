import { useMemo } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import { useAppStore } from '@/store/appStore'
import { fmt } from '@/lib/utils'

const Plot = createPlotlyComponent(Plotly)

export function PlotGridPanel() {
  const h = useAppStore((s) => s.history)

  const layout = useMemo(
    () => ({
      autosize: true,
      paper_bgcolor: '#1e2125',
      plot_bgcolor: '#15181c',
      font: { color: '#c8cdd3', size: 10, family: 'Segoe UI' },
      margin: { l: 54, r: 48, t: 30, b: 34 },
      legend: { orientation: 'h' as const, y: 1.14, font: { size: 9 } },
      xaxis: {
        title: 'Batch Time [min]',
        gridcolor: '#2a3038',
        showspikes: true,
        spikemode: 'across' as const,
        spikesnap: 'cursor' as const,
      },
      yaxis: { gridcolor: '#2a3038', showspikes: true, spikemode: 'across' as const },
    }),
    [],
  )

  return (
    <div style={{ height: '100%', display: 'grid', gridTemplateRows: '1fr 1fr', gap: 2, background: '#121416' }}>
      <Plot
        data={[
          {
            x: h.t,
            y: h.cN_phy,
            name: 'Nitroxylene Concentration — Physics',
            line: { color: '#f5a524', dash: 'dash', width: 1.5 },
            mode: 'lines',
          },
          {
            x: h.t,
            y: h.cN_ekf,
            name: 'Nitroxylene Concentration — EKF Fused',
            line: { color: '#00d4ff', width: 2 },
            mode: 'lines',
          },
          {
            x: h.t,
            y: h.cX_ekf,
            name: 'Xylidine Concentration — EKF Fused',
            line: { color: '#3ddc84', width: 2 },
            mode: 'lines',
          },
        ]}
        layout={{
          ...layout,
          title: {
            text: 'Composition Profiles — Theoretical Physics (dashed) vs EKF Fused Real State (solid)',
            font: { size: 11 },
          },
          yaxis: { ...layout.yaxis, title: 'Concentration [mol/L]' },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
      <Plot
        data={[
          {
            x: h.t,
            y: h.T_phy,
            name: 'Reactor Temperature — Physics',
            line: { color: '#f5a524', dash: 'dash', width: 1.5 },
            mode: 'lines',
          },
          {
            x: h.t,
            y: h.T_ekf,
            name: 'Reactor Temperature — EKF Fused',
            line: { color: '#00d4ff', width: 2 },
            mode: 'lines',
          },
          {
            x: h.t,
            y: h.P,
            name: 'Headspace Pressure',
            line: { color: '#3ddc84', width: 2 },
            mode: 'lines',
            yaxis: 'y2',
          },
        ]}
        layout={{
          ...layout,
          title: {
            text: 'Reactor Temperature (T_reactor) & Headspace Pressure (P_headspace)',
            font: { size: 11 },
          },
          yaxis: { ...layout.yaxis, title: 'Reactor Temperature [°C]' },
          yaxis2: {
            overlaying: 'y',
            side: 'right',
            gridcolor: '#2a3038',
            title: 'Headspace Pressure [bar]',
          },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

export function TelemetryPanel() {
  const twin = useAppStore((s) => s.twin)
  const prefer = [
    'RX.T',
    'RX.TJ',
    'RX.P',
    'RX.C_NITRO',
    'RX.C_XYL',
    'RX.C_OH',
    'JACKET.FLOW',
    'H2.MFC',
    'RX.Q_RXN',
    'EKF.CONF',
  ]
  const map = Object.fromEntries((twin?.metrics || []).map((m) => [m.tag, m]))

  return (
    <div className="panel-card" style={{ margin: 0, height: '100%', borderRadius: 0 }}>
      <div className="hd">Live Engineering Readouts</div>
      <div className="bd">
        <div style={{ marginBottom: 8 }}>
          <span
            className={
              twin?.safety_status === 'CRITICAL'
                ? 'badge-crit'
                : twin?.safety_status === 'WARNING'
                  ? 'badge-warn'
                  : 'badge-nom'
            }
          >
            {twin?.safety_status === 'CRITICAL'
              ? '🔴 CRITICAL INTERLOCK'
              : twin?.safety_status === 'WARNING'
                ? '🟡 WARNING'
                : '🟢 NOMINAL'}
          </span>
        </div>
        {prefer.map((tag) => {
          const m = map[tag]
          if (!m) return null
          return (
            <div className="gauge" key={tag} style={{ gridTemplateColumns: '1fr auto', marginBottom: 6 }}>
              <div className="k" style={{ whiteSpace: 'normal', lineHeight: 1.25 }}>
                {m.display_name}
              </div>
              <div className="v">
                {fmt(m.value, m.precision)}
                <span className="u">{m.unit}</span>
              </div>
            </div>
          )
        })}
        <div style={{ marginTop: 10, color: '#9aa3ad', fontSize: 11 }}>
          Batch Stage: <b style={{ color: '#e6e8eb' }}>{twin?.stage ?? '—'}</b>
          <br />
          EKF State Estimation Confidence:{' '}
          <b style={{ color: '#3ddc84' }}>{fmt(twin?.ekf?.confidence_score, 1)}%</b>
        </div>
      </div>
    </div>
  )
}

export function SafetyPanel() {
  const twin = useAppStore((s) => s.twin)
  const alarms = twin?.decision?.alarms || []
  const ils = twin?.decision?.active_interlocks || []
  const recs = twin?.decision?.optimization_recommendations || []

  return (
    <div className="panel-card" style={{ margin: 0, height: '100%', borderRadius: 0, overflow: 'auto' }}>
      <div className="hd">Process Safety Interlocks</div>
      <div className="bd">
        <div style={{ marginBottom: 10 }}>
          <span
            className={
              twin?.safety_status === 'CRITICAL'
                ? 'badge-crit'
                : twin?.safety_status === 'WARNING'
                  ? 'badge-warn'
                  : 'badge-nom'
            }
            style={{ fontSize: 14 }}
          >
            {twin?.safety_status === 'CRITICAL'
              ? '🔴 CRITICAL INTERLOCK'
              : twin?.safety_status === 'WARNING'
                ? '🟡 WARNING'
                : '🟢 NOMINAL'}
          </span>
        </div>
        <h4 style={{ margin: '0 0 6px', color: '#9aa3ad', fontSize: 11, letterSpacing: '0.06em' }}>
          HARD ALARMS
        </h4>
        {!alarms.length && !ils.length && (
          <div className="opt-item" style={{ color: '#3ddc84' }}>
            No hard safety interlocks are active.
          </div>
        )}
        {ils.map((t) => (
          <div key={t} className="opt-item" style={{ borderLeft: '3px solid #ff5c5c' }}>
            Interlock: {t}
          </div>
        ))}
        {alarms.map((a) => (
          <div key={a} className="opt-item" style={{ borderLeft: '3px solid #f5a524' }}>
            {a}
          </div>
        ))}
        <h4 style={{ margin: '12px 0 6px', color: '#9aa3ad', fontSize: 11, letterSpacing: '0.06em' }}>
          SOFT OPTIMIZATION RULES
        </h4>
        {!recs.length && (
          <div className="opt-item" style={{ color: '#9aa3ad' }}>
            No soft process recommendations at this time.
          </div>
        )}
        {recs.map((r) => (
          <div key={r} className="opt-item">
            {r}
          </div>
        ))}
      </div>
    </div>
  )
}

export function AiAdvisoryPanel() {
  const twin = useAppStore((s) => s.twin)
  const text = twin?.ai_advisory?.advisory_text
  return (
    <div
      className="panel-card"
      style={{ margin: 0, height: '100%', borderRadius: 0, display: 'flex', flexDirection: 'column' }}
    >
      <div className="hd">AI Operator Assistant · Local Ollama Explainer</div>
      <div className="bd" style={{ flex: 1, lineHeight: 1.45 }}>
        {text ||
          'Operator advisory updates from LocalAIExplainer. Advisory only — never issues hardware control commands. Falls back to structured template text when Ollama is offline.'}
        <div style={{ marginTop: 10, color: '#9aa3ad', fontSize: 10 }}>
          Source: {twin?.ai_advisory?.source ?? '—'} · Residual mode:{' '}
          {twin?.residual?.is_pure_physics ? 'pure physics' : 'hybrid ML'}
        </div>
      </div>
    </div>
  )
}

export function SystemConsole() {
  const logs = useAppStore((s) => s.logs)
  const twin = useAppStore((s) => s.twin)
  return (
    <div className="console">
      {logs.map((l, i) => (
        <div key={i} className={`line ${l.level}`}>
          [{l.ts}] {l.message}
        </div>
      ))}
      {twin?.decision?.alarms?.map((a, i) => (
        <div key={`a${i}`} className="line warn">
          [SCADA Safety Interlock] {a}
        </div>
      ))}
      {twin?.ekf?.sensor_residuals && (
        <div className="line info">
          [EKF Residuals] Reactor Temperature={fmt(twin.ekf.sensor_residuals.T_reactor, 3)} °C ·
          Headspace Pressure={fmt(twin.ekf.sensor_residuals.P_headspace, 3)} bar · Hydrogen Mass-Flow=
          {fmt(twin.ekf.sensor_residuals.MFC_H2_rate, 3)}
        </div>
      )}
    </div>
  )
}
