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
      margin: { l: 48, r: 16, t: 28, b: 32 },
      legend: { orientation: 'h' as const, y: 1.15, font: { size: 9 } },
      xaxis: {
        title: 't [min]',
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
          { x: h.t, y: h.cN_phy, name: 'C_nitro physics', line: { color: '#f5a524', dash: 'dash', width: 1.5 }, mode: 'lines' },
          { x: h.t, y: h.cN_ekf, name: 'C_nitro EKF', line: { color: '#00d4ff', width: 2 }, mode: 'lines' },
          { x: h.t, y: h.cX_ekf, name: 'C_xylidine EKF', line: { color: '#3ddc84', width: 2 }, mode: 'lines' },
        ]}
        layout={{ ...layout, title: { text: 'Composition — Physics (dashed) vs EKF (solid)', font: { size: 11 } } }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
      <Plot
        data={[
          { x: h.t, y: h.T_phy, name: 'T physics', line: { color: '#f5a524', dash: 'dash', width: 1.5 }, mode: 'lines' },
          { x: h.t, y: h.T_ekf, name: 'T EKF', line: { color: '#00d4ff', width: 2 }, mode: 'lines' },
          { x: h.t, y: h.P, name: 'P [bar]', line: { color: '#3ddc84', width: 2 }, mode: 'lines', yaxis: 'y2' },
        ]}
        layout={{
          ...layout,
          title: { text: 'T_reactor & P_headspace', font: { size: 11 } },
          yaxis2: { overlaying: 'y', side: 'right', gridcolor: '#2a3038', title: 'P [bar]' },
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
  const tags = [
    ['RX.T', '°C', 2],
    ['RX.TJ', '°C', 2],
    ['RX.P', 'bar', 2],
    ['RX.C_NITRO', 'mol/L', 4],
    ['RX.C_XYL', 'mol/L', 4],
    ['RX.Q_RXN', 'kW', 2],
    ['H2.MFC', 'kg/min', 3],
    ['EKF.CONF', '%', 1],
  ] as const

  const map = Object.fromEntries((twin?.metrics || []).map((m) => [m.tag, m.value]))

  return (
    <div className="panel-card" style={{ margin: 0, height: '100%', borderRadius: 0 }}>
      <div className="hd">Live Telemetry & EKF</div>
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
        {tags.map(([tag, unit, prec]) => (
          <div className="gauge" key={tag}>
            <div className="k">{tag}</div>
            <div className="v">
              {fmt(map[tag], prec)}
              <span className="u">{unit}</span>
            </div>
          </div>
        ))}
        <div style={{ marginTop: 10, color: '#9aa3ad', fontSize: 11 }}>
          Stage: <b style={{ color: '#e6e8eb' }}>{twin?.stage ?? '—'}</b>
          <br />
          EKF conf:{' '}
          <b style={{ color: '#3ddc84' }}>{fmt(twin?.ekf?.confidence_score, 1)}%</b>
        </div>
      </div>
    </div>
  )
}

export function AiAdvisoryPanel() {
  const twin = useAppStore((s) => s.twin)
  const text = twin?.ai_advisory?.advisory_text
  return (
    <div className="panel-card" style={{ margin: 0, flex: 1, borderRadius: 0, display: 'flex', flexDirection: 'column' }}>
      <div className="hd">AI Operator Advisory · Ollama</div>
      <div className="bd" style={{ flex: 1, lineHeight: 1.4 }}>
        {text ||
          'Advisory stream updates periodically from LocalAIExplainer (template fallback when Ollama is offline). No hardware commands.'}
        <div style={{ marginTop: 8, color: '#9aa3ad', fontSize: 10 }}>
          source: {twin?.ai_advisory?.source ?? '—'}
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
          [SCADA] {a}
        </div>
      ))}
      {twin?.ekf?.sensor_residuals && (
        <div className="line info">
          [EKF] residuals T={fmt(twin.ekf.sensor_residuals.T_reactor, 3)} · P=
          {fmt(twin.ekf.sensor_residuals.P_headspace, 3)} · MFC=
          {fmt(twin.ekf.sensor_residuals.MFC_H2_rate, 3)}
        </div>
      )}
    </div>
  )
}
