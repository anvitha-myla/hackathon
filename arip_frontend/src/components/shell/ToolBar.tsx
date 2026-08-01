import {
  FlaskConical,
  FolderTree,
  Gauge,
  Play,
  Pause,
  RotateCcw,
  Settings2,
  FastForward,
} from 'lucide-react'
import { EQUIPMENT, useAppStore, type PrimaryView, type SpeedX } from '@/store/appStore'

type Props = {
  onRun: () => void
  onPause: () => void
  onReset: () => void
  onJump: () => void
  onSpeed: (s: SpeedX) => void
}

export function ToolBar({ onRun, onPause, onReset, onJump, onSpeed }: Props) {
  const simPhase = useAppStore((s) => s.simPhase)
  const speedX = useAppStore((s) => s.speedX)
  const primaryView = useAppStore((s) => s.primaryView)
  const setPrimaryView = useAppStore((s) => s.setPrimaryView)
  const bottomPanel = useAppStore((s) => s.bottomPanel)
  const setBottomPanel = useAppStore((s) => s.setBottomPanel)
  const jumpMinutes = useAppStore((s) => s.jumpMinutes)
  const setJumpMinutes = useAppStore((s) => s.setJumpMinutes)
  const purePhysics = useAppStore((s) => s.purePhysics)
  const openExpert = useAppStore((s) => s.openExpert)
  const selected = useAppStore((s) => s.selectedEquipmentId)

  const busy = simPhase === 'starting' || simPhase === 'pausing' || simPhase === 'resetting' || simPhase === 'jumping'
  const canStart = !busy && (simPhase === 'idle' || simPhase === 'paused')
  const canPause = !busy && simPhase === 'running'
  const canReset = !busy

  const views: Array<{ id: PrimaryView; label: string }> = [
    { id: 'process', label: '1 · Process Diagram & Reactor Visuals' },
    { id: 'analytics', label: '2 · Full-Screen Analytics (LabPlot)' },
    { id: 'safety', label: '3 · Safety Interlocks & AI Assistant' },
  ]

  return (
    <div className="arip-toolbar" style={{ flexWrap: 'wrap', height: 'auto', minHeight: 34, gap: 6, padding: '4px 8px' }}>
      <button className="arip-toolbtn primary" onClick={onRun} disabled={!canStart} title="Start live stream">
        <Play size={14} /> {simPhase === 'starting' ? 'Starting…' : 'Start'}
      </button>
      <button className="arip-toolbtn" onClick={onPause} disabled={!canPause} title="Pause live stream">
        <Pause size={14} /> {simPhase === 'pausing' ? 'Pausing…' : 'Pause'}
      </button>
      <button className="arip-toolbtn danger" onClick={onReset} disabled={!canReset} title="Reset batch">
        <RotateCcw size={14} /> {simPhase === 'resetting' ? 'Resetting…' : 'Reset'}
      </button>

      <div className="arip-tool-sep" />

      <span style={{ color: '#9aa3ad' }}>Live speed</span>
      {([1, 5, 10] as SpeedX[]).map((s) => (
        <button
          key={s}
          className={`arip-toolbtn ${speedX === s ? 'primary' : ''}`}
          onClick={() => onSpeed(s)}
          disabled={busy}
        >
          <FastForward size={12} /> {s}×
        </button>
      ))}

      <div className="arip-tool-sep" />

      <span style={{ color: '#9aa3ad' }}>Time jump</span>
      <input
        value={jumpMinutes}
        onChange={(e) => setJumpMinutes(e.target.value)}
        style={{
          width: 64,
          height: 24,
          background: '#1e2125',
          border: '1px solid #3d444d',
          color: '#e6e8eb',
          padding: '0 6px',
          fontFamily: 'ui-monospace, monospace',
        }}
        title="Jump to batch time [min]"
      />
      <span style={{ color: '#9aa3ad' }}>min</span>
      <button className="arip-toolbtn" onClick={onJump} disabled={busy}>
        {simPhase === 'jumping' ? 'Jumping…' : 'Jump'}
      </button>

      <div className="arip-tool-sep" />

      <button className="arip-toolbtn" onClick={() => openExpert(selected ?? 'EQ-STBR-5000L')}>
        <Settings2 size={14} /> Input Expert
      </button>

      <div className="arip-tool-sep" />

      <div className="view-switcher">
        {views.map((v) => (
          <button
            key={v.id}
            className={`view-btn ${primaryView === v.id ? 'active' : ''}`}
            onClick={() => setPrimaryView(v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
        <span className={`phys-pill ${purePhysics ? 'pure' : 'hybrid'}`}>
          {purePhysics ? 'Pure Physics (δ_ML = 0)' : 'Hybrid ML Residual ON'}
        </span>
        <span style={{ color: '#9aa3ad' }}>Bottom:</span>
        {(['console', 'plots', 'hidden'] as const).map((b) => (
          <button
            key={b}
            className={`arip-toolbtn ${bottomPanel === b ? 'primary' : ''}`}
            onClick={() => setBottomPanel(b)}
          >
            {b === 'console' ? 'Logs' : b === 'plots' ? 'Plots' : 'Hide'}
          </button>
        ))}
        <span style={{ color: '#9aa3ad', fontSize: 11 }}>phase={simPhase}</span>
      </div>
    </div>
  )
}

export function AssetTree() {
  const selected = useAppStore((s) => s.selectedEquipmentId)
  const selectEquipment = useAppStore((s) => s.selectEquipment)
  const openExpert = useAppStore((s) => s.openExpert)

  return (
    <div>
      <div className="tree-section">
        <FolderTree size={12} style={{ display: 'inline', marginRight: 4 }} />
        Equipment Packages
      </div>
      {EQUIPMENT.map((eq) => (
        <div
          key={eq.id}
          className={`tree-item ${selected === eq.id ? 'active' : ''}`}
          onClick={() => selectEquipment(eq.id)}
          onDoubleClick={() => openExpert(eq.id)}
        >
          <Gauge size={13} />
          <span>{eq.name}</span>
        </div>
      ))}
      <div className="tree-section" style={{ marginTop: 10 }}>
        <FlaskConical size={12} style={{ display: 'inline', marginRight: 4 }} />
        Chemical Packages
      </div>
      <div className="tree-item">
        <FlaskConical size={13} />
        <span>2,4-Nitroxylene Hydrogenation</span>
      </div>
      <div className="tree-section" style={{ marginTop: 10 }}>Batch Runs</div>
      <div className="tree-item active">
        <Play size={13} />
        <span>Run_001_Isothermal</span>
      </div>
    </div>
  )
}
