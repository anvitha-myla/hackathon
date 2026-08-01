import {
  FlaskConical,
  FolderTree,
  Gauge,
  Play,
  Pause,
  RotateCcw,
  Settings2,
} from 'lucide-react'
import { EQUIPMENT, useAppStore } from '@/store/appStore'

type Props = {
  onRun: () => void
  onPause: () => void
  onReset: () => void
}

export function ToolBar({ onRun, onPause, onReset }: Props) {
  const running = useAppStore((s) => s.running)
  const openExpert = useAppStore((s) => s.openExpert)
  const selected = useAppStore((s) => s.selectedEquipmentId)

  return (
    <div className="arip-toolbar">
      <button className="arip-toolbtn primary" onClick={onRun} title="Run">
        <Play size={14} /> Run
      </button>
      <button className="arip-toolbtn" onClick={onPause} title="Pause">
        <Pause size={14} /> Pause
      </button>
      <button className="arip-toolbtn danger" onClick={onReset} title="Reset">
        <RotateCcw size={14} /> Reset
      </button>
      <div className="arip-tool-sep" />
      <button
        className="arip-toolbtn"
        onClick={() => openExpert(selected ?? 'EQ-STBR-5000L')}
      >
        <Settings2 size={14} /> Input Expert
      </button>
      <div className="arip-tool-sep" />
      <span style={{ color: '#9aa3ad', fontSize: 11 }}>
        Simulation: {running ? 'RUNNING' : 'PAUSED'} · STBR-5000L Nitroxylene Hydrogenation
      </span>
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
        <span>2,4-Nitroxylene Rxn</span>
      </div>

      <div className="tree-section" style={{ marginTop: 10 }}>Batch Runs</div>
      <div className="tree-item active">
        <Play size={13} />
        <span>Run_001_Isothermal</span>
      </div>
    </div>
  )
}
