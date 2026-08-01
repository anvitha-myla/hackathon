import { useEffect, useState } from 'react'
import { useAppStore } from '@/store/appStore'
import { postReset } from '@/api/twinClient'

type Props = {
  onRun: () => void
  onPause: () => void
  onReset: () => void
}

const MENUS: Record<string, Array<{ label: string; action?: string; sep?: boolean }>> = {
  File: [
    { label: 'New Batch Run…', action: 'new' },
    { label: 'Open Equipment Package…', action: 'open-pkg' },
    { sep: true, label: '' },
    { label: 'Export Telemetry CSV', action: 'export' },
    { sep: true, label: '' },
    { label: 'Exit', action: 'exit' },
  ],
  Edit: [
    { label: 'Undo', action: 'undo' },
    { label: 'Redo', action: 'redo' },
  ],
  Equipment: [
    { label: 'Open Input Expert…', action: 'expert' },
    { label: 'Reload Packages', action: 'reload' },
  ],
  Flowsheet: [
    { label: 'Fit PFD View', action: 'fit' },
    { label: 'Toggle Stream Animation', action: 'anim' },
  ],
  Simulation: [
    { label: 'Run', action: 'run' },
    { label: 'Pause', action: 'pause' },
    { label: 'Reset Batch', action: 'reset' },
  ],
  Safety: [
    { label: 'View Active Interlocks', action: 'interlocks' },
    { label: 'Open Decision Engine Log', action: 'decision-log' },
  ],
  Diagnostics: [
    { label: 'EKF Residuals', action: 'ekf' },
    { label: 'WebSocket Health', action: 'ws' },
  ],
  View: [
    { label: 'Reset Dock Layout', action: 'layout' },
    { label: 'Show System Console', action: 'console' },
  ],
  Help: [
    { label: 'About ARIP Twin', action: 'about' },
  ],
}

export function MenuBar({ onRun, onPause, onReset }: Props) {
  const [open, setOpen] = useState<string | null>(null)
  const pushLog = useAppStore((s) => s.pushLog)
  const openExpert = useAppStore((s) => s.openExpert)
  const selected = useAppStore((s) => s.selectedEquipmentId)

  useEffect(() => {
    const close = () => setOpen(null)
    window.addEventListener('click', close)
    return () => window.removeEventListener('click', close)
  }, [])

  const act = (action?: string) => {
    setOpen(null)
    if (!action) return
    if (action === 'run') onRun()
    else if (action === 'pause') onPause()
    else if (action === 'reset') {
      void postReset()
      onReset()
    } else if (action === 'expert') openExpert(selected ?? 'EQ-STBR-5000L')
    else if (action === 'about')
      pushLog('info', 'ARIP Digital Twin — desktop-class PFD / Dockview / Input Expert workbench')
    else if (action === 'interlocks') {
      const d = useAppStore.getState().twin?.decision
      pushLog('warn', `Interlocks: ${(d?.active_interlocks || []).join(', ') || 'none'}`)
    } else pushLog('info', `Menu action: ${action}`)
  }

  return (
    <div className="arip-menubar" onClick={(e) => e.stopPropagation()}>
      <div className="brand">ARIP Twin</div>
      {Object.keys(MENUS).map((name) => (
        <div
          key={name}
          className={`arip-menu-item ${open === name ? 'open' : ''}`}
          onClick={(e) => {
            e.stopPropagation()
            setOpen(open === name ? null : name)
          }}
        >
          {name}
          {open === name && (
            <div className="arip-menu-dropdown">
              {MENUS[name].map((item, i) =>
                item.sep ? (
                  <div key={i} className="sep" />
                ) : (
                  <button key={i} onClick={() => act(item.action)}>
                    {item.label}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
