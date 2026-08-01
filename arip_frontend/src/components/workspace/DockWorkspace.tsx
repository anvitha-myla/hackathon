import { useCallback, useRef } from 'react'
import {
  DockviewReact,
  type DockviewReadyEvent,
  type IDockviewPanelProps,
} from 'dockview-react'
import { PfdCanvas } from '@/components/pfd/PfdCanvas'
import { PlotGridPanel, SystemConsole } from '@/components/panels/Panels'

function PfdPanel(_: IDockviewPanelProps) {
  return <PfdCanvas />
}

function PlotsPanel(_: IDockviewPanelProps) {
  return <PlotGridPanel />
}

function ConsolePanel(_: IDockviewPanelProps) {
  return <SystemConsole />
}

const components = {
  pfd: PfdPanel,
  plots: PlotsPanel,
  console: ConsolePanel,
}

export function DockWorkspace() {
  const apiRef = useRef<DockviewReadyEvent['api'] | null>(null)

  const onReady = useCallback((event: DockviewReadyEvent) => {
    apiRef.current = event.api
    event.api.addPanel({
      id: 'pfd',
      component: 'pfd',
      title: 'PFD Interactive Canvas',
    })
    event.api.addPanel({
      id: 'plots',
      component: 'plots',
      title: 'Time-Series Plot Grid',
      position: { referencePanel: 'pfd', direction: 'below' },
    })
    event.api.addPanel({
      id: 'console',
      component: 'console',
      title: 'System Console & Logs',
      position: { referencePanel: 'plots', direction: 'below' },
    })

    // Size console smaller
    const consolePanel = event.api.getPanel('console')
    consolePanel?.api.setSize({ height: 140 })
  }, [])

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <DockviewReact
        className="dockview-theme-dark"
        components={components}
        onReady={onReady}
      />
    </div>
  )
}
