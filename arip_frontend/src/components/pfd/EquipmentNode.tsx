import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useAppStore, type EquipmentId } from '@/store/appStore'

export type PfdNodeData = {
  equipmentId: EquipmentId
  title: string
  subtitle: string
  kind: 'reactor' | 'tcu' | 'mfc' | 'filter' | 'wfe'
}

export function EquipmentNode({ id, data, selected }: NodeProps) {
  const d = data as PfdNodeData
  const openExpert = useAppStore((s) => s.openExpert)
  const twin = useAppStore((s) => s.twin)

  const live =
    d.kind === 'reactor'
      ? `T=${(twin?.trend_point?.['RX.T'] ?? 75).toFixed(1)}°C`
      : d.kind === 'mfc'
        ? `F=${(twin?.trend_point?.['H2.MFC'] ?? 0).toFixed(2)} kg/min`
        : d.kind === 'tcu'
          ? `Tj=${(twin?.trend_point?.['RX.TJ'] ?? 90).toFixed(1)}°C`
          : d.subtitle

  return (
    <div
      className={`pfd-node ${d.kind} ${selected ? 'selected' : ''}`}
      onDoubleClick={(e) => {
        e.stopPropagation()
        openExpert(d.equipmentId)
      }}
      title="Double-click to open Input Expert"
    >
      <Handle type="target" position={Position.Left} style={{ background: '#00d4ff' }} />
      <div className="hd">
        <span>{d.title}</span>
        <span className="tag">{id}</span>
      </div>
      <div className="bd">
        <div>{d.subtitle}</div>
        <div style={{ marginTop: 4, color: '#00d4ff', fontFamily: 'ui-monospace, monospace' }}>{live}</div>
      </div>
      <Handle type="source" position={Position.Right} style={{ background: '#3ddc84' }} />
    </div>
  )
}
