import { useCallback, useMemo } from 'react'
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  MarkerType,
  type Edge,
  type Node,
  type NodeTypes,
} from '@xyflow/react'
import { EquipmentNode, type PfdNodeData } from './EquipmentNode'
import { useAppStore } from '@/store/appStore'

const nodeTypes: NodeTypes = {
  equipment: EquipmentNode,
}

const initialNodes: Node<PfdNodeData>[] = [
  {
    id: 'STBR',
    type: 'equipment',
    position: { x: 280, y: 180 },
    data: {
      equipmentId: 'EQ-STBR-5000L',
      title: 'STBR-5000L',
      subtitle: 'Jacketed Semi-Batch Reactor',
      kind: 'reactor',
    },
  },
  {
    id: 'TCU',
    type: 'equipment',
    position: { x: 280, y: 20 },
    data: {
      equipmentId: 'EQ-TCU-SF-01',
      title: 'TCU-SF-01',
      subtitle: 'Thermal Control Loop',
      kind: 'tcu',
    },
  },
  {
    id: 'MFC',
    type: 'equipment',
    position: { x: 40, y: 180 },
    data: {
      equipmentId: 'EQ-MFC-GAS-01',
      title: 'MFC-GAS-01',
      subtitle: 'H₂ Gas Dosing',
      kind: 'mfc',
    },
  },
  {
    id: 'ANF',
    type: 'equipment',
    position: { x: 540, y: 180 },
    data: {
      equipmentId: 'EQ-ANF-3L',
      title: 'ANF-3L',
      subtitle: 'Agitated Nutsche Filter',
      kind: 'filter',
    },
  },
  {
    id: 'WFE',
    type: 'equipment',
    position: { x: 780, y: 180 },
    data: {
      equipmentId: 'EQ-WFE-01',
      title: 'WFE-01',
      subtitle: 'Wiped Film Evaporator',
      kind: 'wfe',
    },
  },
]

function useAnimatedEdges(running: boolean): Edge[] {
  return useMemo(
    () => [
      {
        id: 'e-mfc-stbr',
        source: 'MFC',
        target: 'STBR',
        label: 'H₂ (g)',
        animated: running,
        style: { stroke: '#00d4ff', strokeWidth: 2.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#00d4ff' },
      },
      {
        id: 'e-tcu-stbr',
        source: 'TCU',
        target: 'STBR',
        label: 'Coolant',
        animated: running,
        style: { stroke: '#f5a524', strokeWidth: 2, strokeDasharray: '6 3' },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#f5a524' },
      },
      {
        id: 'e-stbr-anf',
        source: 'STBR',
        target: 'ANF',
        label: 'Slurry',
        animated: running,
        style: { stroke: '#3ddc84', strokeWidth: 2.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#3ddc84' },
      },
      {
        id: 'e-anf-wfe',
        source: 'ANF',
        target: 'WFE',
        label: 'Filtrate',
        animated: running,
        style: { stroke: '#c084fc', strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#c084fc' },
      },
    ],
    [running],
  )
}

export function PfdCanvas() {
  const running = useAppStore((s) => s.simPhase === 'running')
  const openExpert = useAppStore((s) => s.openExpert)
  const selectEquipment = useAppStore((s) => s.selectEquipment)
  const edges = useAnimatedEdges(running)

  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const data = node.data as PfdNodeData
      openExpert(data.equipmentId)
    },
    [openExpert],
  )

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const data = node.data as PfdNodeData
      selectEquipment(data.equipmentId)
    },
    [selectEquipment],
  )

  return (
    <div style={{ width: '100%', height: '100%', background: '#15181c' }}>
      <ReactFlow
        defaultNodes={initialNodes}
        defaultEdges={edges}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        onNodeDoubleClick={onNodeDoubleClick}
        onNodeClick={onNodeClick}
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
      >
        <Background color="#2a3038" gap={18} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(n) => {
            const k = (n.data as PfdNodeData)?.kind
            if (k === 'reactor') return '#4a7ab0'
            if (k === 'tcu') return '#b07a4a'
            if (k === 'mfc') return '#4ab08a'
            if (k === 'filter') return '#8a6ab0'
            return '#a0a04a'
          }}
          maskColor="rgba(0,0,0,0.55)"
        />
      </ReactFlow>
      <div
        style={{
          position: 'absolute',
          left: 10,
          bottom: 10,
          background: 'rgba(30,35,40,0.9)',
          border: '1px solid #3d444d',
          padding: '4px 8px',
          fontSize: 10,
          color: '#9aa3ad',
          pointerEvents: 'none',
        }}
      >
        Double-click equipment → Aspen Input Expert · Animated pipes = live stream
      </div>
    </div>
  )
}
