import { create } from 'zustand'

export type SafetyStatus = 'NOMINAL' | 'WARNING' | 'CRITICAL'
export type SimPhase = 'idle' | 'starting' | 'running' | 'pausing' | 'paused' | 'resetting' | 'jumping'
export type PrimaryView = 'process' | 'analytics' | 'safety'
export type BottomPanel = 'console' | 'plots' | 'hidden'
export type SpeedX = 1 | 5 | 10

export type EquipmentId =
  | 'EQ-STBR-5000L'
  | 'EQ-TCU-SF-01'
  | 'EQ-MFC-GAS-01'
  | 'EQ-ANF-3L'
  | 'EQ-WFE-01'

export interface EquipmentAsset {
  id: EquipmentId
  name: string
  kind: 'reactor' | 'tcu' | 'mfc' | 'filter' | 'wfe'
  packagePath: string
}

export interface TwinFrame {
  t_s: number
  t_min: number
  stage: string
  stage_index: number
  safety_status: SafetyStatus
  scada_badge: string
  batch_id: string
  metrics: Array<{ tag: string; value: number; unit: string; display_name: string; precision: number }>
  trend_point: Record<string, number>
  overlays?: {
    physics?: Record<string, number>
    ekf_fused?: Record<string, number>
    T_jacket_c?: number
    H2_MFC_kg_min?: number
  }
  ekf?: { confidence_score: number; sensor_residuals?: Record<string, number> }
  decision?: {
    safety_status: SafetyStatus
    active_interlocks: string[]
    optimization_recommendations: string[]
    alarms: string[]
  }
  residual?: {
    is_pure_physics?: boolean
    lab_dataset_present?: boolean
    mode?: string
    delta?: Record<string, number>
  }
  ai_advisory?: { advisory_text: string; source: string }
}

export interface LogLine {
  ts: string
  level: 'info' | 'warn' | 'err' | 'ok'
  message: string
}

export interface ExpertDraft {
  equipmentId: EquipmentId
  connections: {
    inletStream: string
    outletStream: string
    energyStream: string
    stagePreference: string
  }
  heatTransfer: {
    U_Wm2K: number
    area_m2: number
    jacketVolume_m3: number
    coolantInlet_C: number
  }
  kinetics: {
    T_sp_C: number
    P_sp_bar: number
    agitator_rpm: number
    catalyst_kg: number
  }
  initialConditions: {
    volume_m3: number
    T0_C: number
    P0_bar: number
    C_nitro: number
  }
}

export type HistoryBag = {
  t: number[]
  cN_phy: number[]
  cN_ekf: number[]
  cX_phy: number[]
  cX_ekf: number[]
  T_phy: number[]
  T_ekf: number[]
  Tj: number[]
  P: number[]
  mfc: number[]
}

interface AppState {
  selectedEquipmentId: EquipmentId | null
  expertOpen: boolean
  expertPage: number
  expertDraft: ExpertDraft | null
  twin: TwinFrame | null
  wsState: 'connecting' | 'live' | 'offline'
  simPhase: SimPhase
  primaryView: PrimaryView
  bottomPanel: BottomPanel
  speedX: SpeedX
  purePhysics: boolean
  jumpMinutes: string
  logs: LogLine[]
  history: HistoryBag
  openExpert: (id: EquipmentId) => void
  closeExpert: () => void
  setExpertPage: (page: number) => void
  patchExpertDraft: (patch: Partial<ExpertDraft>) => void
  setTwin: (frame: TwinFrame) => void
  hydrateTwin: (frame: TwinFrame) => void
  replaceHistoryFromKeyframes: (keyframes: Array<Record<string, number>>) => void
  setWsState: (s: AppState['wsState']) => void
  setSimPhase: (p: SimPhase) => void
  setPrimaryView: (v: PrimaryView) => void
  setBottomPanel: (b: BottomPanel) => void
  setSpeedX: (s: SpeedX) => void
  setPurePhysics: (v: boolean) => void
  setJumpMinutes: (v: string) => void
  pushLog: (level: LogLine['level'], message: string) => void
  clearHistory: () => void
  selectEquipment: (id: EquipmentId | null) => void
}

const emptyHistory = (): HistoryBag => ({
  t: [],
  cN_phy: [],
  cN_ekf: [],
  cX_phy: [],
  cX_ekf: [],
  T_phy: [],
  T_ekf: [],
  Tj: [],
  P: [],
  mfc: [],
})

export const EQUIPMENT: EquipmentAsset[] = [
  { id: 'EQ-STBR-5000L', name: 'STBR-5000L Jacketed Reactor', kind: 'reactor', packagePath: 'module_1_reactors/eq_stbr_5000l.json' },
  { id: 'EQ-TCU-SF-01', name: 'TCU-SF-01 Thermal Control Unit', kind: 'tcu', packagePath: 'module_2_thermal_control/eq_tcu_sf_01.json' },
  { id: 'EQ-MFC-GAS-01', name: 'MFC-Gas-01 Hydrogen Dosing', kind: 'mfc', packagePath: 'module_3_dosing_flow/eq_mfc_gas_01.json' },
  { id: 'EQ-ANF-3L', name: 'ANF-3L Agitated Nutsche Filter', kind: 'filter', packagePath: 'module_4_catalyst_solids/eq_anf_3l.json' },
  { id: 'EQ-WFE-01', name: 'WFE-01 Wiped Film Evaporator', kind: 'wfe', packagePath: 'module_5_purification/eq_wfe_01.json' },
]

function defaultDraft(id: EquipmentId): ExpertDraft {
  return {
    equipmentId: id,
    connections: {
      inletStream: id === 'EQ-MFC-GAS-01' ? 'H2_FEED' : 'NITRO_CHARGE',
      outletStream: id === 'EQ-STBR-5000L' ? 'RXN_SLURRY' : 'PROCESS_OUT',
      energyStream: id === 'EQ-TCU-SF-01' ? 'Q_JACKET' : 'Q_NONE',
      stagePreference: 'Auto',
    },
    heatTransfer: { U_Wm2K: 400, area_m2: 55, jacketVolume_m3: 0.45, coolantInlet_C: 20 },
    kinetics: { T_sp_C: 85, P_sp_bar: 10, agitator_rpm: 180, catalyst_kg: 12.5 },
    initialConditions: { volume_m3: 3.2, T0_C: 75, P0_bar: 1, C_nitro: 2.8 },
  }
}

const MAX = 800

function appendPoint(h: HistoryBag, frame: TwinFrame): HistoryBag {
  const ov = frame.overlays || {}
  const phy = ov.physics || {}
  const ekf = ov.ekf_fused || {}
  return {
    t: [...h.t, frame.t_min].slice(-MAX),
    cN_phy: [...h.cN_phy, phy.C_nitro ?? frame.trend_point['PHY.C_NITRO'] ?? 0].slice(-MAX),
    cN_ekf: [...h.cN_ekf, ekf.C_nitro ?? frame.trend_point['EKF.C_NITRO'] ?? 0].slice(-MAX),
    cX_phy: [...h.cX_phy, phy.C_xylidine ?? frame.trend_point['PHY.C_XYL'] ?? 0].slice(-MAX),
    cX_ekf: [...h.cX_ekf, ekf.C_xylidine ?? frame.trend_point['EKF.C_XYL'] ?? 0].slice(-MAX),
    T_phy: [...h.T_phy, phy.T_reactor_c ?? frame.trend_point['PHY.T'] ?? 0].slice(-MAX),
    T_ekf: [...h.T_ekf, ekf.T_reactor_c ?? frame.trend_point['EKF.T'] ?? 0].slice(-MAX),
    Tj: [...h.Tj, ov.T_jacket_c ?? frame.trend_point['RX.TJ'] ?? 0].slice(-MAX),
    P: [...h.P, ekf.P_headspace_bar ?? frame.trend_point['RX.P'] ?? 0].slice(-MAX),
    mfc: [...h.mfc, ov.H2_MFC_kg_min ?? frame.trend_point['H2.MFC'] ?? 0].slice(-MAX),
  }
}

export const useAppStore = create<AppState>((set, get) => ({
  selectedEquipmentId: null,
  expertOpen: false,
  expertPage: 0,
  expertDraft: null,
  twin: null,
  wsState: 'connecting',
  simPhase: 'idle',
  primaryView: 'process',
  bottomPanel: 'console',
  speedX: 1,
  purePhysics: true,
  jumpMinutes: '32.5',
  logs: [],
  history: emptyHistory(),
  openExpert: (id) =>
    set({
      selectedEquipmentId: id,
      expertOpen: true,
      expertPage: 0,
      expertDraft: defaultDraft(id),
    }),
  closeExpert: () => set({ expertOpen: false }),
  setExpertPage: (page) => set({ expertPage: page }),
  patchExpertDraft: (patch) => {
    const cur = get().expertDraft
    if (!cur) return
    set({ expertDraft: { ...cur, ...patch } as ExpertDraft })
  },
  selectEquipment: (id) => set({ selectedEquipmentId: id }),
  setWsState: (wsState) => set({ wsState }),
  setSimPhase: (simPhase) => set({ simPhase }),
  setPrimaryView: (primaryView) => set({ primaryView }),
  setBottomPanel: (bottomPanel) => set({ bottomPanel }),
  setSpeedX: (speedX) => set({ speedX }),
  setPurePhysics: (purePhysics) => set({ purePhysics }),
  setJumpMinutes: (jumpMinutes) => set({ jumpMinutes }),
  clearHistory: () => set({ history: emptyHistory() }),
  pushLog: (level, message) =>
    set((s) => ({
      logs: [...s.logs.slice(-400), { ts: new Date().toISOString().slice(11, 19), level, message }],
    })),
  setTwin: (frame) =>
    set((s) => ({
      twin: frame,
      history: appendPoint(s.history, frame),
      purePhysics: frame.residual?.is_pure_physics ?? s.purePhysics,
    })),
  hydrateTwin: (frame) =>
    set((s) => ({
      twin: frame,
      purePhysics: frame.residual?.is_pure_physics ?? s.purePhysics,
    })),
  replaceHistoryFromKeyframes: (keyframes) => {
    let h = emptyHistory()
    for (const kp of keyframes) {
      const fake: TwinFrame = {
        t_s: kp.t_s ?? 0,
        t_min: kp.t_min ?? (kp.t_s ?? 0) / 60,
        stage: '',
        stage_index: 0,
        safety_status: 'NOMINAL',
        scada_badge: 'RUN',
        batch_id: '',
        metrics: [],
        trend_point: kp,
        overlays: {
          physics: {
            C_nitro: kp['PHY.C_NITRO'],
            C_xylidine: kp['PHY.C_XYL'],
            T_reactor_c: kp['PHY.T'],
            P_headspace_bar: kp['RX.P'],
          },
          ekf_fused: {
            C_nitro: kp['EKF.C_NITRO'],
            C_xylidine: kp['EKF.C_XYL'],
            T_reactor_c: kp['EKF.T'],
            P_headspace_bar: kp['RX.P'],
          },
          T_jacket_c: kp['RX.TJ'],
          H2_MFC_kg_min: kp['H2.MFC'],
        },
      }
      h = appendPoint(h, fake)
    }
    set({ history: h })
  },
}))
