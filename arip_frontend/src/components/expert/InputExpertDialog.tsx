import { useAppStore, EQUIPMENT } from '@/store/appStore'

const PAGES = [
  { key: 'connections', title: 'Connections', blurb: 'Inlet / outlet / energy stream routing' },
  { key: 'heat', title: 'Heat Transfer Parameters', blurb: 'UA, jacket volume, coolant inlet' },
  { key: 'kinetics', title: 'Kinetic Boundaries', blurb: 'T/P setpoints, agitator, catalyst' },
  { key: 'ic', title: 'Initial Conditions', blurb: 'Charge volume, T0, P0, concentrations' },
  { key: 'review', title: 'Review & Apply', blurb: 'Confirm package bindings before apply' },
] as const

export function InputExpertDialog() {
  const open = useAppStore((s) => s.expertOpen)
  const page = useAppStore((s) => s.expertPage)
  const draft = useAppStore((s) => s.expertDraft)
  const closeExpert = useAppStore((s) => s.closeExpert)
  const setExpertPage = useAppStore((s) => s.setExpertPage)
  const pushLog = useAppStore((s) => s.pushLog)

  if (!open || !draft) return null
  const asset = EQUIPMENT.find((e) => e.id === draft.equipmentId)

  const apply = () => {
    pushLog(
      'ok',
      `Input Expert APPLY → ${draft.equipmentId} | Tin=${draft.connections.inletStream} Tout=${draft.connections.outletStream} Tsp=${draft.kinetics.T_sp_C}°C Psp=${draft.kinetics.P_sp_bar} bar`,
    )
    // Push kinetic setpoints into live twin runtime
    void fetch('/api/v1/twin/setpoints', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        T_sp_c: draft.kinetics.T_sp_C,
        P_sp_bar: draft.kinetics.P_sp_bar,
        agitator_rpm: draft.kinetics.agitator_rpm,
        T_jacket_c: draft.heatTransfer.coolantInlet_C + 15,
      }),
    })
    closeExpert()
  }

  return (
    <div className="expert-backdrop" onClick={closeExpert}>
      <div className="expert-window" onClick={(e) => e.stopPropagation()}>
        <div className="expert-title">
          <span>
            Input Expert — {asset?.name ?? draft.equipmentId}{' '}
            <span style={{ opacity: 0.8, fontWeight: 400 }}>({draft.equipmentId})</span>
          </span>
          <button
            onClick={closeExpert}
            style={{ background: 'transparent', border: 0, color: 'white', cursor: 'pointer', fontSize: 16 }}
          >
            ×
          </button>
        </div>

        <div className="expert-body">
          <div className="expert-nav">
            {PAGES.map((p, i) => (
              <div
                key={p.key}
                className={`expert-nav-item ${page === i ? 'active' : ''}`}
                onClick={() => setExpertPage(i)}
              >
                {p.title}
                <span className="page">
                  {i + 1}/{PAGES.length}
                </span>
              </div>
            ))}
          </div>

          <div className="expert-content">
            <h3>
              {PAGES[page].title}{' '}
              <span style={{ color: '#666', fontWeight: 400, fontSize: 12 }}>
                — {PAGES[page].blurb}
              </span>
            </h3>

            {page === 0 && (
              <>
                <Field label="Inlet Stream">
                  <select
                    value={draft.connections.inletStream}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        connections: { ...draft.connections, inletStream: e.target.value },
                      })
                    }
                  >
                    <option>NITRO_CHARGE</option>
                    <option>H2_FEED</option>
                    <option>SOLVENT_MAKEUP</option>
                    <option>RECYCLE_FILTRATE</option>
                  </select>
                </Field>
                <Field label="Outlet Stream">
                  <select
                    value={draft.connections.outletStream}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        connections: { ...draft.connections, outletStream: e.target.value },
                      })
                    }
                  >
                    <option>RXN_SLURRY</option>
                    <option>PROCESS_OUT</option>
                    <option>VENT_HEADER</option>
                    <option>FILTRATE</option>
                  </select>
                </Field>
                <Field label="Energy Stream">
                  <select
                    value={draft.connections.energyStream}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        connections: { ...draft.connections, energyStream: e.target.value },
                      })
                    }
                  >
                    <option>Q_JACKET</option>
                    <option>Q_Condenser</option>
                    <option>Q_Reboiler</option>
                    <option>Q_NONE</option>
                  </select>
                </Field>
                <Field label="Stage Numbering">
                  <select
                    value={draft.connections.stagePreference}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        connections: { ...draft.connections, stagePreference: e.target.value },
                      })
                    }
                  >
                    <option>Auto</option>
                    <option>Top → Bottom</option>
                    <option>Bottom → Top</option>
                  </select>
                </Field>
              </>
            )}

            {page === 1 && (
              <>
                <Field label="U [W/m²K]">
                  <input
                    type="number"
                    value={draft.heatTransfer.U_Wm2K}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        heatTransfer: { ...draft.heatTransfer, U_Wm2K: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
                <Field label="Area [m²]">
                  <input
                    type="number"
                    value={draft.heatTransfer.area_m2}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        heatTransfer: { ...draft.heatTransfer, area_m2: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
                <Field label="Jacket Volume [m³]">
                  <input
                    type="number"
                    step="0.01"
                    value={draft.heatTransfer.jacketVolume_m3}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        heatTransfer: {
                          ...draft.heatTransfer,
                          jacketVolume_m3: Number(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
                <Field label="Coolant Inlet [°C]">
                  <input
                    type="number"
                    value={draft.heatTransfer.coolantInlet_C}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        heatTransfer: {
                          ...draft.heatTransfer,
                          coolantInlet_C: Number(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
              </>
            )}

            {page === 2 && (
              <>
                <Field label="T setpoint [°C]">
                  <input
                    type="number"
                    value={draft.kinetics.T_sp_C}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        kinetics: { ...draft.kinetics, T_sp_C: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
                <Field label="P setpoint [bar]">
                  <input
                    type="number"
                    step="0.1"
                    value={draft.kinetics.P_sp_bar}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        kinetics: { ...draft.kinetics, P_sp_bar: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
                <Field label="Agitator [RPM]">
                  <input
                    type="number"
                    value={draft.kinetics.agitator_rpm}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        kinetics: { ...draft.kinetics, agitator_rpm: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
                <Field label="Catalyst [kg]">
                  <input
                    type="number"
                    step="0.1"
                    value={draft.kinetics.catalyst_kg}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        kinetics: { ...draft.kinetics, catalyst_kg: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
              </>
            )}

            {page === 3 && (
              <>
                <Field label="Charge Volume [m³]">
                  <input
                    type="number"
                    step="0.1"
                    value={draft.initialConditions.volume_m3}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        initialConditions: {
                          ...draft.initialConditions,
                          volume_m3: Number(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
                <Field label="T₀ [°C]">
                  <input
                    type="number"
                    value={draft.initialConditions.T0_C}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        initialConditions: {
                          ...draft.initialConditions,
                          T0_C: Number(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
                <Field label="P₀ [bar]">
                  <input
                    type="number"
                    step="0.1"
                    value={draft.initialConditions.P0_bar}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        initialConditions: {
                          ...draft.initialConditions,
                          P0_bar: Number(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
                <Field label="C_nitro [mol/L]">
                  <input
                    type="number"
                    step="0.01"
                    value={draft.initialConditions.C_nitro}
                    onChange={(e) =>
                      useAppStore.getState().patchExpertDraft({
                        initialConditions: {
                          ...draft.initialConditions,
                          C_nitro: Number(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
              </>
            )}

            {page === 4 && (
              <div style={{ fontSize: 12, lineHeight: 1.5 }}>
                <p>
                  <b>Package:</b> {asset?.packagePath}
                </p>
                <p>
                  <b>Streams:</b> {draft.connections.inletStream} → unit →{' '}
                  {draft.connections.outletStream} · Energy {draft.connections.energyStream}
                </p>
                <p>
                  <b>UA:</b> {draft.heatTransfer.U_Wm2K} W/m²K · A={draft.heatTransfer.area_m2} m²
                </p>
                <p>
                  <b>Setpoints:</b> T={draft.kinetics.T_sp_C} °C · P={draft.kinetics.P_sp_bar} bar ·{' '}
                  {draft.kinetics.agitator_rpm} RPM
                </p>
                <p style={{ color: '#666', marginTop: 12 }}>
                  Apply writes kinetic setpoints to the live twin runtime and logs the configuration
                  transaction. This dialog mirrors Aspen-style step experts (Connections → Heat →
                  Kinetics → IC → Review).
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="expert-footer">
          <div style={{ color: '#555', fontSize: 11 }}>
            Page {page + 1} of {PAGES.length}
          </div>
          <div className="right">
            <button onClick={closeExpert}>Cancel</button>
            <button disabled={page === 0} onClick={() => setExpertPage(Math.max(0, page - 1))}>
              &lt; Prev
            </button>
            <button
              disabled={page === PAGES.length - 1}
              onClick={() => setExpertPage(Math.min(PAGES.length - 1, page + 1))}
            >
              Next &gt;
            </button>
            <button className="primary" onClick={apply}>
              Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="expert-field">
      <label>{label}</label>
      {children}
    </div>
  )
}
