"""
Operational anomaly injection for EQ-STBR-5000L batch telemetry.

Supports single-fault and compound (combined) anomalies plus operator
shift-note generation for ~10% of batches. Designed to mutate clean or
noisy time-series dictionaries (ODE / SCADA streams) over specified
time windows, and optionally retune ``ReactorEquipmentPackage`` UA /
agitator derates for physics-level anomaly studies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from equipment import ReactorEquipmentPackage

Number = Union[int, float]
TimeSeriesDict = Mapping[str, Any]
MutableTimeSeries = dict[str, Any]

# ---------------------------------------------------------------------------
# Key aliases (aligned with stochastic.py)
# ---------------------------------------------------------------------------
_TIME_KEYS = ("time", "t", "time_s", "timestamp")
_T_REACTOR_KEYS = ("T_reactor", "T_rx", "T_rx_C", "temperature", "temp_reactor")
_T_JACKET_KEYS = ("T_jacket", "T_jkt", "T_jkt_C", "temp_jacket", "jacket_temp")
_TORQUE_KEYS = ("agitator_torque", "torque", "torque_Nm", "agitator_torque_Nm")
_POWER_KEYS = ("agitator_power_kW", "power_kW", "agitator_power", "P_agitator_kW")
_H2_FLOW_KEYS = (
    "h2_flow",
    "H2_flow",
    "m_dot_H2",
    "h2_flow_kg_min",
    "H2_feed_kg_min",
)
_UA_KEYS = ("UA", "UA_W_K", "ua", "ua_W_K")
_Q_COOL_KEYS = ("Q_cooling", "Q_cooling_W", "q_cooling", "cooling_duty_W")
_CONVERSION_KEYS = ("conversion", "X", "X_liquid", "liquid_conversion")

SHIFT_NOTE_PROBABILITY = 0.10


class AnomalyType(str, Enum):
    """Named operational anomalies for single and compound injection."""

    NONE = "NONE"
    SURFACE_FOAMING = "SURFACE_FOAMING"
    COOLING_SPIKE = "COOLING_SPIKE"
    FEED_PAUSE = "FEED_PAUSE"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    COMPOUND_FOAM_COOLING = "COMPOUND_FOAM_COOLING"
    COMPOUND_CASCADE = "COMPOUND_CASCADE"

    @property
    def is_compound(self) -> bool:
        return self in {
            AnomalyType.COMPOUND_FOAM_COOLING,
            AnomalyType.COMPOUND_CASCADE,
        }


# ---------------------------------------------------------------------------
# Anomaly magnitude defaults (exact specs)
# ---------------------------------------------------------------------------
FOAMING_TORQUE_DROP = 0.12  # abrupt −12% torque / power
COOLING_UA_DROP = 0.30  # −30% U·A
COOLING_SPIKE_DURATION_MIN = (15.0, 25.0)  # minutes
FEED_PAUSE_DURATION_MIN = 15.0
SENSOR_DRIFT_BIAS_C = 0.5  # +0.5 °C linear probe bias
CASCADE_FEED_DELAY_MIN = 5.0  # COOLING_SPIKE → FEED_PAUSE lag


def _first_present(data: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    return None


def _as_float_list(values: Iterable[Number]) -> list[float]:
    return [float(v) for v in values]


def _resolve_anomaly(flag: Union[str, AnomalyType, None]) -> AnomalyType:
    if flag is None:
        return AnomalyType.NONE
    if isinstance(flag, AnomalyType):
        return flag
    try:
        return AnomalyType(str(flag).upper())
    except ValueError as exc:
        raise ValueError(
            f"Unknown anomaly_flag '{flag}'. "
            f"Expected one of {[a.value for a in AnomalyType]}"
        ) from exc


def _infer_times_and_dt(
    data: Mapping[str, Any],
) -> tuple[Optional[str], list[float], float]:
    t_key = _first_present(data, _TIME_KEYS)
    if t_key is None:
        # Infer length from any series
        n = 0
        for v in data.values():
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                n = max(n, len(v))
        times = [float(i) for i in range(n)]
        return None, times, 1.0
    times = _as_float_list(data[t_key])
    if len(times) >= 2:
        dts = [times[i] - times[i - 1] for i in range(1, len(times))]
        positive = [d for d in dts if d > 0.0]
        dt = sum(positive) / len(positive) if positive else 1.0
    else:
        dt = 1.0
    return t_key, times, dt


def _window_mask(
    times: Sequence[float],
    t_start: Optional[float],
    t_end: Optional[float],
) -> list[bool]:
    """Inclusive start, exclusive end; open bounds → full span."""
    t0 = times[0] if t_start is None else float(t_start)
    t1 = times[-1] + 1e-9 if t_end is None else float(t_end)
    return [t0 <= t < t1 for t in times]


def _ramp_01(times: Sequence[float], mask: Sequence[bool]) -> list[float]:
    """Linear 0→1 progress across the masked window (0 outside)."""
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return [0.0] * len(times)
    t_a, t_b = times[idx[0]], times[idx[-1]]
    span = max(t_b - t_a, 1e-12)
    out: list[float] = []
    for i, t in enumerate(times):
        if not mask[i]:
            out.append(0.0)
        else:
            out.append((t - t_a) / span)
    return out


def _peak_reaction_window(
    times: Sequence[float],
    data: Mapping[str, Any],
    duration_s: float,
) -> tuple[float, float]:
    """
    Place a window of ``duration_s`` at peak reaction rate.

    Uses max |dX/dt| when conversion is present; otherwise centers on
    the middle third of the batch (typical exotherm peak).
    """
    t0, t1 = times[0], times[-1]
    batch_span = max(t1 - t0, 1e-12)
    x_key = _first_present(data, _CONVERSION_KEYS)
    if x_key is not None and len(data[x_key]) == len(times) and len(times) >= 3:
        X = _as_float_list(data[x_key])
        rates = [
            abs(X[i] - X[i - 1]) / max(times[i] - times[i - 1], 1e-12)
            for i in range(1, len(X))
        ]
        i_peak = 1 + max(range(len(rates)), key=lambda i: rates[i])
        t_center = times[i_peak]
    else:
        t_center = t0 + 0.40 * batch_span  # early–mid peak exotherm
    half = 0.5 * duration_s
    start = max(t0, t_center - half)
    end = min(t1, start + duration_s)
    start = max(t0, end - duration_s)
    return start, end


# ===========================================================================
# Shift notes
# ===========================================================================

_SHIFT_NOTE_TEMPLATES: dict[AnomalyType, tuple[str, ...]] = {
    AnomalyType.NONE: (
        "Batch ran to recipe. No process deviations noted.",
        "Uneventful shift — parameters within normal operating band.",
    ),
    AnomalyType.SURFACE_FOAMING: (
        "Observed surface foaming; agitator load dropped ~12%. Antifoam not charged — monitoring.",
        "Foam head noted mid-batch. Torque step-down on agitator; continued under observation.",
        "Operators report froth on sight glass. Power draw stepped down abruptly.",
    ),
    AnomalyType.COOLING_SPIKE: (
        "Cooling utility temp hunting; jacket duty soft for ~20 min then recovered.",
        "TCU instability — effective UA appeared reduced ~30% during excursion.",
        "Cooling spike / fouling transient. Exotherm briefly elevated; jacket recovered.",
    ),
    AnomalyType.FEED_PAUSE: (
        "H2 feed paused automatically for 15 min mid-batch to control exotherm.",
        "MFC interlock held feed at 0 kg/min for fifteen minutes; resumed per SOP.",
        "Feed pause executed mid-batch. Pressure held; operators verified purge/vent OK.",
    ),
    AnomalyType.SENSOR_DRIFT: (
        "TI-reactor reading trending +0.5°C high vs jacket/redundant probe — suspect drift.",
        "Calibration check flagged: reactor RTD biased high (~0.5°C linear offset).",
        "Sensor drift on T_reactor. Cross-checked with handheld; will recalibrate next turnaround.",
    ),
    AnomalyType.COMPOUND_FOAM_COOLING: (
        "Peak exotherm: surface foaming concurrent with cooling duty loss. Torque −12%, UA soft.",
        "Compound event at peak rate — foaming plus cooling spike. Closely supervised.",
        "Foam + jacket upset during peak reaction. Load drop and cooling transient overlapped.",
    ),
    AnomalyType.COMPOUND_CASCADE: (
        "Cooling utility temp hunting, feed paused automatically to control exotherm.",
        "Cascade: cooling spike tripped auto feed pause (+5 min); T_reactor probe drifted +0.5°C.",
        "Cooling excursion cascaded to H2 hold. Sensor bias noted on reactor temperature.",
    ),
}


def generate_shift_note(
    anomaly_flag: Union[str, AnomalyType, None],
    *,
    always: bool = False,
    probability: float = SHIFT_NOTE_PROBABILITY,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> Optional[str]:
    """
    Return a realistic operator shift-log string for the given anomaly.

    By default emits a note for ~10% of calls (``probability=0.10``),
    matching sparse historian annotation rates. Pass ``always=True`` to
    force a note (useful for labeled test-set compounds).
    """
    anomaly = _resolve_anomaly(anomaly_flag)
    r = rng if rng is not None else random.Random(seed)
    if not always and r.random() >= probability:
        return None
    templates = _SHIFT_NOTE_TEMPLATES.get(anomaly) or _SHIFT_NOTE_TEMPLATES[AnomalyType.NONE]
    return r.choice(templates)


# ===========================================================================
# Anomaly schedule / result metadata
# ===========================================================================


@dataclass
class AnomalyWindow:
    """Time window [t_start, t_end) in the same units as the series time axis."""

    name: str
    t_start: float
    t_end: float
    params: dict[str, float] = field(default_factory=dict)

    def contains(self, t: float) -> bool:
        return self.t_start <= t < self.t_end

    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)


@dataclass
class AnomalyInjectionResult:
    """Payload returned by injection wrappers."""

    anomaly: AnomalyType
    time_series: MutableTimeSeries
    windows: list[AnomalyWindow] = field(default_factory=list)
    shift_note: Optional[str] = None
    components: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "anomaly_flag": self.anomaly.value,
            "components": list(self.components),
            "windows": [
                {
                    "name": w.name,
                    "t_start": w.t_start,
                    "t_end": w.t_end,
                    "params": dict(w.params),
                }
                for w in self.windows
            ],
            "shift_note": self.shift_note,
            "time_series": self.time_series,
        }


# ===========================================================================
# Primitive injectors (mutate series in place)
# ===========================================================================


def _inject_surface_foaming(
    data: MutableTimeSeries,
    times: Sequence[float],
    mask: Sequence[bool],
    drop_fraction: float = FOAMING_TORQUE_DROP,
) -> AnomalyWindow:
    """Abrupt −12% agitator torque / power draw inside the window."""
    scale = 1.0 - drop_fraction
    for keys in (_TORQUE_KEYS, _POWER_KEYS):
        key = _first_present(data, keys)
        if key is None:
            continue
        series = _as_float_list(data[key])
        data[key] = [
            series[i] * scale if mask[i] else series[i] for i in range(len(series))
        ]
    # If torque missing, synthesize a marker channel for downstream models
    if _first_present(data, _TORQUE_KEYS) is None:
        data["agitator_torque_anomaly_scale"] = [
            scale if m else 1.0 for m in mask
        ]
    idx = [i for i, m in enumerate(mask) if m]
    t_start = times[idx[0]] if idx else times[0]
    t_end = times[idx[-1]] if idx else times[0]
    data["anomaly_foaming_active"] = list(mask)
    return AnomalyWindow(
        name=AnomalyType.SURFACE_FOAMING.value,
        t_start=t_start,
        t_end=t_end + 1e-9,
        params={"torque_power_drop_fraction": drop_fraction},
    )


def _inject_cooling_spike(
    data: MutableTimeSeries,
    times: Sequence[float],
    mask: Sequence[bool],
    ua_drop_fraction: float = COOLING_UA_DROP,
    equipment: Optional[ReactorEquipmentPackage] = None,
) -> AnomalyWindow:
    """Temporary −30% U·A (and proportional Q_cooling) inside the window."""
    scale = 1.0 - ua_drop_fraction
    ua_key = _first_present(data, _UA_KEYS)
    if ua_key is not None:
        series = _as_float_list(data[ua_key])
        data[ua_key] = [
            series[i] * scale if mask[i] else series[i] for i in range(len(series))
        ]
    else:
        # Publish UA multiplier schedule for physics integrators
        base_ua = equipment.ua_W_K if equipment is not None else 350.0 * 12.5
        data["UA"] = [base_ua * (scale if m else 1.0) for m in mask]
        ua_key = "UA"

    q_key = _first_present(data, _Q_COOL_KEYS)
    if q_key is not None:
        series = _as_float_list(data[q_key])
        data[q_key] = [
            series[i] * scale if mask[i] else series[i] for i in range(len(series))
        ]

    data["anomaly_ua_scale"] = [scale if m else 1.0 for m in mask]
    data["anomaly_cooling_spike_active"] = list(mask)

    # Optional equipment retune snapshot (callers may restore later)
    if equipment is not None:
        data["equipment_cooling_efficiency_during_spike"] = (
            equipment.cooling_efficiency * scale
        )

    idx = [i for i, m in enumerate(mask) if m]
    t_start = times[idx[0]] if idx else times[0]
    t_end = times[idx[-1]] if idx else times[0]
    return AnomalyWindow(
        name=AnomalyType.COOLING_SPIKE.value,
        t_start=t_start,
        t_end=t_end + 1e-9,
        params={"ua_drop_fraction": ua_drop_fraction, "ua_scale": scale},
    )


def _inject_feed_pause(
    data: MutableTimeSeries,
    times: Sequence[float],
    mask: Sequence[bool],
) -> AnomalyWindow:
    """Force H2 mass-flow to 0 for the masked interval."""
    key = _first_present(data, _H2_FLOW_KEYS)
    if key is None:
        key = "h2_flow"
        data[key] = [0.0 for _ in times]
    series = _as_float_list(data[key])
    data[key] = [0.0 if mask[i] else series[i] for i in range(len(series))]
    data["anomaly_feed_pause_active"] = list(mask)
    idx = [i for i, m in enumerate(mask) if m]
    t_start = times[idx[0]] if idx else times[0]
    t_end = times[idx[-1]] if idx else times[0]
    return AnomalyWindow(
        name=AnomalyType.FEED_PAUSE.value,
        t_start=t_start,
        t_end=t_end + 1e-9,
        params={"h2_flow_kg_min": 0.0},
    )


def _inject_sensor_drift(
    data: MutableTimeSeries,
    times: Sequence[float],
    mask: Sequence[bool],
    bias_C: float = SENSOR_DRIFT_BIAS_C,
) -> AnomalyWindow:
    """
    Apply a linear probe bias on T_reactor ramping 0 → +bias_C over the window.

    Outside the window the accumulated end-of-window bias is held if the
    window ends before batch end (persistent calibration offset); if the
    mask covers the full series, bias simply tracks the linear ramp.
    """
    key = _first_present(data, _T_REACTOR_KEYS)
    if key is None:
        key = "T_reactor"
        data[key] = [0.0 for _ in times]
    series = _as_float_list(data[key])
    ramp = _ramp_01(times, mask)
    idx = [i for i, m in enumerate(mask) if m]
    hold_bias = bias_C if idx else 0.0
    last_mask_i = idx[-1] if idx else -1

    biased: list[float] = []
    bias_trace: list[float] = []
    for i in range(len(series)):
        if mask[i]:
            b = bias_C * ramp[i]
        elif i > last_mask_i >= 0:
            b = hold_bias  # persistent offset after drift develops
        else:
            b = 0.0
        biased.append(series[i] + b)
        bias_trace.append(b)
    data[key] = biased
    data["T_reactor_sensor_bias_C"] = bias_trace
    data["anomaly_sensor_drift_active"] = list(mask)

    t_start = times[idx[0]] if idx else times[0]
    t_end = times[idx[-1]] if idx else times[0]
    return AnomalyWindow(
        name=AnomalyType.SENSOR_DRIFT.value,
        t_start=t_start,
        t_end=t_end + 1e-9,
        params={"bias_C": bias_C},
    )


# ===========================================================================
# Equipment-level helpers
# ===========================================================================


def apply_anomaly_to_equipment(
    equipment: ReactorEquipmentPackage,
    anomaly_flag: Union[str, AnomalyType],
    *,
    active: bool = True,
) -> dict[str, float]:
    """
    Retune equipment package multipliers for physics-level anomaly injection.

    Returns a dict of previous values so callers can restore state::

        prev = apply_anomaly_to_equipment(eq, "COOLING_SPIKE", active=True)
        ...
        eq.configure(**prev)
    """
    anomaly = _resolve_anomaly(anomaly_flag)
    previous = {
        "cooling_efficiency": equipment.cooling_efficiency,
        "agitator_power_derate": equipment.agitator_power_derate,
    }
    if not active or anomaly is AnomalyType.NONE:
        return previous

    if anomaly in (
        AnomalyType.COOLING_SPIKE,
        AnomalyType.COMPOUND_FOAM_COOLING,
        AnomalyType.COMPOUND_CASCADE,
    ):
        equipment.cooling_efficiency = previous["cooling_efficiency"] * (
            1.0 - COOLING_UA_DROP
        )
    if anomaly in (
        AnomalyType.SURFACE_FOAMING,
        AnomalyType.COMPOUND_FOAM_COOLING,
    ):
        equipment.agitator_power_derate = previous["agitator_power_derate"] * (
            1.0 - FOAMING_TORQUE_DROP
        )
    return previous


# ===========================================================================
# Public wrappers
# ===========================================================================


def _copy_series(time_series_dict: TimeSeriesDict) -> MutableTimeSeries:
    out: MutableTimeSeries = {}
    for k, v in time_series_dict.items():
        if isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def inject_anomaly(
    time_series_dict: TimeSeriesDict,
    anomaly_flag: Union[str, AnomalyType],
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    duration_min: Optional[float] = None,
    equipment: Optional[ReactorEquipmentPackage] = None,
    write_shift_note: bool = True,
    shift_note_always: bool = False,
    shift_note_probability: float = SHIFT_NOTE_PROBABILITY,
    rng_seed: Optional[int] = None,
) -> AnomalyInjectionResult:
    """
    Inject a single operational anomaly into ``time_series_dict``.

    Parameters
    ----------
    time_series_dict
        Clean or noisy ODE / SCADA streams (must include a time axis or
        equal-length numeric series).
    anomaly_flag
        One of ``SURFACE_FOAMING``, ``COOLING_SPIKE``, ``FEED_PAUSE``,
        ``SENSOR_DRIFT`` (compounds should use ``inject_compound_anomaly``).
    t_start, t_end
        Window bounds in the series time unit (seconds). If omitted, a
        sensible mid-batch / peak-reaction window is chosen.
    duration_min
        Optional duration override [minutes] used when ``t_end`` is omitted.
    """
    anomaly = _resolve_anomaly(anomaly_flag)
    if anomaly.is_compound:
        return inject_compound_anomaly(
            time_series_dict,
            anomaly,
            t_start=t_start,
            t_end=t_end,
            duration_min=duration_min,
            equipment=equipment,
            write_shift_note=write_shift_note,
            shift_note_always=shift_note_always,
            shift_note_probability=shift_note_probability,
            rng_seed=rng_seed,
        )

    rng = random.Random(rng_seed)
    data = _copy_series(time_series_dict)
    _, times, dt = _infer_times_and_dt(data)
    if not times:
        raise ValueError("time_series_dict has no usable time axis / series length")

    # Resolve default windows
    if t_start is None or (t_end is None and duration_min is None):
        if anomaly is AnomalyType.COOLING_SPIKE:
            dur_min = (
                duration_min
                if duration_min is not None
                else rng.uniform(*COOLING_SPIKE_DURATION_MIN)
            )
            dur_s = dur_min * 60.0
            t_start, t_end = _peak_reaction_window(times, data, dur_s)
        elif anomaly is AnomalyType.FEED_PAUSE:
            dur_s = (duration_min or FEED_PAUSE_DURATION_MIN) * 60.0
            mid = times[0] + 0.5 * (times[-1] - times[0])
            t_start = mid - 0.5 * dur_s if t_start is None else t_start
            t_end = t_start + dur_s
        elif anomaly is AnomalyType.SURFACE_FOAMING:
            # Abrupt event: short window (~2 min) at mid/peak unless specified
            dur_s = (duration_min or 2.0) * 60.0
            t_start, t_end = _peak_reaction_window(times, data, dur_s)
        elif anomaly is AnomalyType.SENSOR_DRIFT:
            # Drift develops over latter half of batch by default
            t_start = times[0] + 0.5 * (times[-1] - times[0])
            t_end = times[-1] + dt
        else:
            t_start, t_end = times[0], times[-1] + dt
    elif t_end is None:
        dur_min = duration_min or {
            AnomalyType.COOLING_SPIKE: rng.uniform(*COOLING_SPIKE_DURATION_MIN),
            AnomalyType.FEED_PAUSE: FEED_PAUSE_DURATION_MIN,
            AnomalyType.SURFACE_FOAMING: 2.0,
            AnomalyType.SENSOR_DRIFT: max(
                (times[-1] - float(t_start)) / 60.0, 1.0
            ),
        }.get(anomaly, 15.0)
        t_end = float(t_start) + float(dur_min) * 60.0

    assert t_start is not None and t_end is not None
    mask = _window_mask(times, t_start, t_end)
    windows: list[AnomalyWindow] = []
    components: list[str] = []

    if anomaly is AnomalyType.NONE:
        pass
    elif anomaly is AnomalyType.SURFACE_FOAMING:
        windows.append(_inject_surface_foaming(data, times, mask))
        components.append(AnomalyType.SURFACE_FOAMING.value)
    elif anomaly is AnomalyType.COOLING_SPIKE:
        windows.append(
            _inject_cooling_spike(data, times, mask, equipment=equipment)
        )
        components.append(AnomalyType.COOLING_SPIKE.value)
    elif anomaly is AnomalyType.FEED_PAUSE:
        windows.append(_inject_feed_pause(data, times, mask))
        components.append(AnomalyType.FEED_PAUSE.value)
    elif anomaly is AnomalyType.SENSOR_DRIFT:
        windows.append(_inject_sensor_drift(data, times, mask))
        components.append(AnomalyType.SENSOR_DRIFT.value)
    else:
        raise ValueError(f"Unsupported single anomaly: {anomaly}")

    note = None
    if write_shift_note:
        note = generate_shift_note(
            anomaly,
            always=shift_note_always,
            probability=shift_note_probability,
            rng=rng,
        )
    data["anomaly_flag"] = anomaly.value
    data["anomaly_components"] = list(components)
    if note is not None:
        data["shift_note"] = note

    return AnomalyInjectionResult(
        anomaly=anomaly,
        time_series=data,
        windows=windows,
        shift_note=note,
        components=components,
    )


def inject_compound_anomaly(
    time_series_dict: TimeSeriesDict,
    anomaly_flag: Union[str, AnomalyType] = AnomalyType.COMPOUND_FOAM_COOLING,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    duration_min: Optional[float] = None,
    equipment: Optional[ReactorEquipmentPackage] = None,
    write_shift_note: bool = True,
    shift_note_always: bool = True,
    shift_note_probability: float = 1.0,
    rng_seed: Optional[int] = None,
) -> AnomalyInjectionResult:
    """
    Inject a compound (combined) anomaly intended for the labeled test set.

    COMPOUND_FOAM_COOLING
        Concurrent SURFACE_FOAMING and COOLING_SPIKE during peak reaction.

    COMPOUND_CASCADE
        COOLING_SPIKE triggers automatic FEED_PAUSE 5 minutes later,
        accompanied by SENSOR_DRIFT.
    """
    anomaly = _resolve_anomaly(anomaly_flag)
    if anomaly not in (
        AnomalyType.COMPOUND_FOAM_COOLING,
        AnomalyType.COMPOUND_CASCADE,
    ):
        raise ValueError(
            f"inject_compound_anomaly expects a compound flag, got {anomaly.value}"
        )

    rng = random.Random(rng_seed)
    data = _copy_series(time_series_dict)
    _, times, dt = _infer_times_and_dt(data)
    if not times:
        raise ValueError("time_series_dict has no usable time axis / series length")

    windows: list[AnomalyWindow] = []
    components: list[str] = []

    if anomaly is AnomalyType.COMPOUND_FOAM_COOLING:
        dur_min = (
            duration_min
            if duration_min is not None
            else rng.uniform(*COOLING_SPIKE_DURATION_MIN)
        )
        dur_s = dur_min * 60.0
        if t_start is None or t_end is None:
            ts, te = _peak_reaction_window(times, data, dur_s)
            t_start = t_start if t_start is not None else ts
            t_end = t_end if t_end is not None else te
        mask = _window_mask(times, t_start, t_end)
        windows.append(_inject_surface_foaming(data, times, mask))
        windows.append(
            _inject_cooling_spike(data, times, mask, equipment=equipment)
        )
        components.extend(
            [
                AnomalyType.SURFACE_FOAMING.value,
                AnomalyType.COOLING_SPIKE.value,
            ]
        )

    elif anomaly is AnomalyType.COMPOUND_CASCADE:
        # 1) Cooling spike
        cool_dur_min = (
            duration_min
            if duration_min is not None
            else rng.uniform(*COOLING_SPIKE_DURATION_MIN)
        )
        cool_dur_s = cool_dur_min * 60.0
        if t_start is None:
            t_cool0, t_cool1 = _peak_reaction_window(times, data, cool_dur_s)
        else:
            t_cool0 = float(t_start)
            t_cool1 = float(t_end) if t_end is not None else t_cool0 + cool_dur_s
        mask_cool = _window_mask(times, t_cool0, t_cool1)
        windows.append(
            _inject_cooling_spike(data, times, mask_cool, equipment=equipment)
        )
        components.append(AnomalyType.COOLING_SPIKE.value)

        # 2) Automatic feed pause 5 min after cooling spike onset
        t_feed0 = t_cool0 + CASCADE_FEED_DELAY_MIN * 60.0
        t_feed1 = t_feed0 + FEED_PAUSE_DURATION_MIN * 60.0
        mask_feed = _window_mask(times, t_feed0, t_feed1)
        windows.append(_inject_feed_pause(data, times, mask_feed))
        components.append(AnomalyType.FEED_PAUSE.value)

        # 3) Sensor drift accompanies the cascade (from cooling onset → batch end)
        mask_drift = _window_mask(times, t_cool0, times[-1] + dt)
        windows.append(_inject_sensor_drift(data, times, mask_drift))
        components.append(AnomalyType.SENSOR_DRIFT.value)

    note = None
    if write_shift_note:
        note = generate_shift_note(
            anomaly,
            always=shift_note_always,
            probability=shift_note_probability,
            rng=rng,
        )

    data["anomaly_flag"] = anomaly.value
    data["anomaly_components"] = list(components)
    if note is not None:
        data["shift_note"] = note

    return AnomalyInjectionResult(
        anomaly=anomaly,
        time_series=data,
        windows=windows,
        shift_note=note,
        components=components,
    )


def inject_anomalies_batch(
    time_series_dict: TimeSeriesDict,
    anomaly_flag: Union[str, AnomalyType],
    **kwargs: Any,
) -> AnomalyInjectionResult:
    """
    Convenience dispatcher: routes compound flags to
    ``inject_compound_anomaly``, otherwise ``inject_anomaly``.
    """
    anomaly = _resolve_anomaly(anomaly_flag)
    if anomaly.is_compound:
        return inject_compound_anomaly(time_series_dict, anomaly, **kwargs)
    return inject_anomaly(time_series_dict, anomaly, **kwargs)


__all__ = [
    "AnomalyType",
    "AnomalyWindow",
    "AnomalyInjectionResult",
    "inject_anomaly",
    "inject_compound_anomaly",
    "inject_anomalies_batch",
    "apply_anomaly_to_equipment",
    "generate_shift_note",
    "FOAMING_TORQUE_DROP",
    "COOLING_UA_DROP",
    "COOLING_SPIKE_DURATION_MIN",
    "FEED_PAUSE_DURATION_MIN",
    "SENSOR_DRIFT_BIAS_C",
    "CASCADE_FEED_DELAY_MIN",
    "SHIFT_NOTE_PROBABILITY",
]
