"""
Step-bounded operational anomalies for EQ-STBR-5000L batches.

Single faults are constrained to allowed process steps. Compound faults are
intended for the labeled TEST set only. Shift notes are emitted for ~10% of
batches, and only reference steps where an anomaly is active.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from equipment import ProcessStep, STEP_BOUNDS

SHIFT_NOTE_PROBABILITY = 0.10

FOAMING_TORQUE_DROP = 0.12
COOLING_UA_DROP = 0.30
COOLING_SPIKE_DURATION_MIN = (15, 25)
FEED_PAUSE_DURATION_MIN = 15
SENSOR_DRIFT_BIAS_C = 0.5
CASCADE_FEED_DELAY_MIN = 5


class AnomalyType(str, Enum):
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


# Steps in which each primitive anomaly is allowed
ALLOWED_STEPS: Dict[AnomalyType, Set[ProcessStep]] = {
    AnomalyType.SURFACE_FOAMING: {ProcessStep.ACTIVE_HYDROGENATION},
    AnomalyType.COOLING_SPIKE: {
        ProcessStep.PRE_HEATING,
        ProcessStep.ACTIVE_HYDROGENATION,
        ProcessStep.DEGASSING_COOLING,
    },
    AnomalyType.FEED_PAUSE: {ProcessStep.ACTIVE_HYDROGENATION},
    AnomalyType.SENSOR_DRIFT: set(ProcessStep),  # batch-wide probe bias
}


@dataclass
class AnomalyWindow:
    name: str
    t_start: int
    t_end: int
    step: ProcessStep
    params: dict = field(default_factory=dict)

    def contains(self, t: int) -> bool:
        return self.t_start <= t < self.t_end


@dataclass
class AnomalySchedule:
    """Resolved minute-level schedules for one batch."""

    anomaly: AnomalyType
    windows: List[AnomalyWindow] = field(default_factory=list)
    ua_scale: Optional[List[float]] = None
    feed_scale: Optional[List[float]] = None
    foam_scale: Optional[List[float]] = None
    sensor_bias_C: Optional[List[float]] = None
    shift_note: Optional[str] = None
    components: List[str] = field(default_factory=list)


_SHIFT_NOTES: Dict[AnomalyType, Tuple[str, ...]] = {
    AnomalyType.SURFACE_FOAMING: (
        "Step 5: Agitator power draw dropped abruptly. Foaming risk flagged.",
        "Step 5: Surface foam observed on sight glass; torque stepped down ~12%.",
        "Step 5: Gas-liquid aeration suspected — agitator load dip noted.",
    ),
    AnomalyType.COOLING_SPIKE: (
        "Cooling utility temp hunting; UA soft for ~20 min then recovered.",
        "TCU valve hunting — jacket duty reduced ~30% during excursion.",
        "Step cooling spike flagged. Exotherm briefly elevated; jacket recovered.",
    ),
    AnomalyType.FEED_PAUSE: (
        "Step 5: H2 feed paused automatically for 15 min to control exotherm.",
        "Step 5: MFC interlock held feed at 0 kg/min for fifteen minutes.",
        "Step 5: Feed pause mid-reaction; kinetics held flat until resume.",
    ),
    AnomalyType.SENSOR_DRIFT: (
        "TI-reactor trending +0.5°C high vs redundant probe — suspect drift.",
        "Calibration flag: reactor RTD biased high (~0.5°C linear offset).",
        "Sensor drift on T_reactor across batch; recalibrate next turnaround.",
    ),
    AnomalyType.COMPOUND_FOAM_COOLING: (
        "Step 5: Foaming concurrent with cooling duty loss. Torque −12%, UA soft.",
        "Step 5 compound event — foam + cooling spike during peak rate.",
        "Step 5: Agitator load drop and jacket upset overlapped at peak exotherm.",
    ),
    AnomalyType.COMPOUND_CASCADE: (
        "Cooling utility temp hunting, feed paused automatically to control exotherm.",
        "Cascade: Step 5 cooling spike tripped auto feed pause (+5 min); T_rx drift +0.5°C.",
        "Step 5 cascade — cooling excursion → H2 hold; reactor probe bias noted.",
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
    Return a short operator log for ~10% of batches (anomaly-active steps).

    Returns ``None`` for normal batches and for the remaining 90%.
    """
    anomaly = _resolve(anomaly_flag)
    if anomaly is AnomalyType.NONE:
        return None
    r = rng if rng is not None else random.Random(seed)
    if not always and r.random() >= probability:
        return None
    templates = _SHIFT_NOTES.get(anomaly)
    if not templates:
        return None
    return r.choice(templates)


def _resolve(flag: Union[str, AnomalyType, None]) -> AnomalyType:
    if flag is None:
        return AnomalyType.NONE
    if isinstance(flag, AnomalyType):
        return flag
    return AnomalyType(str(flag).upper())


def _step_span(step: ProcessStep) -> Tuple[int, int]:
    return STEP_BOUNDS[step]


def build_anomaly_schedule(
    anomaly_flag: Union[str, AnomalyType],
    n_minutes: int,
    *,
    rng_seed: Optional[int] = None,
    write_shift_note: bool = True,
    shift_note_probability: float = SHIFT_NOTE_PROBABILITY,
) -> AnomalySchedule:
    """
    Build minute-resolved UA / feed / foam / sensor-bias schedules.

    Schedules are consumed by the step-state simulator so anomalies affect
    both physics (feed pause, UA drop) and telemetry (torque, probe bias).
    """
    anomaly = _resolve(anomaly_flag)
    rng = random.Random(rng_seed)
    ua = [1.0] * n_minutes
    feed = [1.0] * n_minutes
    foam = [1.0] * n_minutes
    bias = [0.0] * n_minutes
    windows: List[AnomalyWindow] = []
    components: List[str] = []

    def _apply_cooling(step: ProcessStep) -> AnomalyWindow:
        a, b = _step_span(step)
        dur = int(rng.randint(*COOLING_SPIKE_DURATION_MIN))
        # Prefer mid-step placement
        latest_start = max(a, b - dur)
        start = int(rng.randint(a, max(a, latest_start)))
        end = min(b, start + dur)
        for t in range(start, end):
            ua[t] = 1.0 - COOLING_UA_DROP
        w = AnomalyWindow(
            name=AnomalyType.COOLING_SPIKE.value,
            t_start=start,
            t_end=end,
            step=step,
            params={"ua_drop_fraction": COOLING_UA_DROP},
        )
        windows.append(w)
        return w

    def _apply_foaming() -> AnomalyWindow:
        step = ProcessStep.ACTIVE_HYDROGENATION
        a, b = _step_span(step)
        dur = int(rng.randint(8, 16))
        start = int(rng.randint(a + 10, max(a + 10, b - dur - 5)))
        end = min(b, start + dur)
        for t in range(start, end):
            foam[t] = 1.0 - FOAMING_TORQUE_DROP
        w = AnomalyWindow(
            name=AnomalyType.SURFACE_FOAMING.value,
            t_start=start,
            t_end=end,
            step=step,
            params={"torque_drop_fraction": FOAMING_TORQUE_DROP},
        )
        windows.append(w)
        return w

    def _apply_feed_pause(start: Optional[int] = None) -> AnomalyWindow:
        step = ProcessStep.ACTIVE_HYDROGENATION
        a, b = _step_span(step)
        dur = FEED_PAUSE_DURATION_MIN
        if start is None:
            start = int(rng.randint(a + 20, max(a + 20, b - dur - 10)))
        start = int(max(a, min(start, b - 1)))
        end = min(b, start + dur)
        for t in range(start, end):
            feed[t] = 0.0
        w = AnomalyWindow(
            name=AnomalyType.FEED_PAUSE.value,
            t_start=start,
            t_end=end,
            step=step,
            params={"h2_flow_kg_min": 0.0},
        )
        windows.append(w)
        return w

    def _apply_sensor_drift(t0: int = 0) -> AnomalyWindow:
        # Linear 0 → +0.5 °C from t0 to end of batch
        span = max(n_minutes - 1 - t0, 1)
        for t in range(t0, n_minutes):
            bias[t] = SENSOR_DRIFT_BIAS_C * (t - t0) / span
        w = AnomalyWindow(
            name=AnomalyType.SENSOR_DRIFT.value,
            t_start=t0,
            t_end=n_minutes,
            step=ProcessStep.PREPARATION_TARE,
            params={"bias_C": SENSOR_DRIFT_BIAS_C},
        )
        windows.append(w)
        return w

    if anomaly is AnomalyType.NONE:
        pass
    elif anomaly is AnomalyType.SURFACE_FOAMING:
        _apply_foaming()
        components.append(AnomalyType.SURFACE_FOAMING.value)
    elif anomaly is AnomalyType.COOLING_SPIKE:
        # Prefer Step 5; occasionally Step 4 or 7
        step = rng.choice(
            [
                ProcessStep.ACTIVE_HYDROGENATION,
                ProcessStep.ACTIVE_HYDROGENATION,
                ProcessStep.PRE_HEATING,
                ProcessStep.DEGASSING_COOLING,
            ]
        )
        _apply_cooling(step)
        components.append(AnomalyType.COOLING_SPIKE.value)
    elif anomaly is AnomalyType.FEED_PAUSE:
        _apply_feed_pause()
        components.append(AnomalyType.FEED_PAUSE.value)
    elif anomaly is AnomalyType.SENSOR_DRIFT:
        _apply_sensor_drift(0)
        components.append(AnomalyType.SENSOR_DRIFT.value)
    elif anomaly is AnomalyType.COMPOUND_FOAM_COOLING:
        _apply_foaming()
        _apply_cooling(ProcessStep.ACTIVE_HYDROGENATION)
        components.extend(
            [AnomalyType.SURFACE_FOAMING.value, AnomalyType.COOLING_SPIKE.value]
        )
    elif anomaly is AnomalyType.COMPOUND_CASCADE:
        cool = _apply_cooling(ProcessStep.ACTIVE_HYDROGENATION)
        _apply_feed_pause(start=cool.t_start + CASCADE_FEED_DELAY_MIN)
        _apply_sensor_drift(cool.t_start)
        components.extend(
            [
                AnomalyType.COOLING_SPIKE.value,
                AnomalyType.FEED_PAUSE.value,
                AnomalyType.SENSOR_DRIFT.value,
            ]
        )
    else:
        raise ValueError(f"Unsupported anomaly: {anomaly}")

    note = None
    if write_shift_note and anomaly is not AnomalyType.NONE:
        # Compounds always annotated for test-set clarity; singles ~10%
        always = anomaly.is_compound
        note = generate_shift_note(
            anomaly,
            always=always,
            probability=1.0 if always else shift_note_probability,
            rng=rng,
        )

    return AnomalySchedule(
        anomaly=anomaly,
        windows=windows,
        ua_scale=ua,
        feed_scale=feed,
        foam_scale=foam,
        sensor_bias_C=bias,
        shift_note=note,
        components=components,
    )


__all__ = [
    "AnomalyType",
    "AnomalyWindow",
    "AnomalySchedule",
    "ALLOWED_STEPS",
    "build_anomaly_schedule",
    "generate_shift_note",
    "FOAMING_TORQUE_DROP",
    "COOLING_UA_DROP",
    "COOLING_SPIKE_DURATION_MIN",
    "FEED_PAUSE_DURATION_MIN",
    "SENSOR_DRIFT_BIAS_C",
    "CASCADE_FEED_DELAY_MIN",
    "SHIFT_NOTE_PROBABILITY",
]
