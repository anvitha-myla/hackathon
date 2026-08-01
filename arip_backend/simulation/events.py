"""Event state machine for batch digital-twin simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class BatchPhase(str, Enum):
    INIT = "INIT"
    HEAT_UP = "HEAT_UP"
    HYDROGENATION = "HYDROGENATION"
    HOLD = "HOLD"
    COMPLETE = "COMPLETE"
    TRIPPED = "TRIPPED"


@dataclass
class SimulationEvent:
    time_s: float
    event_type: str
    message: str
    phase: BatchPhase
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventStateMachine:
    """Tracks batch phase transitions and interlock trips."""

    max_reactor_temp_c: float
    max_rha_mole_fraction: float
    max_pressure_bar: float
    conversion_complete_threshold: float = 0.98
    phase: BatchPhase = BatchPhase.INIT
    events: list[SimulationEvent] = field(default_factory=list)
    tripped: bool = False

    def start(self, time_s: float = 0.0) -> None:
        self._transition(time_s, BatchPhase.HEAT_UP, "Batch started — heat-up / pressurize")

    def _transition(self, time_s: float, new_phase: BatchPhase, message: str, **data: Any) -> None:
        self.phase = new_phase
        self.events.append(
            SimulationEvent(
                time_s=time_s,
                event_type="phase_transition",
                message=message,
                phase=new_phase,
                data=data,
            )
        )

    def record(self, time_s: float, event_type: str, message: str, **data: Any) -> None:
        self.events.append(
            SimulationEvent(
                time_s=time_s,
                event_type=event_type,
                message=message,
                phase=self.phase,
                data=data,
            )
        )

    def evaluate(
        self,
        time_s: float,
        *,
        temperature_c: float,
        pressure_bar: float,
        rha_mole_fraction: float,
        nitro_conversion: float,
    ) -> BatchPhase:
        if self.tripped:
            return BatchPhase.TRIPPED

        if temperature_c >= self.max_reactor_temp_c:
            self.tripped = True
            self._transition(
                time_s,
                BatchPhase.TRIPPED,
                f"Temperature interlock trip at {temperature_c:.1f} °C",
                temperature_c=temperature_c,
            )
            return self.phase

        if pressure_bar >= self.max_pressure_bar:
            self.tripped = True
            self._transition(
                time_s,
                BatchPhase.TRIPPED,
                f"Pressure interlock trip at {pressure_bar:.2f} bar",
                pressure_bar=pressure_bar,
            )
            return self.phase

        if rha_mole_fraction >= self.max_rha_mole_fraction:
            self.tripped = True
            self._transition(
                time_s,
                BatchPhase.TRIPPED,
                f"RHA accumulation interlock trip at x={rha_mole_fraction:.4f}",
                rha_mole_fraction=rha_mole_fraction,
            )
            return self.phase

        if self.phase == BatchPhase.HEAT_UP and temperature_c >= 60.0:
            self._transition(time_s, BatchPhase.HYDROGENATION, "Entered hydrogenation phase")

        if (
            self.phase in (BatchPhase.HYDROGENATION, BatchPhase.HOLD)
            and nitro_conversion >= self.conversion_complete_threshold
        ):
            self._transition(
                time_s,
                BatchPhase.COMPLETE,
                f"Conversion complete ({nitro_conversion:.3f})",
                nitro_conversion=nitro_conversion,
            )

        return self.phase

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "time_s": e.time_s,
                "event_type": e.event_type,
                "message": e.message,
                "phase": e.phase.value,
                "data": e.data,
            }
            for e in self.events
        ]


def make_scipy_events(
    sm: EventStateMachine,
    state_index: dict[str, int],
    c0_nitro: float,
) -> list[Callable]:
    """Build ``solve_ivp`` event callables that terminate on interlock / completion."""

    def _temp_event(t: float, y):  # noqa: ANN001
        return sm.max_reactor_temp_c - y[state_index["T_c"]]

    _temp_event.terminal = True
    _temp_event.direction = -1

    def _pressure_event(t: float, y):  # noqa: ANN001
        return sm.max_pressure_bar - y[state_index["P_bar"]]

    _pressure_event.terminal = True
    _pressure_event.direction = -1

    def _rha_event(t: float, y):  # noqa: ANN001
        species = ["NX", "NSX", "XHA", "XYL", "H2O", "H2"]
        total = sum(max(y[state_index[s]], 0.0) for s in species)
        x_rha = (max(y[state_index["XHA"]], 0.0) / total) if total > 0 else 0.0
        return sm.max_rha_mole_fraction - x_rha

    _rha_event.terminal = True
    _rha_event.direction = -1

    def _conversion_event(t: float, y):  # noqa: ANN001
        nx = max(y[state_index["NX"]], 0.0)
        conv = 1.0 - nx / c0_nitro if c0_nitro > 0 else 0.0
        return conv - sm.conversion_complete_threshold

    _conversion_event.terminal = True
    _conversion_event.direction = 1

    return [_temp_event, _pressure_event, _rha_event, _conversion_event]
