"""Process control logic — TCU thermal and H2 pressure loops."""

from __future__ import annotations

from dataclasses import dataclass, field

from arip_backend.schemas.control import ControlPackage


@dataclass
class ProcessControllers:
    """Thermal + pressure loops for the batch digital twin.

    Controllers command setpoints that the plant ODEs track with first-order lags.
    PID gains from the control package are retained for SCADA metadata; the
    plant-facing commands use constrained setpoint tracking for stiff BDF stability.
    """

    control: ControlPackage
    jacket_temp_c: float = field(init=False)
    pressure_bar: float = field(init=False)

    def __post_init__(self) -> None:
        tl = self.control.thermal_loop
        pl = self.control.pressure_loop
        self.jacket_temp_c = tl.jacket_setpoint_c if tl.jacket_setpoint_c is not None else tl.setpoint_c
        self.pressure_bar = pl.setpoint_bar

    def step_thermal(self, reactor_temp_c: float, dt: float) -> float:
        """Command jacket temperature for isothermal hold / cooling trim."""
        sp = self.control.thermal_loop.setpoint_c
        # Proportional jacket trim: heat if reactor below SP, cool if above
        err = sp - reactor_temp_c
        kp = abs(self.control.thermal_loop.pid.Kp)
        self.jacket_temp_c = sp + kp * err
        self.jacket_temp_c = min(
            self.control.thermal_loop.max_jacket_temp_c,
            max(self.control.thermal_loop.min_jacket_temp_c, self.jacket_temp_c),
        )
        return self.jacket_temp_c

    def step_pressure(self, reactor_pressure_bar: float, dt: float) -> float:
        """Command H2 headspace pressure setpoint (MFC / regulator)."""
        # Tight regulation to configured setpoint within interlock envelope
        sp = self.control.pressure_loop.setpoint_bar
        self.pressure_bar = min(
            self.control.pressure_loop.max_pressure_bar,
            max(self.control.pressure_loop.min_pressure_bar, sp),
        )
        return self.pressure_bar

    @property
    def ua_W_K(self) -> float:
        if self.control.thermal_loop.ua_override_W_K is not None:
            return float(self.control.thermal_loop.ua_override_W_K)
        return 40.0
