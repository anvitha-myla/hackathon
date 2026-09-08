"""Off-gas oxygen/carbon balances and respiratory quotient.

Formulas are fixed. Inlet air composition is the standard dry-air assumption
used when IndPenSim does not record inlet mole fractions.

OUR and CER here are molar rates (mol/h) from an inert (N2) balance:

    n_in  = P * Fg / (R * T)
    n_out = n_in * (1 - yO2_in - yCO2_in) / (1 - yO2_out - yCO2_out)
    OUR   = n_in * yO2_in  - n_out * yO2_out
    CER   = n_out * yCO2_out - n_in * yCO2_in
    RQ    = CER / OUR   if |OUR| >= epsilon else NaN

``Fg`` is aeration rate in L/h, ``P`` in bar, ``T`` in K.
Off-gas O2/CO2 may be provided as mole percent or mole fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Y_O2_IN_AIR = 0.2095
Y_CO2_IN_AIR = 0.0004
R_L_BAR_MOL_K = 0.08314462618
O2_MOLAR_MASS_G = 32.0
CO2_MOLAR_MASS_G = 44.01
RQ_OUR_EPSILON = 1e-8


def _as_float(values: np.ndarray | pd.Series | float) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mole_fraction(
    values: np.ndarray | pd.Series,
    *,
    treat_as_percent: bool | None = None,
) -> np.ndarray:
    """Convert off-gas analyzer readings to mole fraction.

    If ``treat_as_percent`` is True (typical when the source header contains ``%``),
    values are divided by 100. If False, values are already fractions. If None,
    a heuristic is used: median magnitude > 1.5 → percent.
    """
    y = _as_float(values)
    if treat_as_percent is True:
        return y / 100.0
    if treat_as_percent is False:
        return y
    sample = y[np.isfinite(y)]
    if sample.size and float(np.nanmedian(np.abs(sample))) > 1.5:
        return y / 100.0
    return y


def molar_inlet_flow_mol_h(
    aeration_l_h: np.ndarray | pd.Series,
    temperature_k: np.ndarray | pd.Series | None = None,
    pressure_bar: np.ndarray | pd.Series | None = None,
    r_l_bar_mol_k: float = R_L_BAR_MOL_K,
) -> np.ndarray:
    fg = _as_float(aeration_l_h)
    if temperature_k is None:
        temp = np.full_like(fg, 298.15)
    else:
        temp = _as_float(temperature_k)
        # IndPenSim vessel temperature is Kelvin (~298). If values look like °C, shift.
        finite = temp[np.isfinite(temp)]
        if finite.size and float(np.nanmedian(finite)) < 200.0:
            temp = temp + 273.15
    if pressure_bar is None:
        pres = np.ones_like(fg)
    else:
        pres = _as_float(pressure_bar)
        finite_p = pres[np.isfinite(pres)]
        if finite_p.size and float(np.nanmedian(finite_p)) > 20.0:
            # Looks like kPa rather than bar.
            pres = pres / 100.0
    n_in = np.full_like(fg, np.nan, dtype=float)
    ok = np.isfinite(fg) & np.isfinite(temp) & np.isfinite(pres) & (temp > 0)
    n_in[ok] = pres[ok] * fg[ok] / (r_l_bar_mol_k * temp[ok])
    return n_in


def offgas_our_cer_mol_h(
    aeration_l_h: np.ndarray | pd.Series,
    o2_offgas: np.ndarray | pd.Series,
    co2_offgas: np.ndarray | pd.Series,
    temperature_k: np.ndarray | pd.Series | None = None,
    pressure_bar: np.ndarray | pd.Series | None = None,
    y_o2_in: float = Y_O2_IN_AIR,
    y_co2_in: float = Y_CO2_IN_AIR,
    r_l_bar_mol_k: float = R_L_BAR_MOL_K,
    o2_as_percent: bool | None = None,
    co2_as_percent: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """OUR and CER from off-gas mole fractions and aeration (mol/h)."""
    y_o2_out = mole_fraction(o2_offgas, treat_as_percent=o2_as_percent)
    y_co2_out = mole_fraction(co2_offgas, treat_as_percent=co2_as_percent)
    n_in = molar_inlet_flow_mol_h(
        aeration_l_h,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        r_l_bar_mol_k=r_l_bar_mol_k,
    )
    inert_in = 1.0 - y_o2_in - y_co2_in
    inert_out = 1.0 - y_o2_out - y_co2_out
    n_out = np.full_like(n_in, np.nan, dtype=float)
    ok = np.isfinite(n_in) & np.isfinite(inert_out) & (inert_out > 1e-8)
    n_out[ok] = n_in[ok] * inert_in / inert_out[ok]
    our = n_in * y_o2_in - n_out * y_o2_out
    cer = n_out * y_co2_out - n_in * y_co2_in
    return our, cer


def respiratory_quotient(
    cer: np.ndarray | pd.Series,
    our: np.ndarray | pd.Series,
    epsilon: float = RQ_OUR_EPSILON,
) -> np.ndarray:
    """RQ = CER / OUR with a near-zero OUR guard (returns NaN, never inf)."""
    if epsilon <= 0:
        raise ValueError("RQ OUR epsilon must be positive")
    cer_a = _as_float(cer)
    our_a = _as_float(our)
    rq = np.full_like(our_a, np.nan, dtype=float)
    ok = np.isfinite(cer_a) & np.isfinite(our_a) & (np.abs(our_a) >= epsilon)
    rq[ok] = cer_a[ok] / our_a[ok]
    return rq


def dataset_our_cer_to_mol_h(
    our_raw: np.ndarray | pd.Series,
    cer_raw: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert published IndPenSim OUR (g/min) and CER (g/h) to mol/h.

    Goldrick 100-batch dump stores OUR in g min^-1 and CER in g/h. This
    conversion is only used as a fallback when off-gas O2/CO2 are absent.
    """
    our_g_min = _as_float(our_raw)
    cer_g_h = _as_float(cer_raw)
    our_mol_h = (our_g_min * 60.0) / O2_MOLAR_MASS_G
    cer_mol_h = cer_g_h / CO2_MOLAR_MASS_G
    return our_mol_h, cer_mol_h
