"""Reaction package schemas for the ARIP Digital Twin.

Validates the nitroxylene hydrogenation master package, micro-kinetic step
files (Langmuir–Hinshelwood / power-law rate laws for SciPy ``solve_ivp`` BDF),
and thermodynamic / transport property tables (Henry's law, Cp, kLa).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

R_J_MOL_K = 8.314462618
R_KJ_MOL_K = R_J_MOL_K / 1000.0


class Stoichiometry(BaseModel):
    """Reactant / product stoichiometric coefficients."""

    model_config = ConfigDict(extra="forbid")

    reactants: dict[str, PositiveFloat]
    products: dict[str, PositiveFloat]


class ReactionKinetics(BaseModel):
    """Arrhenius kinetics for an elementary hydrogenation step."""

    model_config = ConfigDict(extra="forbid")

    rate_law_type: str
    pre_exponential_k0: PositiveFloat = Field(..., description="Arrhenius pre-exponential factor k0")
    activation_energy_ea_kj_mol: PositiveFloat = Field(
        ...,
        description="Activation energy Ea in kJ/mol",
    )
    reaction_order: dict[str, float] = Field(
        default_factory=dict,
        description="Partial orders keyed by species / catalyst terms (e.g. H2_dissolved).",
    )


class ReactionStepThermodynamics(BaseModel):
    """Step-level heat of reaction."""

    model_config = ConfigDict(extra="forbid")

    reaction_enthalpy_kj_mol: float = Field(
        ...,
        description="Heat of reaction ΔH_rxn [kJ/mol] (negative for exothermic)",
    )
    exothermic: bool = True

    @model_validator(mode="after")
    def _check_exothermic_flag(self) -> ReactionStepThermodynamics:
        if self.exothermic and self.reaction_enthalpy_kj_mol > 0:
            raise ValueError("exothermic=True requires reaction_enthalpy_kj_mol <= 0")
        if not self.exothermic and self.reaction_enthalpy_kj_mol < 0:
            raise ValueError("exothermic=False requires reaction_enthalpy_kj_mol >= 0")
        return self


class ReactionStepPackage(BaseModel):
    """Micro-kinetic step package (r_i for stiff ODE integration)."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    description: str
    stoichiometry: Stoichiometry
    kinetics: ReactionKinetics
    thermodynamics: ReactionStepThermodynamics


class SpeciesProperty(BaseModel):
    """Physical / transport properties for one chemical species."""

    model_config = ConfigDict(extra="forbid")

    molecular_weight_g_mol: PositiveFloat
    liquid_density_kg_m3: Optional[PositiveFloat] = None
    specific_heat_capacity_j_kgk: Optional[PositiveFloat] = None
    henry_law_constant_bar_m3_kmol: Optional[PositiveFloat] = None
    diffusion_coefficient_m2_s: Optional[PositiveFloat] = None


class MixtureTransportProperties(BaseModel):
    """Bulk mixture properties for energy balance and mass transfer."""

    model_config = ConfigDict(extra="forbid")

    bulk_liquid_density_kg_m3: PositiveFloat = 1050.0
    bulk_heat_capacity_j_kgk: PositiveFloat = 2100.0
    liquid_viscosity_pas: PositiveFloat = 0.0012
    kla_base_coefficient_1_s: PositiveFloat = Field(
        0.085,
        description="Base gas–liquid mass-transfer coefficient kLa [1/s]",
    )


class ThermodynamicPropertiesPackage(BaseModel):
    """Species + mixture properties for dT/dt and Henry's law C_H2*."""

    model_config = ConfigDict(extra="forbid")

    chemical_species: dict[str, SpeciesProperty]
    mixture_transport_properties: MixtureTransportProperties


class OperatingWindow(BaseModel):
    """Process operating envelope for the master reaction."""

    model_config = ConfigDict(extra="forbid")

    target_temperature_c: float
    min_temperature_c: float
    max_temperature_c: float
    target_pressure_bar: PositiveFloat
    min_pressure_bar: PositiveFloat
    max_safe_pressure_bar: PositiveFloat

    @model_validator(mode="after")
    def _check_window(self) -> OperatingWindow:
        if not (self.min_temperature_c <= self.target_temperature_c <= self.max_temperature_c):
            raise ValueError("target_temperature_c must lie within [min, max]")
        if not (self.min_pressure_bar <= self.target_pressure_bar <= self.max_safe_pressure_bar):
            raise ValueError("target_pressure_bar must lie within [min, max_safe]")
        return self


class SafetyLimits(BaseModel):
    """Safety-critical limits for hydroxylamine accumulation and runaway metrics."""

    model_config = ConfigDict(extra="forbid")

    max_rha_accumulation_mole_fraction: float = Field(
        ...,
        ge=0,
        le=1,
        description="Max allowable hydroxylamine (RHA) accumulation mole fraction",
    )
    max_adiabatic_temperature_rise_k: PositiveFloat
    max_self_heating_rate_k_min: PositiveFloat
    onset_decomposition_temp_c: float


class ChemicalSystem(BaseModel):
    """Chemical system metadata for the master package."""

    model_config = ConfigDict(extra="forbid")

    target_product: str
    cas_number: str
    solvent: str
    catalyst: str
    catalyst_loading_kg_m3: PositiveFloat


class MasterReactionPackage(BaseModel):
    """Master multi-step nitroxylene hydrogenation package."""

    model_config = ConfigDict(extra="forbid")

    reaction_id: str
    name: str
    chemical_system: ChemicalSystem
    overall_stoichiometry: Stoichiometry
    operating_window: OperatingWindow
    safety_critical_limits: SafetyLimits
    mechanism_steps: list[str] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Parsing / in-memory kinetic assembly for SciPy BDF ODE models
# ---------------------------------------------------------------------------


def parse_reaction_document(data: dict[str, Any], *, source_name: str = "") -> BaseModel:
    """Dispatch a reaction JSON document to the appropriate schema."""
    if "mechanism_steps" in data and "reaction_id" in data:
        return MasterReactionPackage.model_validate(data)
    if "chemical_species" in data and "mixture_transport_properties" in data:
        return ThermodynamicPropertiesPackage.model_validate(data)
    if "step_id" in data and "kinetics" in data:
        return ReactionStepPackage.model_validate(data)
    raise ValueError(
        f"Unrecognized reaction document in {source_name or '<unknown>'}: "
        f"keys={sorted(data.keys())}"
    )


def arrhenius_k(k0: float, ea_kj_mol: float, temperature_k: float) -> float:
    """Evaluate k(T) = k0 · exp(-Ea /(R·T)) with Ea in kJ/mol."""
    return k0 * math.exp(-ea_kj_mol / (R_KJ_MOL_K * temperature_k))


def henrys_law_ch2_star(p_h2_bar: float, henry_bar_m3_kmol: float) -> float:
    """Equilibrium dissolved H2 concentration C_H2* [kmol/m³] via Henry's law.

    Uses ``C* = P / H`` with H in bar·m³/kmol.
    """
    if henry_bar_m3_kmol <= 0:
        raise ValueError("Henry's law constant must be positive")
    return p_h2_bar / henry_bar_m3_kmol


def build_rate_law_callable(step: ReactionStepPackage) -> Callable[..., float]:
    """Build ``r_i(concentrations, T_K, c_cat)`` power-law rate callable.

    ``concentrations`` is a mapping of species name → concentration.
    Uses dissolved hydrogen key ``H2_dissolved`` when present in reaction orders.
    """
    k0 = float(step.kinetics.pre_exponential_k0)
    ea = float(step.kinetics.activation_energy_ea_kj_mol)
    orders = dict(step.kinetics.reaction_order)

    def rate(concentrations: dict[str, float], temperature_k: float, c_cat: float = 1.0) -> float:
        k = arrhenius_k(k0, ea, temperature_k)
        r = k
        for species, order in orders.items():
            if species == "catalyst_concentration":
                r *= c_cat**order
            else:
                r *= concentrations.get(species, 0.0) ** order
        return r

    return rate


def build_kinetic_memory(
    master: MasterReactionPackage,
    steps: list[ReactionStepPackage],
    thermo: ThermodynamicPropertiesPackage,
    *,
    temperature_c: float | None = None,
    pressure_bar: float | None = None,
    concentrations: dict[str, float] | None = None,
    catalyst_concentration: float | None = None,
) -> dict[str, Any]:
    """Assemble in-memory dict with r1–r3, Q_rxn, and Henry's-law C_H2*.

    Rate equations are symbolic descriptions plus evaluated values at the
    master operating-window target (or overrides). Heat generation:

        Q_rxn = Σ r_i · (-ΔH_i)   [consistent concentration / time basis]

    Dissolved hydrogen equilibrium:

        C_H2* = P_H2 / H
    """
    if len(steps) != 3:
        raise ValueError(f"Expected 3 mechanism steps, got {len(steps)}")

    steps_sorted = sorted(steps, key=lambda s: s.step_id)
    t_c = temperature_c if temperature_c is not None else master.operating_window.target_temperature_c
    p_bar = pressure_bar if pressure_bar is not None else master.operating_window.target_pressure_bar
    t_k = t_c + 273.15
    c_cat = (
        catalyst_concentration
        if catalyst_concentration is not None
        else master.chemical_system.catalyst_loading_kg_m3
    )

    h2_props = thermo.chemical_species.get("H2")
    if h2_props is None or h2_props.henry_law_constant_bar_m3_kmol is None:
        raise ValueError("thermodynamic_properties must define H2.henry_law_constant_bar_m3_kmol")
    henry = float(h2_props.henry_law_constant_bar_m3_kmol)
    c_h2_star = henrys_law_ch2_star(p_bar, henry)

    # Default evaluation concentrations (kmol/m³-scale placeholders for digital twin)
    conc = {
        "2,4-nitroxylene": 1.0,
        "2,4-nitrosoxylene": 0.01,
        "2,4-xylidylhydroxylamine": 0.01,
        "2,4-xylidine": 0.0,
        "H2_dissolved": c_h2_star,
        "H2O": 0.0,
    }
    if concentrations:
        conc.update(concentrations)

    rate_equations: dict[str, Any] = {}
    q_rxn = 0.0
    for idx, step in enumerate(steps_sorted, start=1):
        rate_fn = build_rate_law_callable(step)
        r_i = rate_fn(conc, t_k, c_cat)
        delta_h = float(step.thermodynamics.reaction_enthalpy_kj_mol)
        q_i = r_i * (-delta_h)
        q_rxn += q_i
        order_terms = " * ".join(
            f"C_{sp}^{ord_}" if sp != "catalyst_concentration" else f"C_cat^{ord_}"
            for sp, ord_ in step.kinetics.reaction_order.items()
        )
        rate_equations[f"r{idx}"] = {
            "step_id": step.step_id,
            "description": step.description,
            "rate_law_type": step.kinetics.rate_law_type,
            "equation": (
                f"r_{idx} = k0_{idx} * exp(-Ea_{idx}/(R*T)) * {order_terms}"
                if order_terms
                else f"r_{idx} = k0_{idx} * exp(-Ea_{idx}/(R*T))"
            ),
            "k0": float(step.kinetics.pre_exponential_k0),
            "Ea_kj_mol": float(step.kinetics.activation_energy_ea_kj_mol),
            "delta_H_kj_mol": delta_h,
            "k_T": arrhenius_k(
                float(step.kinetics.pre_exponential_k0),
                float(step.kinetics.activation_energy_ea_kj_mol),
                t_k,
            ),
            "r_value": r_i,
            "q_i": q_i,
            "stoichiometry": step.stoichiometry.model_dump(),
        }

    return {
        "reaction_id": master.reaction_id,
        "name": master.name,
        "temperature_c": t_c,
        "temperature_k": t_k,
        "pressure_bar": p_bar,
        "catalyst_concentration": c_cat,
        "rate_equations": rate_equations,
        "Q_rxn": q_rxn,
        "Q_rxn_definition": "sum_i r_i * (-delta_H_i)",
        "C_H2_star": c_h2_star,
        "C_H2_star_definition": "P_H2 / H  (kmol/m3)",
        "henry_law_constant_bar_m3_kmol": henry,
        "concentrations_used": conc,
        "mixture_transport_properties": thermo.mixture_transport_properties.model_dump(),
        "safety_critical_limits": master.safety_critical_limits.model_dump(),
        "operating_window": master.operating_window.model_dump(),
    }


# Backward-compatible aliases used elsewhere in the package
ReactionPackage = ReactionStepPackage
ReactionMasterPackage = MasterReactionPackage
KineticParameters = ReactionKinetics
StoichiometricSpecies = Stoichiometry
