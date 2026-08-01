"""Reaction package schemas for the ARIP Digital Twin.

Supports master reaction packages, elementary reaction steps
(stoichiometry, Arrhenius kinetics E_a / k_0, ΔH_rxn, operating windows),
and species thermodynamic property tables.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RateLawType(str, Enum):
    """Supported kinetic rate-law forms."""

    POWER_LAW = "power_law"
    ARRHENIUS_POWER_LAW = "arrhenius_power_law"
    LANGMUIR_HINSHELWOOD = "langmuir_hinshelwood"
    ELEY_RIDEAL = "eley_rideal"
    CUSTOM = "custom"


class StoichiometricSpecies(BaseModel):
    """A single species entry in a stoichiometric equation."""

    model_config = ConfigDict(extra="forbid")

    species_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    coefficient: float = Field(
        ...,
        description="Negative = reactant, positive = product, zero = tracked inert/solvent.",
    )
    role: Optional[str] = Field(
        default=None,
        description="reactant | product | intermediate | catalyst | solvent | inert",
    )


class KineticParameters(BaseModel):
    """Arrhenius / power-law kinetic parameters: k = k_0 · exp(-E_a /(R·T))."""

    model_config = ConfigDict(extra="forbid")

    rate_law: RateLawType = RateLawType.ARRHENIUS_POWER_LAW
    k0: float = Field(..., gt=0, description="Pre-exponential factor k₀.")
    Ea_J_mol: float = Field(..., ge=0, description="Activation energy E_a [J/mol].")
    delta_H_rxn_J_mol: float = Field(
        ...,
        description="Heat of reaction ΔH_rxn [J/mol extent]; negative = exothermic.",
    )
    reaction_orders: dict[str, float] = Field(default_factory=dict)
    reference_temperature_K: Optional[float] = Field(default=None, gt=0)
    gas_constant_J_molK: float = Field(default=8.314462618, gt=0)
    notes: Optional[str] = None


class OperatingWindow(BaseModel):
    """Recommended / allowable operating envelope."""

    model_config = ConfigDict(extra="forbid")

    t_min_c: float
    t_max_c: float
    p_min_barg: float = Field(default=0.0, ge=0)
    p_max_barg: float = Field(..., ge=0)
    concentration_min_mol_L: Optional[dict[str, float]] = None
    concentration_max_mol_L: Optional[dict[str, float]] = None
    max_addition_rate_mol_s: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_windows(self) -> OperatingWindow:
        if self.t_max_c <= self.t_min_c:
            raise ValueError("t_max_c must be greater than t_min_c")
        if self.p_max_barg < self.p_min_barg:
            raise ValueError("p_max_barg must be >= p_min_barg")
        return self


class ReactionPackage(BaseModel):
    """Standalone or elementary reaction package (step-level)."""

    model_config = ConfigDict(extra="forbid")

    reaction_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    version: str = "1.0.0"
    description: Optional[str] = None
    step_index: Optional[int] = Field(default=None, ge=1)
    parent_reaction_id: Optional[str] = None
    stoichiometry: list[StoichiometricSpecies] = Field(..., min_length=1)
    kinetics: KineticParameters
    operating_window: OperatingWindow
    solvent: Optional[str] = None
    catalyst: Optional[str] = None
    references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_stoichiometry(self) -> ReactionPackage:
        coeffs = [s.coefficient for s in self.stoichiometry]
        if not any(c < 0 for c in coeffs):
            raise ValueError("stoichiometry must include at least one reactant")
        if not any(c > 0 for c in coeffs):
            raise ValueError("stoichiometry must include at least one product")
        ids = [s.species_id for s in self.stoichiometry]
        if len(ids) != len(set(ids)):
            raise ValueError("stoichiometry species_id values must be unique")
        return self


class ReactionStepRef(BaseModel):
    """Pointer from a master package to an elementary step file."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=1)
    reaction_id: str
    relative_path: str
    name: str


class CatalystSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = Field(..., description="e.g. supported noble metal, Raney Ni")
    active_metal: Optional[str] = None
    support: Optional[str] = None
    metal_loading_wt_pct: Optional[float] = Field(default=None, ge=0)
    typical_loading_g_L: Optional[float] = Field(default=None, gt=0)


class ReactionMasterPackage(BaseModel):
    """Master multi-step reaction package (e.g. nitroxylene hydrogenation)."""

    model_config = ConfigDict(extra="forbid")

    package_type: str = Field(default="reaction_master")
    reaction_id: str
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    chemistry_family: Optional[str] = None
    overall_stoichiometry: list[StoichiometricSpecies] = Field(..., min_length=1)
    overall_kinetics: Optional[KineticParameters] = None
    operating_window: OperatingWindow
    catalyst: Optional[CatalystSpec] = None
    solvent: Optional[str] = None
    steps: list[ReactionStepRef] = Field(..., min_length=1)
    thermodynamic_properties_path: Optional[str] = None
    hazard_notes: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_overall(self) -> ReactionMasterPackage:
        coeffs = [s.coefficient for s in self.overall_stoichiometry]
        if not any(c < 0 for c in coeffs) or not any(c > 0 for c in coeffs):
            raise ValueError("overall_stoichiometry needs reactants and products")
        indices = [s.step_index for s in self.steps]
        if len(indices) != len(set(indices)):
            raise ValueError("step_index values must be unique")
        return self


class SpeciesThermoProperties(BaseModel):
    """Thermodynamic properties for one species."""

    model_config = ConfigDict(extra="forbid")

    species_id: str
    name: str
    formula: Optional[str] = None
    molecular_weight_g_mol: float = Field(..., gt=0)
    cas_number: Optional[str] = None
    phase_reference: str = Field(default="liquid", description="Reference phase for ΔHf / Cp.")
    delta_Hf_298_kJ_mol: float = Field(..., description="Standard enthalpy of formation [kJ/mol].")
    delta_Gf_298_kJ_mol: Optional[float] = None
    Cp_J_molK: float = Field(..., gt=0, description="Heat capacity at reference conditions [J/(mol·K)].")
    Cp_poly_A: Optional[float] = Field(default=None, description="Optional Cp(T)=A+BT+CT^2 coefficient A.")
    Cp_poly_B: Optional[float] = None
    Cp_poly_C: Optional[float] = None
    boiling_point_c: Optional[float] = None
    melting_point_c: Optional[float] = None
    density_kg_m3: Optional[float] = Field(default=None, gt=0)
    notes: Optional[str] = None


class ThermodynamicPropertiesPackage(BaseModel):
    """Species thermodynamic property table for a reaction family."""

    model_config = ConfigDict(extra="forbid")

    package_type: str = Field(default="thermodynamic_properties")
    package_id: str
    name: str
    version: str = "1.0.0"
    parent_reaction_id: Optional[str] = None
    reference_temperature_K: float = Field(default=298.15, gt=0)
    reference_pressure_bar: float = Field(default=1.0, gt=0)
    species: list[SpeciesThermoProperties] = Field(..., min_length=1)
    references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_species(self) -> ThermodynamicPropertiesPackage:
        ids = [s.species_id for s in self.species]
        if len(ids) != len(set(ids)):
            raise ValueError("species_id values must be unique")
        return self


def parse_reaction_document(data: dict[str, Any], *, source_name: str = "") -> BaseModel:
    """Dispatch a reaction JSON document to the appropriate schema."""
    package_type = data.get("package_type")
    if package_type == "reaction_master" or "steps" in data and "overall_stoichiometry" in data:
        return ReactionMasterPackage.model_validate(data)
    if package_type == "thermodynamic_properties" or "species" in data and "package_id" in data:
        return ThermodynamicPropertiesPackage.model_validate(data)
    if "stoichiometry" in data and "kinetics" in data:
        return ReactionPackage.model_validate(data)
    raise ValueError(
        f"Unrecognized reaction document in {source_name or '<unknown>'}: "
        f"keys={sorted(data.keys())}"
    )
