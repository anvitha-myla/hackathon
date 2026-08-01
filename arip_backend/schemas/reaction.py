"""Reaction package schemas for the ARIP Digital Twin.

Captures stoichiometry, Arrhenius kinetics (E_a, k_0), reaction enthalpy
(ΔH_rxn), and safe operating windows used by digital-twin simulation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RateLawType(str, Enum):
    """Supported kinetic rate-law forms."""

    POWER_LAW = "power_law"
    ARRHENIUS_POWER_LAW = "arrhenius_power_law"
    MICHAELIS_MENTEN = "michaelis_menten"
    LANGMUIR_HINSHELWOOD = "langmuir_hinshelwood"
    CUSTOM = "custom"


class StoichiometricSpecies(BaseModel):
    """A single species entry in a stoichiometric equation."""

    model_config = ConfigDict(extra="forbid")

    species_id: str = Field(..., min_length=1, description="Canonical species identifier.")
    name: Optional[str] = Field(default=None, description="Display name / IUPAC or common name.")
    coefficient: float = Field(
        ...,
        description=(
            "Stoichiometric coefficient. Negative = reactant, positive = product, "
            "zero = inert / solvent (tracked but not consumed)."
        ),
    )
    role: Optional[str] = Field(
        default=None,
        description="Optional role tag: reactant | product | catalyst | solvent | inert.",
    )


class KineticParameters(BaseModel):
    """Arrhenius / power-law kinetic parameters.

    Rate constant form:  k = k_0 · exp(-E_a / (R·T))
    where k_0 has units consistent with the rate law and concentration basis.
    """

    model_config = ConfigDict(extra="forbid")

    rate_law: RateLawType = Field(
        default=RateLawType.ARRHENIUS_POWER_LAW,
        description="Kinetic rate-law family.",
    )
    k0: float = Field(
        ...,
        gt=0,
        description="Pre-exponential factor k₀ (units depend on rate law / order).",
    )
    Ea_J_mol: float = Field(
        ...,
        ge=0,
        description="Activation energy E_a [J/mol].",
    )
    delta_H_rxn_J_mol: float = Field(
        ...,
        description=(
            "Standard heat of reaction ΔH_rxn [J/mol of extent]. "
            "Negative = exothermic, positive = endothermic."
        ),
    )
    reaction_orders: dict[str, float] = Field(
        default_factory=dict,
        description="Partial reaction orders keyed by species_id (power-law).",
    )
    reference_temperature_K: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional reference temperature for k₀ reporting [K].",
    )
    gas_constant_J_molK: float = Field(
        default=8.314462618,
        gt=0,
        description="Gas constant R used with E_a [J/(mol·K)].",
    )
    notes: Optional[str] = Field(default=None, description="Kinetic model notes / literature source.")


class OperatingWindow(BaseModel):
    """Recommended / allowable operating envelope for the reaction."""

    model_config = ConfigDict(extra="forbid")

    t_min_c: float = Field(..., description="Minimum recommended reaction temperature [°C].")
    t_max_c: float = Field(..., description="Maximum recommended reaction temperature [°C].")
    p_min_barg: float = Field(default=0.0, ge=0, description="Minimum operating pressure [barg].")
    p_max_barg: float = Field(..., ge=0, description="Maximum operating pressure [barg].")
    concentration_min_mol_L: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional lower concentration bounds [mol/L] keyed by species_id.",
    )
    concentration_max_mol_L: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional upper concentration bounds [mol/L] keyed by species_id.",
    )
    max_addition_rate_mol_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional maximum reagent addition rate [mol/s] for semi-batch safety.",
    )

    @model_validator(mode="after")
    def _check_windows(self) -> OperatingWindow:
        if self.t_max_c <= self.t_min_c:
            raise ValueError("t_max_c must be greater than t_min_c")
        if self.p_max_barg < self.p_min_barg:
            raise ValueError("p_max_barg must be >= p_min_barg")
        return self


class ReactionPackage(BaseModel):
    """Complete reaction package for digital-twin simulation."""

    model_config = ConfigDict(extra="forbid")

    reaction_id: str = Field(..., min_length=1, description="Unique reaction package ID.")
    name: str = Field(..., min_length=1, description="Human-readable reaction name.")
    version: str = Field(default="1.0.0", description="Semantic version of this package.")
    description: Optional[str] = Field(default=None, description="Reaction chemistry summary.")
    stoichiometry: list[StoichiometricSpecies] = Field(
        ...,
        min_length=1,
        description="Stoichiometric species list (reactants negative, products positive).",
    )
    kinetics: KineticParameters = Field(..., description="Kinetic and thermochemical parameters.")
    operating_window: OperatingWindow = Field(
        ...,
        description="Safe / recommended temperature, pressure, and concentration envelope.",
    )
    solvent: Optional[str] = Field(default=None, description="Primary solvent species_id or name.")
    references: list[str] = Field(
        default_factory=list,
        description="Literature or internal document references.",
    )

    @model_validator(mode="after")
    def _check_stoichiometry(self) -> ReactionPackage:
        coeffs = [s.coefficient for s in self.stoichiometry]
        if not any(c < 0 for c in coeffs):
            raise ValueError("stoichiometry must include at least one reactant (negative coefficient)")
        if not any(c > 0 for c in coeffs):
            raise ValueError("stoichiometry must include at least one product (positive coefficient)")
        ids = [s.species_id for s in self.stoichiometry]
        if len(ids) != len(set(ids)):
            raise ValueError("stoichiometry species_id values must be unique")
        return self
