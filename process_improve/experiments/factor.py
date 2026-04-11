# (c) Kevin Dunn, 2010-2026. MIT License.

"""Factor, Constraint, and DesignResult definitions for design generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from process_improve.experiments.structures import Expt


class FactorType(str, Enum):
    """Type of experimental factor."""

    continuous = "continuous"
    categorical = "categorical"
    mixture = "mixture"


class Factor(BaseModel):
    """Specification for a single experimental factor.

    Parameters
    ----------
    name : str
        Name of the factor (e.g. "Temperature", "Pressure").
    type : FactorType
        One of "continuous", "categorical", or "mixture".
    low : float or None
        Low level for continuous/mixture factors.
    high : float or None
        High level for continuous/mixture factors.
    levels : list or None
        Explicit levels for categorical factors, or for continuous factors
        with more than 2 levels.
    units : str
        Engineering units (e.g. "degC", "bar", "g/L").

    Examples
    --------
    >>> Factor(name="Temperature", low=150, high=200, units="degC")
    >>> Factor(name="Catalyst", type="categorical", levels=["A", "B", "C"])
    >>> Factor(name="x1", type="mixture", low=0.0, high=1.0)
    """

    name: str
    type: FactorType = FactorType.continuous
    low: float | None = None
    high: float | None = None
    levels: list[Any] | None = None
    units: str = ""

    @model_validator(mode="after")
    def _validate_factor(self) -> Factor:
        if self.type == FactorType.continuous:
            if self.low is None or self.high is None:
                raise ValueError(f"Factor '{self.name}': continuous factors require 'low' and 'high'.")
            if self.low >= self.high:
                raise ValueError(f"Factor '{self.name}': 'low' ({self.low}) must be less than 'high' ({self.high}).")
        elif self.type == FactorType.categorical:
            if not self.levels or len(self.levels) < 2:
                raise ValueError(f"Factor '{self.name}': categorical factors require 'levels' with at least 2 entries.")
        elif self.type == FactorType.mixture:
            if self.low is None:
                self.low = 0.0
            if self.high is None:
                self.high = 1.0
        return self

    @property
    def center(self) -> float | None:
        """Return the center point value, or None for categorical factors."""
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / 2.0
        return None

    @property
    def range(self) -> tuple[float, float] | None:
        """Return the (low, high) range tuple, or None for categorical factors."""
        if self.low is not None and self.high is not None:
            return (self.low, self.high)
        return None


class Constraint(BaseModel):
    """A constraint on the experimental factor space.

    Parameters
    ----------
    expression : str
        A string expression, e.g. ``"A + B <= 1.0"`` or ``"3*T + 5*D <= 600"``.
        Factor names in the expression must match the ``name`` fields of the
        corresponding ``Factor`` objects.
    type : str
        Either ``"linear"`` or ``"nonlinear"``.

    Examples
    --------
    >>> Constraint(expression="A + B <= 1.0")
    >>> Constraint(expression="3*T + 5*D <= 600", type="linear")
    """

    expression: str
    type: Literal["linear", "nonlinear"] = "linear"


@dataclass
class DesignResult:
    """Container for a generated experimental design and its metadata.

    Parameters
    ----------
    design : Expt
        Design matrix in coded units (-1 / +1).
    design_actual : Expt
        Design matrix in actual (real-world) units.
    run_order : list[int]
        Randomized run order (1-based).
    design_type : str
        Name of the design that was generated.
    n_runs : int
        Total number of experimental runs.
    n_factors : int
        Number of factors in the design.
    factor_names : list[str]
        Ordered list of factor names.
    generators : list[str] or None
        Generator strings for fractional factorials (e.g. ``["E=ABC"]``).
    defining_relation : list[str] or None
        Defining relation words (e.g. ``["I=ABCE"]``).
    resolution : int or None
        Resolution of the design (III, IV, V, ...).
    alpha : float or None
        Axial distance used for CCD designs.
    blocks : list[int] or None
        Block assignment for each run.
    metadata : dict[str, Any]
        Additional design-specific information.
    """

    design: Expt
    design_actual: Expt
    run_order: list[int]
    design_type: str
    n_runs: int
    n_factors: int
    factor_names: list[str]
    generators: list[str] | None = None
    defining_relation: list[str] | None = None
    resolution: int | None = None
    alpha: float | None = None
    blocks: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        parts = [
            f"DesignResult(type={self.design_type!r}",
            f"runs={self.n_runs}",
            f"factors={self.n_factors}",
        ]
        if self.resolution is not None:
            parts.append(f"resolution={self.resolution}")
        if self.generators:
            parts.append(f"generators={self.generators}")
        return ", ".join(parts) + ")"
