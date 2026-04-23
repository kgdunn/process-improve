"""(c) Kevin Dunn, 2010-2026. MIT License.

Deterministic fake-data simulator for DOE demonstrations.

The simulator is fully specified by a JSON-serialisable ``private_state``
dict containing:

- ``seed``:             int  — base RNG seed for coefficient generation.
- ``factors``:          list of ``{name, low, high, units?}``.
- ``outputs``:          list of ``{name, units?, direction?}``.
- ``structural_hints``: list of free-text hints ("negative interaction
  between pH and surfactant", etc).
- ``noise_level``:      ``"low"`` | ``"medium"`` | ``"high"``.
- ``time_drift``:       bool.
- ``model_version``:    int (schema version for future migrations).

Given this state, :func:`materialize_model` regenerates the exact same
response-surface coefficients on every call, across processes, across
machines.  :func:`simulate` evaluates that surface at a given factor
setting and adds *fresh* Gaussian noise each call so identical inputs
yield similar-but-not-identical outputs — matching the behaviour of a
real physical asset.
"""

# The public tool-call contract uses ``dict[str, Any]`` and ``list[dict[str, Any]]``
# deliberately: the schema that the LLM sees is declared as JSON Schema in
# ``simulation/tools.py``, not as Python types, so tightening the Python side
# with TypedDicts would be duplicative and brittle.
from __future__ import annotations

import re
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION: int = 1

# Fraction of the total coefficient magnitude used as the Gaussian noise
# standard deviation, keyed by the ``noise_level`` string.
_NOISE_FRACTIONS: dict[str, float] = {
    "low": 0.01,
    "medium": 0.05,
    "high": 0.15,
}

# Hint-parsing keyword sets.  Any token intersection with one of these sets
# carries the corresponding meaning; ties (both positive *and* negative in
# the same hint) fall back to "no direction" and are ignored.
_POS_WORDS: frozenset[str] = frozenset(
    {
        "positive", "increase", "increases", "increasing", "synergy",
        "synergistic", "synergise", "synergize", "boost", "boosts", "promotes",
    }
)
_NEG_WORDS: frozenset[str] = frozenset(
    {
        "negative", "decrease", "decreases", "decreasing", "antagonistic",
        "antagonise", "antagonize", "antagonism", "adverse", "inhibits",
        "suppresses",
    }
)
_QUAD_WORDS: frozenset[str] = frozenset(
    {
        "quadratic", "curvature", "curved", "nonlinear", "non", "parabolic",
        "optimum", "maximum", "minimum",
    }
)

_VALID_NOISE_LEVELS: tuple[str, ...] = tuple(_NOISE_FRACTIONS)

# ---------------------------------------------------------------------------
# Input validation (shared by the tool layer)
# ---------------------------------------------------------------------------


def validate_factors(factors: list[dict[str, Any]]) -> None:
    """Validate the ``factors`` list from a ``create_simulator`` call."""
    if not isinstance(factors, list) or not factors:
        raise ValueError("'factors' must be a non-empty list of dicts.")
    seen: set[str] = set()
    for f in factors:
        if not isinstance(f, dict):
            raise TypeError("Each factor must be a dict.")
        name = f.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each factor must have a non-empty 'name' string.")
        if name in seen:
            raise ValueError(f"Duplicate factor name: {name!r}.")
        seen.add(name)
        low = f.get("low")
        high = f.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError(f"Factor {name!r}: 'low' and 'high' must be numbers.")
        if float(low) >= float(high):
            raise ValueError(f"Factor {name!r}: low ({low}) must be < high ({high}).")


def validate_outputs(outputs: list[dict[str, Any]]) -> None:
    """Validate the ``outputs`` list from a ``create_simulator`` call."""
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("'outputs' must be a non-empty list of dicts.")
    seen: set[str] = set()
    for o in outputs:
        if not isinstance(o, dict):
            raise TypeError("Each output must be a dict.")
        name = o.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each output must have a non-empty 'name' string.")
        if name in seen:
            raise ValueError(f"Duplicate output name: {name!r}.")
        seen.add(name)


def validate_noise_level(noise_level: str) -> None:
    """Reject any ``noise_level`` outside the fixed enum."""
    if noise_level not in _VALID_NOISE_LEVELS:
        raise ValueError(
            f"'noise_level' must be one of {list(_VALID_NOISE_LEVELS)}, got {noise_level!r}."
        )


# ---------------------------------------------------------------------------
# Hint parsing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenise(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _parse_hint(
    hint: str,
    factor_names: list[str],
    output_names: list[str],
) -> dict[str, Any]:
    """Turn a free-text hint into a structured directive.

    The returned dict has keys:
    ``factors`` (subset of *factor_names*),
    ``outputs`` (subset of *output_names*; empty means "all"),
    ``direction`` (``"+"`` / ``"-"`` / ``None``),
    ``is_quadratic`` (bool).
    """
    tokens = _tokenise(hint)
    matched_factors = [f for f in factor_names if f.lower() in tokens]
    matched_outputs = [o for o in output_names if o.lower() in tokens]
    has_pos = bool(tokens & _POS_WORDS)
    has_neg = bool(tokens & _NEG_WORDS)
    # Ambiguous hints (both directions) are treated as direction-less.
    direction: str | None = None
    if has_pos and not has_neg:
        direction = "+"
    elif has_neg and not has_pos:
        direction = "-"
    # Bare "non" also matches ("non-linear" etc).
    is_quadratic = bool(tokens & _QUAD_WORDS)
    return {
        "factors": matched_factors,
        "outputs": matched_outputs,
        "direction": direction,
        "is_quadratic": is_quadratic,
    }


def _apply_interaction_hint(
    out_coefs: dict[str, Any],
    factors: list[str],
    sign: float,
    rng: np.random.Generator,
) -> None:
    """Set (or strengthen) the interaction coefficient for a factor pair."""
    a, b = sorted(factors)
    magnitude = float(rng.uniform(1.8, 3.5))
    for inter in out_coefs["interactions"]:
        if tuple(sorted(inter["factors"])) == (a, b):
            inter["coefficient"] = sign * max(abs(inter["coefficient"]), magnitude)
            return
    out_coefs["interactions"].append(
        {"factors": [a, b], "coefficient": sign * magnitude}
    )


def _apply_main_hint(out_coefs: dict[str, Any], factor: str, sign: float) -> None:
    """Set the main-effect coefficient for *factor* to the given sign."""
    current = out_coefs["main"].get(factor, 0.0)
    magnitude = max(abs(current), 3.0)
    out_coefs["main"][factor] = sign * magnitude


def _apply_quadratic_hint(
    out_coefs: dict[str, Any],
    factor: str,
    direction: str | None,
    rng: np.random.Generator,
) -> None:
    """Set the quadratic coefficient for *factor* (concave by default)."""
    magnitude = float(rng.uniform(1.5, 3.0))
    if direction == "+":
        out_coefs["quadratic"][factor] = magnitude
    elif direction == "-":
        out_coefs["quadratic"][factor] = -magnitude
    else:
        # Default to concave (typical optimum-in-the-middle shape).
        out_coefs["quadratic"][factor] = -magnitude


def _apply_hint(
    coefficients: dict[str, dict[str, Any]],
    parsed: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Mutate *coefficients* to reflect a single parsed hint."""
    factors = parsed["factors"]
    direction = parsed["direction"]
    is_quadratic = parsed["is_quadratic"]
    applied_outputs = parsed["outputs"] or list(coefficients.keys())
    sign_map = {"+": 1.0, "-": -1.0, None: 0.0}
    has_dir = direction is not None

    for out_name in applied_outputs:
        if out_name not in coefficients:
            continue
        out_coefs = coefficients[out_name]
        sign = sign_map[direction]

        if len(factors) == 2 and has_dir and not is_quadratic:
            _apply_interaction_hint(out_coefs, factors, sign, rng)
        elif len(factors) == 1 and has_dir and not is_quadratic:
            _apply_main_hint(out_coefs, factors[0], sign)
        elif len(factors) == 1 and is_quadratic:
            _apply_quadratic_hint(out_coefs, factors[0], direction, rng)


# ---------------------------------------------------------------------------
# Coefficient materialisation
# ---------------------------------------------------------------------------


def _empty_output_coefs(rng: np.random.Generator, factor_names: list[str]) -> dict[str, Any]:
    """Draw a baseline set of coefficients for one output."""
    k = len(factor_names)
    intercept = 50.0 + float(rng.normal(0.0, 10.0))
    main = {f: float(rng.normal(0.0, 5.0)) for f in factor_names}

    # Sparse: each pair is non-zero with 30 % probability.
    interactions: list[dict[str, Any]] = [
        {
            "factors": [factor_names[i], factor_names[j]],
            "coefficient": float(rng.normal(0.0, 2.5)),
        }
        for i in range(k)
        for j in range(i + 1, k)
        if rng.random() < 0.30
    ]

    quadratic: dict[str, float] = {}
    for f in factor_names:
        if rng.random() < 0.40:
            quadratic[f] = float(rng.normal(0.0, 2.0))

    return {
        "intercept": intercept,
        "main": main,
        "interactions": interactions,
        "quadratic": quadratic,
    }


def _total_abs_magnitude(out_coefs: dict[str, Any]) -> float:
    return (
        abs(out_coefs["intercept"])
        + sum(abs(v) for v in out_coefs["main"].values())
        + sum(abs(inter["coefficient"]) for inter in out_coefs["interactions"])
        + sum(abs(v) for v in out_coefs["quadratic"].values())
    )


def materialize_model(private_state: dict[str, Any]) -> dict[str, Any]:
    """Regenerate the full coefficient set from a ``private_state`` dict.

    Deterministic: identical *private_state* always returns identical
    coefficients, across processes and machines.  This is the secret
    the simulator hides from the LLM — callers that have *private_state*
    already have the model, so "revealing" is free.
    """
    seed = int(private_state["seed"])
    factors = private_state["factors"]
    outputs = private_state["outputs"]
    hints = private_state.get("structural_hints") or []
    noise_level = private_state.get("noise_level", "medium")
    time_drift = bool(private_state.get("time_drift", False))

    factor_names = [f["name"] for f in factors]
    output_names = [o["name"] for o in outputs]

    rng = np.random.default_rng(seed)

    per_output: dict[str, dict[str, Any]] = {
        name: _empty_output_coefs(rng, factor_names) for name in output_names
    }

    for hint in hints:
        if not isinstance(hint, str):
            continue
        _apply_hint(per_output, _parse_hint(hint, factor_names, output_names), rng)

    noise_fraction = _NOISE_FRACTIONS[noise_level]
    for coefs in per_output.values():
        coefs["noise_sigma"] = noise_fraction * _total_abs_magnitude(coefs)
        if time_drift:
            coefs["drift_rate_per_day"] = float(
                rng.normal(0.0, 0.01) * abs(coefs["intercept"])
            )
        else:
            coefs["drift_rate_per_day"] = 0.0

    return {
        "per_output": per_output,
        "noise_level": noise_level,
        "time_drift": time_drift,
        "model_version": MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _coded_setting(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return 2.0 * (value - low) / (high - low) - 1.0


def _resolve_setting(
    name: str,
    low: float,
    high: float,
    settings: dict[str, float],
    warnings: list[str],
) -> float:
    """Return the effective factor value, warning on missing/clipped inputs."""
    if name not in settings:
        val = (low + high) / 2.0
        warnings.append(f"Factor {name!r} not provided; using mid-range value {val}.")
        return val
    val = float(settings[name])
    if val < low:
        warnings.append(f"Factor {name!r}={val} below low={low}; clipped to {low}.")
        return low
    if val > high:
        warnings.append(f"Factor {name!r}={val} above high={high}; clipped to {high}.")
        return high
    return val


def _evaluate_surface(
    out_coefs: dict[str, Any],
    coded: dict[str, float],
    timestamp_offset_days: float,
    noise_rng: np.random.Generator,
) -> float:
    """Evaluate the full polynomial for one output, including noise + drift."""
    y = float(out_coefs["intercept"])
    for f_name, coef in out_coefs["main"].items():
        y += coef * coded[f_name]
    for inter in out_coefs["interactions"]:
        a, b = inter["factors"]
        y += inter["coefficient"] * coded[a] * coded[b]
    for f_name, coef in out_coefs["quadratic"].items():
        y += coef * (coded[f_name] ** 2)
    y += out_coefs["drift_rate_per_day"] * float(timestamp_offset_days)
    y += float(noise_rng.normal(0.0, out_coefs["noise_sigma"]))
    return y


def simulate(
    private_state: dict[str, Any],
    settings: dict[str, float],
    timestamp_offset_days: float = 0.0,
) -> dict[str, Any]:
    """Evaluate the hidden response surface at *settings*, with fresh noise.

    Parameters
    ----------
    private_state:
        The state dict produced by ``create_simulator`` (and persisted
        by the host).  Passed in whole so this function stays pure.
    settings:
        Mapping of factor-name → numeric value, in the same units as
        the factor's declared range.  Missing factors are filled with
        the mid-range value (with a warning).  Out-of-range values are
        clipped to the declared bounds (with a warning).
    timestamp_offset_days:
        Optional time axis passed by the caller when ``time_drift`` is
        enabled on the simulator.  Ignored otherwise.

    Returns
    -------
    dict
        ``{settings, outputs, warnings, timestamp_offset_days}``.
    """
    model = materialize_model(private_state)
    factor_ranges = {
        f["name"]: (float(f["low"]), float(f["high"]))
        for f in private_state["factors"]
    }

    warnings: list[str] = []
    effective_settings: dict[str, float] = {}
    coded: dict[str, float] = {}
    for name, (low, high) in factor_ranges.items():
        val = _resolve_setting(name, low, high, settings, warnings)
        effective_settings[name] = val
        coded[name] = _coded_setting(val, low, high)

    # Fresh (unseeded) RNG — noise is genuinely different on every call.
    noise_rng = np.random.default_rng()
    outputs: dict[str, float] = {
        out_name: float(_evaluate_surface(out_coefs, coded, timestamp_offset_days, noise_rng))
        for out_name, out_coefs in model["per_output"].items()
    }

    return {
        "settings": effective_settings,
        "outputs": outputs,
        "warnings": warnings,
        "timestamp_offset_days": float(timestamp_offset_days),
    }


def draw_initial_seed() -> int:
    """Return a non-negative int suitable for use as a seed in *private_state*."""
    # Keep it in a comfortable numpy range (< 2**31) so JSON int handling and
    # SQLAlchemy Integer columns don't need to think about width.
    return int(np.random.SeedSequence().entropy % (2**31))
