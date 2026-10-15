"""Record what ``compare_designs`` returns without a region, for an equivalence test.

``compare_designs`` gained a ``region`` argument; without it, every number must stay
exactly what it was. This script records the result of a fixed set of comparisons, made
BEFORE the region was added, so that ``tests/test_compare_designs.py`` can check the
current code reproduces them.

Each case stores its designs as exact runs (float reprs, so the test does not depend on
``generate_design`` staying the same), the model and settings, and the comparison's
``to_dict()``. The correlation maps are stored for the cases of three factors only; for
more factors they are large and ``max_r`` already reads them.

Run from the repository root::

    uv run python tests/fixtures/compare_designs_golden/prepare_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import scipy

from process_improve.experiments import Factor, compare, compare_designs, generate_design

HERE = pathlib.Path(__file__).parent


def _factors(names: str) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def _runs(design: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return a design's coded runs and factor names, exactly."""
    frame = design.design[list(design.factor_names)]
    return {"columns": list(frame.columns), "runs": frame.to_numpy(dtype=float).tolist()}


def _bbd(n_centre: int) -> dict[str, Any]:
    """Return the book's twelve-point Box-Behnken arrangement with ``n_centre`` centre runs."""
    half = np.array([[0, 1, -1], [0, 1, 1], [1, -1, 0], [1, 0, -1], [1, 0, 1], [1, 1, 0]], dtype=float)
    runs = np.vstack([half, -half, np.zeros((n_centre, 3))])
    return {"columns": ["A", "B", "C"], "runs": runs.tolist()}


def cases() -> list[dict[str, Any]]:
    """Return the comparisons to record: each one's designs, model and settings."""
    abc = _factors("ABC")
    return [
        {
            "name": "book_centre_runs",
            "model": "main_quadratic",
            "kwargs": {},
            "designs": {"BBD 1": _bbd(1), "BBD 3": _bbd(3), "BBD 5": _bbd(5)},
            "correlations": True,
        },
        {
            "name": "three_factor_designs",
            "model": "quadratic",
            "kwargs": {"effect_size": 1.5, "significance_level": 0.1},
            "designs": {
                "Box-Behnken": _runs(generate_design(abc, "box_behnken", n_center_points=3)),
                "Rotatable CCD": _runs(generate_design(abc, "ccd", alpha="rotatable", n_center_points=3)),
                "Face-centred CCD": _runs(generate_design(abc, "ccd", alpha="face_centered", n_center_points=2)),
                "2^3": _runs(generate_design(abc, "full_factorial", n_center_points=0)),
            },
            "correlations": True,
        },
        {
            "name": "three_factor_screening",
            "model": "main_quadratic",
            "kwargs": {},
            "designs": {
                "DSD": _runs(generate_design(abc, "dsd")),
                "PB": _runs(generate_design(abc, "plackett_burman", n_center_points=0)),
            },
            "correlations": True,
        },
        {
            "name": "six_factor_box_behnken",
            "model": "main_quadratic",
            "kwargs": {},
            "designs": {"BBD": _runs(generate_design(_factors("ABCDEF"), "box_behnken", n_center_points=3))},
            "correlations": False,
        },
        {
            "name": "eleven_factor_dsd",
            "model": "main_effects",
            "kwargs": {},
            "designs": {"DSD": _runs(generate_design(_factors("ABCDEFGHJKL"), "dsd", n_center_points=3))},
            "correlations": False,
        },
    ]


def compared(case: dict[str, Any]) -> dict[str, Any]:
    """Return ``compare_designs(...).to_dict()`` for a case, without the correlations when it says so."""
    designs = {
        label: pd.DataFrame(design["runs"], columns=design["columns"]) for label, design in case["designs"].items()
    }
    result = compare_designs(designs, case["model"], **case["kwargs"]).to_dict()
    if not case["correlations"]:
        result.pop("correlations")
    return result


def main() -> None:
    """Write ``golden.json``."""
    print(f"Recording from {compare.__file__}")
    described = subprocess.run(
        ["git", "describe", "--always", "--dirty"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=HERE,
    ).stdout.strip()
    recorded = [{**case, "result": compared(case)} for case in cases()]
    payload = {
        "generated_from": {
            "tree": described,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "cases": recorded,
    }
    lines = [json.dumps(case) for case in payload["cases"]]
    text = '{"generated_from": ' + json.dumps(payload["generated_from"]) + ', "cases": [\n'
    text += ",\n".join(lines) + "\n]}\n"
    (HERE / "golden.json").write_text(text)


if __name__ == "__main__":
    main()
