"""Without the ``expt`` extra (pyDOE3), every design builds except the Taguchi orthogonal arrays."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.modules["pyDOE3"] = None  # import pyDOE3 now raises ImportError
    from process_improve.experiments import Factor, generate_design

    four = [Factor(name=n, low=0, high=1) for n in "ABCD"]
    fifteen = [Factor(name=f"x{i}", low=0, high=1) for i in range(15)]
    cases = {
        "full_factorial": (four, {}),
        "fractional_factorial": (four + [Factor(name="E", low=0, high=1)], {}),
        "plackett_burman": (four, {}),
        "box_behnken": (four, {}),
        "ccd": (four, {}),
        "supersaturated": (fifteen, {"budget": 12}),
    }
    for design_type, (factors, kwargs) in cases.items():
        print(design_type, generate_design(factors, design_type, **kwargs).n_runs)
    try:
        generate_design(four, "taguchi")
    except ImportError as exc:
        print("taguchi ImportError", "process-improve[expt]" in str(exc))
    """
)


@pytest.mark.slow
def test_designs_build_without_pydoe3() -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test-controlled code
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True, timeout=300
    )
    lines = dict(line.split(" ", 1) for line in result.stdout.strip().splitlines())
    assert lines == {
        "full_factorial": "19",  # 2^4 plus 3 centre runs
        "fractional_factorial": "19",  # 2^(5-1) plus 3 centre runs
        "plackett_burman": "11",  # 8 plus 3 centre runs
        "box_behnken": "27",
        "ccd": "27",
        "supersaturated": "12",
        "taguchi": "ImportError True",
    }
