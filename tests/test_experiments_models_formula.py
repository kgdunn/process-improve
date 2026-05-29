"""Unit tests for the SEC-01 formula guard in ``experiments/models.py``.

``validate_formula_is_safe`` is the boundary that stops an untrusted model
formula from reaching patsy, which evaluates formula terms as arbitrary
Python expressions.
"""

from __future__ import annotations

import pytest

from process_improve.experiments.models import UnsafeFormulaError, validate_formula_is_safe

_COLUMNS = ["A", "B", "C", "y"]


class TestValidateFormulaIsSafe:
    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ A*B*C",
            "y ~ A + B + C",
            "y ~ A*B",
            "y ~ A:B + C",
            "y ~ (A + B)^2",
            "y ~ A + B - 1",
            "  y ~ A + B  ",
        ],
    )
    def test_legitimate_wilkinson_formulas_pass(self, formula: str) -> None:
        # Should not raise.
        validate_formula_is_safe(formula, _COLUMNS)

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ I(__import__('os').system('id'))",  # classic RCE payload
            "y ~ A + __import__('os')",               # dunder
            "y ~ A + np.sin(B)",                       # attribute access + unknown name
            "y ~ A + open('secret')",                  # string literal + unknown call
            "y ~ A + eval('1')",                       # eval
            "y ~ A; import os",                         # statement injection char ';'
            "y ~ A + B == C",                           # '=' not permitted
        ],
    )
    def test_malicious_formulas_are_rejected(self, formula: str) -> None:
        with pytest.raises(UnsafeFormulaError):
            validate_formula_is_safe(formula, _COLUMNS)

    def test_unknown_column_is_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError, match="unknown name"):
            validate_formula_is_safe("y ~ A + Z", _COLUMNS)

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError, match="must be a string"):
            validate_formula_is_safe(["y ~ A"], _COLUMNS)  # type: ignore[arg-type]

    def test_dunder_is_rejected_first(self) -> None:
        with pytest.raises(UnsafeFormulaError, match="dunder"):
            validate_formula_is_safe("y ~ A.__class__", _COLUMNS)
