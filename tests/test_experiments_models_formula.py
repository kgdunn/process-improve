"""Unit tests for the SEC-01 formula guard in ``experiments/models.py``.

``validate_formula_is_safe`` is the boundary that stops an untrusted model
formula from reaching patsy, which evaluates formula terms as arbitrary
Python expressions.
"""

from __future__ import annotations

import re

import pytest

from process_improve.experiments.models import (
    UnsafeFormulaError,
    validate_formula_is_safe,
    validate_identifier_is_safe,
)

_COLUMNS = ["A", "B", "C", "y", "d"]


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
            "y ~ A + __import__('os')",  # dunder
            "y ~ A + np.sin(B)",  # attribute access + unknown name
            "y ~ A + open('secret')",  # string literal + unknown call
            "y ~ A + eval('1')",  # eval
            "y ~ A; import os",  # statement injection char ';'
            "y ~ A + B == C",  # '=' not permitted
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


class TestValidateFormulaAllowTransforms:
    """SEC-14: ``allow_transforms`` permits I()/Q() of column arithmetic only."""

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ A + I(A ** 2)",
            "y ~ I(1 / A)",
            "(A + B) ** 2 + I(A ** 2) + I(B ** 2)",  # the 'quadratic' shorthand RHS
            "y ~ Q(A) + B",
            "y ~ A*B*C",  # plain Wilkinson still fine
        ],
    )
    def test_transforms_pass(self, formula: str) -> None:
        validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True)

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ A + np.log(A)",  # numpy is off in this mode
            "y ~ I(__import__('os').system('id'))",
            "y ~ I(os.system(A))",  # attribute call on a non-np name
            "y ~ I('x')",  # string literal argument
            "y ~ I(A); import os",  # statement injection
            "y ~ open(A)",  # disallowed call
            "y ~ A + Z",  # unknown column
        ],
    )
    def test_unsafe_transforms_rejected(self, formula: str) -> None:
        with pytest.raises(UnsafeFormulaError):
            validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True)


class TestValidateFormulaAllowNumpy:
    """SEC-14: ``allow_numpy`` permits a curated allowlist of np.<func> calls."""

    @pytest.mark.parametrize(
        "formula",
        [
            "np.log10(y) ~ C",  # transform on the response (LHS) side
            "y ~ d + I(np.power(d, 2))",
            "y ~ d + I(1 / np.sqrt(d))",
            "y ~ np.log(A) + np.exp(B)",
        ],
    )
    def test_numpy_transforms_pass(self, formula: str) -> None:
        validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True, allow_numpy=True)

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ np.load(A)",  # not in the curated allowlist (deserialises pickles)
            "y ~ np.fromfile(A)",
            "y ~ np.__class__",  # dunder via attribute
            "y ~ os.system(A)",  # non-np module
            "y ~ np.log('A')",  # string literal argument
        ],
    )
    def test_unsafe_numpy_rejected(self, formula: str) -> None:
        with pytest.raises(UnsafeFormulaError):
            validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True, allow_numpy=True)

    def test_too_many_tildes_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError, match="at most one"):
            validate_formula_is_safe("y ~ A ~ B", _COLUMNS, allow_transforms=True)


class TestValidateIdentifierIsSafe:
    """SEC-14: column / response names must be plain identifiers."""

    @pytest.mark.parametrize("name", ["A", "foo_1", "Temperature", "_x"])
    def test_valid_identifiers_pass(self, name: str) -> None:
        validate_identifier_is_safe(name)

    @pytest.mark.parametrize(
        "name",
        [
            "A); import os; (",
            "__import__",
            "1bad",
            "a b",
            "np.x",
            "A-B",
        ],
    )
    def test_unsafe_identifiers_rejected(self, name: str) -> None:
        with pytest.raises(UnsafeFormulaError):
            validate_identifier_is_safe(name)

    def test_non_string_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError, match="must be a string"):
            validate_identifier_is_safe(123)  # type: ignore[arg-type]


class TestCategoricalContrasts:
    """Patsy categorical-contrast helpers (C, Treatment, Sum, ...) are allowed."""

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ C(A)",
            "y ~ C(A, Sum)",
            "y ~ C(A, Treatment)",
            "y ~ C(A, Treatment(reference=0))",
            "y ~ C(A, Diff) + C(B, Helmert)",
            "y ~ B + C(A) + C(A):B",
            "y ~ C(A, Poly)",
        ],
    )
    def test_categorical_contrasts_allowed(self, formula: str) -> None:
        validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True)

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ C(A, os.system)",  # contrast arg is not a helper/literal/column
            "y ~ C(A, evil())",  # nested non-contrast call
            "y ~ Contrast(A)",  # not an allowlisted contrast helper
            "y ~ C(A, __import__)",  # dunder name as contrast
        ],
    )
    def test_unsafe_categorical_calls_rejected(self, formula: str) -> None:
        with pytest.raises(UnsafeFormulaError):
            validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True, allow_numpy=True)

    @pytest.mark.parametrize(
        "formula",
        [
            pytest.param("y ~ C(A, Treatment('lo'))", id="positional-string-reference-level"),
            pytest.param("y ~ C(A, levels=['lo', 'hi'])", id="keyword-list-of-levels"),
        ],
    )
    def test_literal_contrast_arguments_allowed(self, formula: str) -> None:
        """A reference level or level list given as literals is a legitimate contrast specification."""
        validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True, allow_numpy=True)


class TestValidateFormulaRejectionMessages:
    """Each construct the AST check refuses is named in the error, so a caller knows what to change."""

    @pytest.mark.parametrize(
        ("formula", "message"),
        [
            pytest.param("y ~ A % B", "operator Mod is not allowed in a formula.", id="modulo-operator"),
            pytest.param("y ~ not A", "unary operator Not is not allowed in a formula.", id="unary-not"),
            pytest.param("y ~ I(A, x=1)", "keyword arguments are not allowed in a formula call.", id="transform-kwarg"),
            pytest.param("y ~ I(*A)", "starred arguments are not allowed in a formula call.", id="transform-starred"),
            pytest.param("y ~ C(*A)", "starred arguments are not allowed in a formula call.", id="contrast-starred"),
            pytest.param("y ~ C(A, **B)", "**kwargs are not allowed in a formula call.", id="contrast-double-star"),
            pytest.param(
                "y ~ C(A, Treatment(reference=B))",
                "categorical-contrast arguments must be column names, contrast helpers, or literals.",
                id="contrast-keyword-names-a-column",
            ),
            pytest.param(
                "y ~ C(A, levels=[True])",
                "categorical-contrast arguments must be column names, contrast helpers, or literals.",
                id="contrast-boolean-level",
            ),
        ],
    )
    def test_rejected_construct_is_named(self, formula: str, message: str) -> None:
        """The error message names the refused construct exactly."""
        with pytest.raises(UnsafeFormulaError, match=f"^{re.escape(message)}$"):
            validate_formula_is_safe(formula, _COLUMNS, allow_transforms=True, allow_numpy=True)
