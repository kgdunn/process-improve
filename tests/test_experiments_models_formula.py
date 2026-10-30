"""Unit tests for the SEC-01 formula guard in ``experiments/models.py``.

``validate_formula_is_safe`` is the boundary that stops an untrusted model
formula from reaching patsy, which evaluates formula terms as arbitrary
Python expressions.
"""

from __future__ import annotations

import keyword
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from process_improve.experiments._lm import (
    real_names,
    stand_ins_for,
    suggest_identifier,
    validate_identifiers_are_safe,
    with_stand_ins,
)
from process_improve.experiments.models import (
    Model,
    UnsafeFormulaError,
    lm,
    predict,
    summary,
    validate_formula_is_safe,
    validate_identifier_is_safe,
)

_COLUMNS = ["A", "B", "C", "y", "d"]
#: No column C beside a formula that calls C(): patsy would find the column first (see TestColumnNamedLikeACall).
_CATEGORICAL_COLUMNS = ["A", "B", "y", "d"]


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

    def test_refusal_suggests_a_name(self) -> None:
        with pytest.raises(UnsafeFormulaError, match=r"Try 'Water_pct'\.$"):
            validate_identifier_is_safe("Water %")


class TestSuggestIdentifier:
    """A refused name comes with a name close to it that the guard and Python both accept."""

    @pytest.mark.parametrize(
        ("name", "suggestion"),
        [
            ("Water %", "Water_pct"),
            ("Yield (%)", "Yield_pct"),
            ("Temp (°C)", "Temp_deg_C"),
            ("Run #", "Run_num"),
            ("Conc. [g/L]", "Conc_g_L"),
            ("2nd stage", "x2nd_stage"),
            ("  Yield  ", "Yield"),
            ("a__b", "a_b"),
            ("__import__", "import_"),
            ("lambda", "lambda_"),
            ("Été", "xÉté"),
            ("x²", "x"),
            ("", "x"),
            (0, "x0"),
        ],
    )
    def test_suggestion(self, name: object, suggestion: str) -> None:
        assert suggest_identifier(name) == suggestion

    @given(st.text())
    def test_every_suggestion_is_accepted(self, name: str) -> None:
        suggestion = suggest_identifier(name)
        validate_identifier_is_safe(suggestion)
        assert suggestion.isidentifier()
        assert not keyword.iskeyword(suggestion)


class TestValidateIdentifiersAreSafe:
    """Every refused name is in one error, with a rename map whose names are all distinct."""

    def test_safe_names_pass(self) -> None:
        validate_identifiers_are_safe(["A", "Temp_2", "_x"])

    def test_every_refused_name_in_one_error(self) -> None:
        with pytest.raises(UnsafeFormulaError, match=re.escape("{'Water %': 'Water_pct', 'Temp (°C)': 'Temp_deg_C'}")):
            validate_identifiers_are_safe(["Water %", "A", "Temp (°C)"])

    def test_suggestions_do_not_take_a_name_already_used(self) -> None:
        """'Water %' and 'Water (%)' both suggest 'Water_pct', which a third column already is."""
        renames = "{'Water %': 'Water_pct_2', 'Water (%)': 'Water_pct_3'}"
        with pytest.raises(UnsafeFormulaError, match=re.escape(renames)):
            validate_identifiers_are_safe(["Water %", "Water (%)", "Water_pct"])


class TestColumnNamedLikeACall:
    """Patsy finds a column before its own function of the same name, so the formula would call the column."""

    @pytest.mark.parametrize(
        ("formula", "columns", "renames"),
        [
            pytest.param("y ~ (I + P) ** 2 + I(I ** 2) + I(P ** 2)", ["y", "I", "P"], {"I": "I_"}, id="I-squares"),
            pytest.param("y ~ A + I(P ** 2)", ["y", "A", "P", "I", "I_"], {"I": "I_2"}, id="I_-already-a-column"),
            pytest.param("y ~ C(A, Sum)", ["y", "A", "C"], {"C": "C_"}, id="C-contrast"),
            pytest.param("y ~ C(A, Sum)", ["y", "A", "Sum"], {"Sum": "Sum_"}, id="Sum-contrast"),
            pytest.param("y ~ np.log(A)", ["y", "A", "np"], {"np": "np_"}, id="np-module"),
        ],
    )
    def test_column_named_like_a_call_is_refused(self, formula: str, columns: list[str], renames: dict) -> None:
        with pytest.raises(UnsafeFormulaError, match=re.escape(f".rename(columns={renames!r})")):
            validate_formula_is_safe(formula, columns, allow_transforms=True, allow_numpy=True)

    @pytest.mark.parametrize(
        ("formula", "columns"),
        [
            pytest.param("y ~ I + P + I:P", ["y", "I", "P"], id="I-named-not-called"),
            pytest.param("y ~ C(Sum)", ["y", "Sum"], id="Sum-is-the-data-C-codes"),
            pytest.param("y ~ A + B + C", ["y", "A", "B", "C"], id="C-named-not-called"),
        ],
    )
    def test_column_the_formula_only_names_is_accepted(self, formula: str, columns: list[str]) -> None:
        validate_formula_is_safe(formula, columns, allow_transforms=True)


class TestStandInNames:
    """A column named like a function the formula calls is fitted under a stand-in, and reported as itself (#986)."""

    @pytest.mark.parametrize(
        ("formula", "columns", "fitted"),
        [
            pytest.param(
                "y ~ (I + P) ** 2 + I(I ** 2) + I(P ** 2)",
                ["y", "I", "P"],
                "y ~ (I_ + P) ** 2 + I(I_ ** 2) + I(P ** 2)",
                id="I-squares",
            ),
            pytest.param(
                "y ~ A + I(P ** 2)", ["y", "A", "P", "I", "I_"], "y ~ A + I(P ** 2)", id="I_-already-a-column"
            ),
            pytest.param("C(C, Sum) + C(B, Sum)", ["B", "C"], "C(C_, Sum) + C(B, Sum)", id="C-contrast"),
            pytest.param("y ~ C(A, Sum) + Sum", ["y", "A", "Sum"], "y ~ C(A, Sum) + Sum_", id="Sum-contrast"),
            pytest.param("y ~ np.log(np)", ["y", "np"], "y ~ np.log(np_)", id="np-module"),
            pytest.param("I ~ A + I(A ** 2)", ["I", "A"], "I_ ~ A + I(A ** 2)", id="I-response"),
            pytest.param(
                "y ~ (I:P +\n  I(I ** 2))", ["y", "I", "P"], "y ~ (I_:P +\n  I(I_ ** 2))", id="over-two-lines"
            ),
            pytest.param("y ~ I + P + I:P", ["y", "I", "P"], "y ~ I + P + I:P", id="I-named-not-called"),
            pytest.param(
                "y ~ C(A, Treatment('I')) + I(I ** 2)",
                ["y", "A", "I"],
                "y ~ C(A, Treatment('I')) + I(I_ ** 2)",
                id="a-level-I-is-not-the-column",
            ),
        ],
    )
    def test_the_formula_is_written_with_each_hidden_column_under_its_stand_in(
        self, formula: str, columns: list[str], fitted: str
    ) -> None:
        stand_ins = stand_ins_for(formula, columns)
        assert with_stand_ins(formula, stand_ins) == fitted
        assert real_names(fitted, stand_ins) == formula
        validate_formula_is_safe(
            fitted, [stand_ins.get(c, c) for c in columns], allow_transforms=True, allow_numpy=True
        )

    def test_a_formula_that_does_not_parse_has_no_stand_ins(self) -> None:
        assert stand_ins_for("y ~ I(I ** 2) +", ["y", "I"]) == {}
        assert stand_ins_for(["y ~ I(I ** 2)"], ["y", "I"]) == {}  # type: ignore[arg-type]

    def test_the_results_are_read_under_the_real_names(self) -> None:
        """A column named ``I_`` keeps its name beside a column ``I`` fitted as ``I_2``, as does a level ``I_2``."""
        results = {"I_2": ["I(I_2 ** 2)", ("I_2:I_", "I_"), "P[T.I_2]", 1.5], "note": "I_2 is I_"}
        assert real_names(results, {"I": "I_2"}) == {
            "I": ["I(I ** 2)", ("I:I_", "I_"), "P[T.I_2]", 1.5],
            "note": "I is I_",
        }


class TestLinearModelOfAFactorNamedLikeAFunction:
    """``lm`` fits a factor named I in a model with squares, and names every term with I (#986)."""

    @staticmethod
    def _fit(name: str) -> Model:
        axial = 2**0.5
        x = np.array([-1, 1, -1, 1, 0, 0, 0, -axial, axial, 0, 0])
        z = np.array([-1, -1, 1, 1, 0, 0, 0, 0, 0, -axial, axial])
        noise = np.random.default_rng(0).normal(0, 0.1, len(x))
        data = pd.DataFrame({name: x, "P": z, "y": 5 + x - x**2 + z + noise})
        return lm(f"y ~ {name} * P + I({name} ** 2)", data)

    def test_the_fit_is_that_of_the_same_factor_named_otherwise(self) -> None:
        named_i, named_x = self._fit("I"), self._fit("X")
        expected = named_x.get_parameters(drop_intercept=False).rename(lambda term: term.replace("X", "I"))
        pd.testing.assert_series_equal(named_i.get_parameters(drop_intercept=False), expected)
        assert named_i.get_factor_names(1) == ["I", "P", "I(I ** 2)"]
        assert predict(named_i, I=0.5, P=0.1).iloc[0] == pytest.approx(predict(named_x, X=0.5, P=0.1).iloc[0])
        assert "I(I ** 2)" in str(summary(named_i, show=False))
        assert "I_" not in str(summary(named_i, show=False))

    def test_aliases_are_named_with_the_real_name(self) -> None:
        """Two runs at I = P alias I with P; the alias reads I, not the stand-in."""
        data = pd.DataFrame({"I": [-1.0, 1.0, -1.0, 1.0, 0.0], "y": [1.0, 3.0, 1.2, 2.9, 2.5]})
        data["P"] = data["I"]
        fitted = lm("y ~ I + P + I(I ** 2)", data)
        assert fitted.aliasing == {("I",): [["+", "P"]], ("P",): [["+", "I"]]}
        assert list(fitted.get_parameters().index) == ["I", "I(I ** 2)"]


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
        validate_formula_is_safe(formula, _CATEGORICAL_COLUMNS, allow_transforms=True)

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
        validate_formula_is_safe(formula, _CATEGORICAL_COLUMNS, allow_transforms=True, allow_numpy=True)


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
