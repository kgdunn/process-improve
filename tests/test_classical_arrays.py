"""The built-in classical arrays reproduce pyDOE3's element for element, so designs do not change.

The equivalence tests need pyDOE3; the input checks do not.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from process_improve.experiments import _classical


@pytest.fixture(scope="module")
def pydoe3() -> ModuleType:
    return pytest.importorskip("pyDOE3")


@pytest.mark.parametrize("k", range(1, 11))
def test_ff2n_matches_pydoe3(pydoe3: ModuleType, k: int) -> None:
    np.testing.assert_array_equal(_classical.ff2n(k), pydoe3.ff2n(k))


@pytest.mark.parametrize(
    "gen",
    [
        "a b ab",
        "A B AB",
        "a b -ab c +abc",
        "a b c abc",
        "a b c d abc abd",
        "a b c d e abcd -abce bcde",
        "a b c ab ac bc",
    ],
)
def test_fracfact_matches_pydoe3(pydoe3: ModuleType, gen: str) -> None:
    np.testing.assert_array_equal(_classical.fracfact(gen), pydoe3.fracfact(gen))


@pytest.mark.parametrize("k", range(1, 48))
def test_pbdesign_matches_pydoe3(pydoe3: ModuleType, k: int) -> None:
    try:
        expected = pydoe3.pbdesign(k)
    except (AssertionError, ValueError, IndexError):
        with pytest.raises(ValueError, match="No Plackett-Burman construction"):
            _classical.pbdesign(k)
        return
    np.testing.assert_array_equal(_classical.pbdesign(k), expected)


@pytest.mark.parametrize(("k", "center"), [(k, c) for k in range(3, 18) for c in (None, 0, 3)])
def test_bbdesign_matches_pydoe3(pydoe3: ModuleType, k: int, center: int | None) -> None:
    np.testing.assert_array_equal(_classical.bbdesign(k, center=center), pydoe3.bbdesign(k, center=center))


@pytest.mark.parametrize(
    ("k", "center", "alpha", "face"),
    [
        (k, center, alpha, face)
        for k in range(2, 7)
        for center in ((4, 4), (0, 0), (2, 1))
        for alpha, face in itertools.product(("orthogonal", "rotatable"), ("circumscribed", "inscribed", "faced"))
    ],
)
def test_ccdesign_matches_pydoe3(pydoe3: ModuleType, k: int, center: tuple[int, int], alpha: str, face: str) -> None:
    np.testing.assert_allclose(
        _classical.ccdesign(k, center=center, alpha=alpha, face=face),
        pydoe3.ccdesign(k, center=center, alpha=alpha, face=face),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("gen", "message"),
    [
        ("ab ac", "At least one unconfounded main factor"),
        ("a a", "confounded with each other"),
        ("a c", "Use the letters"),
        ("a b ab ab", "not unique"),
        ("a b ac", "not valid"),
    ],
)
def test_fracfact_rejects_bad_generators(gen: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _classical.fracfact(gen)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _classical.fracfact("a  b"), "does not match the number of factors"),
        (lambda: _classical.pbdesign(0), "positive integer"),
        (lambda: _classical.bbdesign(2), "at least 3"),
        (lambda: _classical.ccdesign(1), "greater than 1"),
        (lambda: _classical.ccdesign(3, alpha="wide"), 'Invalid value for "alpha"'),
        (lambda: _classical.ccdesign(3, face="outside"), 'Invalid value for "face"'),
        (lambda: _classical.ccdesign(3, center=(1, 2, 3)), "expected 2, but got 3"),
    ],
    ids=["fracfact-spacing", "pb-zero", "bb-two", "ccd-one", "ccd-alpha", "ccd-face", "ccd-center"],
)
def test_invalid_arguments_raise(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()
