"""Tests for the SEC-07 matrix-conditioning guards in ``process_improve._linalg``.

``np.linalg.inv`` returns overflow-driven garbage on an ill-conditioned matrix
without raising. ``safe_inverse`` / ``is_singular`` detect that up front.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve._linalg import DEFAULT_COND_LIMIT, is_singular, safe_inverse


class TestIsSingular:
    def test_well_conditioned_identity(self) -> None:
        assert not is_singular(np.eye(3))

    def test_well_conditioned_random(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 4)) + 4 * np.eye(4)
        assert not is_singular(a)

    def test_exactly_singular(self) -> None:
        a = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        assert is_singular(a)

    def test_ill_conditioned(self) -> None:
        # A nearly rank-deficient matrix that np.linalg.inv would not reject.
        a = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-16]])
        assert is_singular(a)

    def test_non_square(self) -> None:
        assert is_singular(np.ones((2, 3)))

    def test_non_finite(self) -> None:
        assert is_singular(np.array([[np.nan, 0.0], [0.0, 1.0]]))


class TestSafeInverse:
    def test_matches_inv_for_well_conditioned(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(3, 3)) + 3 * np.eye(3)
        # Bit-identical to a bare inv for well-conditioned input.
        np.testing.assert_array_equal(safe_inverse(a), np.linalg.inv(a))

    def test_raises_on_singular(self) -> None:
        a = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(np.linalg.LinAlgError, match="singular or ill-conditioned"):
            safe_inverse(a, what="X'X")

    def test_error_message_includes_what_and_limit(self) -> None:
        a = np.zeros((2, 2))
        with pytest.raises(np.linalg.LinAlgError) as exc:
            safe_inverse(a, what="super-score covariance")
        msg = str(exc.value)
        assert "super-score covariance" in msg
        assert f"{DEFAULT_COND_LIMIT:.2e}" in msg

    def test_unguarded_inv_of_singular_is_not_finite(self) -> None:
        # Documents the failure mode the guard prevents: inv of a singular
        # matrix is non-finite (or raises), never a usable result.
        a = np.array([[1.0, 2.0], [2.0, 4.0]])
        with np.errstate(all="ignore"):
            try:
                inv = np.linalg.inv(a)
                assert not np.all(np.isfinite(inv))
            except np.linalg.LinAlgError:
                pass  # exactly-singular path: inv raises, also fine
