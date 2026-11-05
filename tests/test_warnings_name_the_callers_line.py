"""Every warning a user's call can raise in multivariate/ and batch/ names that call's line (#1079, #1092).

Each one passes ``stacklevel=user_stacklevel()``, which walks out of process_improve, scikit-learn
and mutmut. A fixed count named a line in the library as soon as a frame came between the call
and the warning: a private helper, scikit-learn's ``_fit_context`` around ``fit``, a subclass's
``fit`` or another model's inner fit, or the trampoline through which mutmut calls each function
it mutates. Every case below calls the library from this file, so the warning's file is this one.
"""

from __future__ import annotations

import types
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PCA

_RNG = np.random.default_rng(0)
#: Thirty rows of four columns and one response that depends on three of them.
X = pd.DataFrame(_RNG.normal(size=(30, 4)), columns=list("abcd"))


#: Where mutmut keeps the trampoline through which it calls each function it mutates.
_TRAMPOLINE = "/site-packages/mutmut/mutation/trampoline.py"


def _as_mutmut_calls(method: Callable) -> Callable:
    """Return ``method`` called through a frame of mutmut's trampoline, as mutmut calls a mutated function."""

    def trampoline(*args: object, **kwargs: object) -> object:
        return method(*args, **kwargs)

    code = trampoline.__code__.replace(co_filename=_TRAMPOLINE)
    return types.FunctionType(code, {"__name__": "mutmut.mutation.trampoline"}, closure=trampoline.__closure__)


def test_a_warning_under_mutation_testing_names_the_callers_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutmut's trampoline sits between a method and the library code that called it (#1092).

    Its file is outside the package, so a walk to the first frame outside the package stopped
    there, and the nightly mutation run failed before it tried a single mutant.
    """
    monkeypatch.setattr(PCA, "_fit_nipals", _as_mutmut_calls(PCA._fit_nipals))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PCA(n_components=1, algorithm="nipals", max_iter=1).fit(X)
    found = [w for w in caught if "without converging" in str(w.message)]
    assert [w.filename for w in found] == [__file__]
