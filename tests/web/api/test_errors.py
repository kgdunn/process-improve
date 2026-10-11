"""What the page tells a reader whose request it refuses.

Every refusal is a sentence the page shows as it is, so it must say what is wrong in the
reader's terms: which factor, which value, what to do instead.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        (FACTORS[:1], "at least two"),
        ([FACTORS[0], {**FACTORS[1], "name": "T"}], "unique"),
        ([FACTORS[0], {**FACTORS[1], "name": "a b"}], "variable names"),
        ([FACTORS[0], {**FACTORS[1], "low": 5}], "below the high"),
        ([FACTORS[0], {**FACTORS[1], "low": None}], "enter a number"),  # a blank box, which the page sends as null
        ([FACTORS[0], {**FACTORS[1], "high": "abc"}], "enter a number"),
    ],
)
def test_bad_factor_input_is_reported(api, factors, message):
    assert message in api.error("api_make_design", {"design_type": "box_behnken", "factors": factors})
