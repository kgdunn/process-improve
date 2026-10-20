"""The defining relation of a fraction, on request, however many generators it has (#788 A9).

A design lists its defining relation up to ten generators (1023 words). Montgomery's Appendix
Table X (design z) is the saturated 2^(15-11) in 16 runs, whose relation has 2047 words; the
design lists none of them, and ``defining_relation(design)`` works them all out. These tests hold
it to the definition: a word is in the relation when its product column is the same on every run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from process_improve.experiments import Factor, defining_relation, generate_design

FIFTEEN = "ABCDEFGHJKLMNOP"


def fraction(names: str, **options):
    factors = [Factor(name=name, low=-1, high=1) for name in names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # no centre runs, so no pure error: wanted here
        return generate_design(factors, design_type="fractional_factorial", n_center_points=0, **options)


def words_constant_on_every_run(design) -> set[str]:
    """Return every word whose product column is +1 or -1 on all runs, by trying each of the 2^k - 1 words."""
    signs = design.design[list(design.factor_names)].to_numpy(dtype=float)
    runs = (signs < 0).astype(np.int64) @ (1 << np.arange(signs.shape[1], dtype=np.int64))
    words = np.arange(1, 1 << signs.shape[1], dtype=np.int64)
    odd = np.bitwise_count(words[:, None] & runs[None, :]) % 2  # 1 where the word's product is -1 on that run
    constant = (odd == odd[:, :1]).all(axis=1)
    found = set()
    for word, first in zip(words[constant], odd[constant, 0], strict=True):
        letters = "".join(name for i, name in enumerate(design.factor_names) if int(word) >> i & 1)
        found.add(("I=-" if first else "I=") + letters)
    return found


def test_the_saturated_2_to_the_15_minus_11_has_2047_words():
    design = fraction(FIFTEEN, resolution=3)
    assert design.n_runs == 16
    assert len(design.generators) == 11
    assert design.defining_relation is None, "past ten generators the design lists no relation"
    assert "defining_relation(design) lists them all" in design.metadata["defining_relation_note"]
    relation = defining_relation(design)
    assert len(relation) == 2**11 - 1
    assert set(relation) == words_constant_on_every_run(design)
    assert min(len(word.removeprefix("I=")) for word in relation) == 3, "resolution III"


def test_the_order_is_shortest_first_then_by_factor():
    relation = defining_relation(fraction(FIFTEEN, resolution=3))
    lengths = [len(word.removeprefix("I=")) for word in relation]
    assert lengths == sorted(lengths)


def test_a_relation_the_design_lists_is_the_same_list():
    """Montgomery's Table 8.14: the 16-run fraction for 12 factors has 16 words of length three."""
    design = fraction("ABCDEFGHJKLM", resolution=3)
    assert design.n_runs == 16
    assert defining_relation(design) == design.defining_relation
    assert len(design.defining_relation) == 255
    assert sum(len(word) == len("I=ABC") for word in design.defining_relation) == 16


def test_the_signs_of_negated_generators_carry_through():
    design = fraction("ABCDE", generators=["D=-AB", "E=AC"])
    assert defining_relation(design) == design.defining_relation == ["I=-ABD", "I=ACE", "I=-BCDE"]
    assert set(defining_relation(design)) == words_constant_on_every_run(design)


def test_a_full_factorial_has_an_empty_relation():
    factors = [Factor(name=name, low=-1, high=1) for name in "ABC"]
    assert defining_relation(generate_design(factors, design_type="full_factorial")) == []


def test_a_design_with_no_generators_has_no_relation_to_list():
    factors = [Factor(name=name, low=-1, high=1) for name in "ABCDEFG"]
    with pytest.raises(ValueError, match="has no generators, so it has no defining relation"):
        defining_relation(generate_design(factors, design_type="plackett_burman"))
