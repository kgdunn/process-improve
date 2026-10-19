"""Blocks of a two-level factorial from the interaction words the user gives (#788 A8).

Montgomery's chapters 7 and 8 block factorials and fractions by named words: a 2^5 in four blocks
by ADE and BCE (p. 314), a 2^6 in eight blocks by ABEF, ABCD and ACE (p. 315), the arrangements
of Table 7.9, and the photoresist half fraction blocked by ABCE (p. 371). These tests build each
from its printed words and check what is confounded with the blocks, the aliases that go with
them, and the runs themselves.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from process_improve.experiments import Factor, generate_design


def factorial(names: str | list[str], **options):
    factors = [Factor(name=name, low=-1, high=1) for name in names]
    design_type = "fractional_factorial" if "generators" in options else "full_factorial"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # no centre runs, so no pure error: wanted here
        return generate_design(factors, design_type=design_type, n_center_points=0, **options)


def word_column(design, word: str) -> np.ndarray:
    return np.prod(design.design[list(word)].to_numpy(dtype=float), axis=1)


# --------------------------------------------------------------------------- Montgomery's arrangements
def test_a_2_to_the_5_in_four_blocks_by_ade_and_bce():
    """Page 314: "ADE, BCE, and ABCD are all confounded with blocks"; the blocks hold 8 runs each."""
    design = factorial("ABCDE", block_generators=["ADE", "BCE"])
    blocking = design.metadata["blocking"]
    assert blocking["generators"] == ["ADE", "BCE"]
    assert sorted(blocking["confounded_with"]) == ["ABCD", "ADE", "BCE"]
    assert "aliases" not in blocking, "a full factorial has no aliases"
    blocks = design.design["Block"].to_numpy()
    assert sorted(np.bincount(blocks)[1:]) == [8, 8, 8, 8]
    for block in range(1, 5):
        inside = blocks == block
        for word in ("ADE", "BCE", "ABCD"):
            assert len(set(word_column(design, word)[inside])) == 1, (word, block)
        for word in ("A", "B", "C", "D", "E", "AB", "CE"):
            assert word_column(design, word)[inside].sum() == 0, (word, block)


def test_a_2_to_the_6_in_eight_blocks():
    """Page 315: ABEF, ABCD and ACE, and their generalized interactions CDEF, BCF, BDE and ADF."""
    design = factorial("ABCDEF", block_generators=["ABEF", "ABCD", "ACE"])
    assert sorted(design.metadata["blocking"]["confounded_with"]) == sorted(
        ["ABEF", "ABCD", "ACE", "CDEF", "BCF", "BDE", "ADF"]
    )
    assert sorted(np.bincount(design.design["Block"].to_numpy())[1:]) == [8] * 8


@pytest.mark.parametrize(
    ("names", "words", "confounded"),
    [
        ("ABC", ["ABC"], ["ABC"]),
        ("ABC", ["AB", "AC"], ["AB", "AC", "BC"]),
        ("ABCD", ["ABD", "ACD"], ["ABD", "ACD", "BC"]),
        ("ABCDE", ["ABC", "CDE"], ["ABC", "ABDE", "CDE"]),
    ],
)
def test_the_arrangements_of_table_7_9(names, words, confounded):
    design = factorial(names, block_generators=words)
    assert sorted(design.metadata["blocking"]["confounded_with"]) == sorted(confounded)


def test_the_photoresist_half_fraction_blocked_by_abce():
    """Page 371: with F = BCD, "ABCE plus its alias ADEF are confounded with blocks"."""
    design = factorial("ABCDEF", generators=["F=BCD"], block_generators=["ABCE"])
    blocking = design.metadata["blocking"]
    assert design.defining_relation == ["I=BCDF"]
    assert blocking["confounded_with"] == ["ABCE"]
    assert blocking["aliases"] == {"ABCE": ["ADEF"]}
    product = word_column(design, "ABCE")
    assert all(len(set(product[design.design["Block"] == b])) == 1 for b in (1, 2))
    assert np.array_equal(product, word_column(design, "ADEF"))


def test_the_library_s_own_blocking_of_a_fraction_reports_the_aliases_too():
    design = factorial("ABCDE", generators=["E=ABCD"], n_blocks=2)
    blocking = design.metadata["blocking"]
    [word] = blocking["confounded_with"]
    assert blocking["aliases"][word] == [
        "".join(sorted(set(word) ^ set("ABCDE"), key="ABCDE".index)),
    ], "each word's only alias is its product with ABCDE"


def test_factor_names_longer_than_one_letter_are_joined_with_a_star():
    design = factorial(["Temp", "Feed", "Speed"], block_generators=["Temp*Feed*Speed"])
    assert design.metadata["blocking"]["confounded_with"] == ["TempFeedSpeed"]
    product = np.prod(design.design[["Temp", "Feed", "Speed"]].to_numpy(dtype=float), axis=1)
    assert all(len(set(product[design.design["Block"] == b])) == 1 for b in (1, 2))


def test_n_blocks_may_be_given_too_when_it_agrees():
    design = factorial("ABCDE", n_blocks=4, block_generators=["ADE", "BCE"])
    assert design.design["Block"].nunique() == 4


def test_a_replicated_factorial_keeps_its_blocks_on_the_words():
    design = factorial("ABC", n_replicates=2, block_generators=["ABC"])
    product = word_column(design, "ABC")
    for block in (1, 2):
        assert len(set(product[design.design["Block"] == block])) == 1


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("names", "options", "message"),
    [
        ("ABCDE", {"n_blocks": 8, "block_generators": ["ADE", "BCE"]}, "8 blocks need 3 block generators"),
        ("ABCD", {"block_generators": ["AB", "CD", "ABCD"]}, "are not independent"),
        ("ABC", {"block_generators": ["AB", "ABC"]}, "confound the main effect C"),
        ("ABC", {"block_generators": ["AXZ"]}, "must be a product of the factors"),
        ("ABC", {"block_generators": "ABC"}, "must be a list of one or more words"),
        ("ABC", {"block_generators": []}, "must be a list of one or more words"),
        ("ABCDEF", {"generators": ["F=BCD"], "block_generators": ["BCDF"]}, "in the defining relation"),
    ],
)
def test_block_generators_that_cannot_block_the_design_are_refused(names, options, message):
    with pytest.raises(ValueError, match=message):
        factorial(names, **options)


def test_block_generators_on_a_design_that_is_not_a_two_level_factorial_are_refused():
    factors = [Factor(name=name, low=-1, high=1) for name in "AB"]
    with pytest.raises(ValueError, match="block_generators confound interaction words"):
        generate_design(factors, design_type="ccd", block_generators=["AB"])


def test_a_main_effect_through_an_alias_is_refused():
    """In the half fraction with C = AB, the word ABCD is D's alias, so it would confound D."""
    with pytest.raises(ValueError, match="confound the main effect D"):
        factorial("ABCD", generators=["C=AB"], block_generators=["ABCD"])


@pytest.mark.parametrize(("first", "second"), list(itertools.permutations(["ADE", "BCE"])))
def test_the_order_of_the_words_only_relabels_the_blocks(first, second):
    one = factorial("ABCDE", block_generators=[first, second], random_state=0)
    other = factorial("ABCDE", block_generators=["ADE", "BCE"], random_state=0)
    assert sorted(one.metadata["blocking"]["confounded_with"]) == sorted(other.metadata["blocking"]["confounded_with"])
