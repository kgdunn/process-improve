"""Three-level fractional factorials made from generators in arithmetic modulo 3 (#788 A5).

Montgomery (2013, section 9.3) builds his 3^(3-1), 3^(4-1) and 3^(4-2) fractions from generators, and
the reference catalogue holds his runs. The L9 of NIST and of Schmidt and Launsby, Schmidt and
Launsby's Latin square and the fraction of Montgomery's Problem 12.10 are the same construction. Each
is rebuilt here from its generators, and the relation and aliases are checked by brute force.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments import Factor, evaluate_design, generate_design
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments

CODED = (-1, 0, 1)


def catalogue(name: str) -> dict:
    [entry] = [e for e in reference_designs() + reference_experiments() if e["name"] == name]
    return entry


def fraction(names, generators, levels=CODED, **options):
    factors = [Factor(name=n, low=min(levels), high=max(levels), levels=list(levels)) for n in names]
    options.setdefault("n_center_points", 0)
    return generate_design(factors, design_type="fractional_factorial", generators=generators, **options)


def digits(design, names) -> np.ndarray:
    """Return each run's levels as the digits 0, 1 and 2: the coded -1, 0 and +1."""
    return design.design[list(names)].to_numpy(dtype=float).astype(int) + 1


def printed(entry: dict, names, levels, keep=None) -> set[tuple[int, ...]]:
    """Return the source's runs as digits, each value's place in *levels*."""
    columns = entry["runs"]["columns"]
    rows = [row for row in entry["runs"]["rows"] if keep is None or keep(dict(zip(columns, row, strict=True)))]
    return {tuple(list(levels).index(row[columns.index(n)]) for n in names) for row in rows}


def component(word, names) -> str:
    return "".join(n + ("2" if e == 2 else "") for n, e in zip(names, word, strict=True) if e)


def up_to_the_last_derived(generators: list[str]) -> str:
    """Return the one-letter factors A, B, C, ... up to the last one a generator derives."""
    return "ABCDEF"[: max("ABCDEF".index(g[0]) for g in generators) + 1]


def all_components(k: int) -> list[tuple[int, ...]]:
    """Every component of k three-level factors, written with its first exponent 1."""
    return [w for w in itertools.product(range(3), repeat=k) if any(w) and w[np.flatnonzero(w)[0]] == 1]


# --------------------------------------------------------------------------- Montgomery, section 9.3
@pytest.mark.dataset
def test_montgomery_s_one_third_fraction_from_its_generator():
    """C = AB2 is x3 = x1 + 2x2, the fraction u = 0 of I = AB2C2, of resolution III (Figure 9.9)."""
    design = fraction("ABC", ["C=AB2"], levels=(0, 1, 2))
    entry = catalogue("montgomery-3-3-1-one-third-fractions")
    principal = printed(entry, "ABC", (0, 1, 2), keep=lambda run: str(run["u"]) == "0")
    assert set(map(tuple, digits(design, "ABC").tolist())) == principal
    assert (design.n_runs, design.defining_relation, design.resolution) == (9, ["I=AB2C2"], 3)
    # A = ABC = BC, B = AC2 = ABC2, C = AB2 = AB2C, the alias structure Montgomery prints.
    assert design.metadata["aliases"] == {"A": ["BC", "ABC"], "B": ["AC2", "ABC2"], "C": ["AB2", "AB2C"]}


@pytest.mark.dataset
def test_montgomery_s_resolution_iv_fraction_from_its_generator():
    """x4 = 2x1 + x2 + 2x3 gives I = AB2CD (Table 9.6), and A is aliased with ABC2D2 and BC2D2."""
    design = fraction("ABCD", ["D=A2BC2"], levels=(0, 1, 2))
    entry = catalogue("montgomery-3-4-1-resolution-iv")
    assert set(map(tuple, digits(design, "ABCD").tolist())) == printed(entry, "ABCD", (0, 1, 2))
    assert (design.n_runs, design.defining_relation, design.resolution) == (27, ["I=AB2CD"], 4)
    assert design.metadata["aliases"]["A"] == ["BC2D2", "ABC2D2"]


@pytest.mark.dataset
def test_montgomery_s_resolution_iii_fraction_and_its_alias_table():
    """x3 = 2x1 + x2 and x4 = 2x2 + 2x3, that is x1 + x2: Table 9.7's runs and Table 9.8's aliases."""
    design = fraction("ABCD", ["C=A2B", "D=AB"], levels=(0, 1, 2))
    entry = catalogue("montgomery-3-4-2-resolution-iii")
    assert set(map(tuple, digits(design, "ABCD").tolist())) == printed(entry, "ABCD", (0, 1, 2))
    assert set(design.defining_relation) == {"I=AB2C", "I=BCD", "I=AC2D", "I=ABD2"}
    assert design.resolution == 3
    table = {}
    for expected in entry["expected"]:
        if expected["what"] == "aliases":
            table.setdefault(expected["term"][0], set()).add(expected["value"].replace("²", "2"))
    assert {effect: set(chain) for effect, chain in design.metadata["aliases"].items()} == table


@pytest.mark.dataset
def test_montgomery_problem_12_10_fraction_with_longer_names():
    """Names longer than one letter are joined with '*': x3 = x1 + 2 x2 and x4 = x1 + x2."""
    names = ["x1", "x2", "x3", "x4"]
    design = fraction(names, ["x3=x1*x2^2", "x4=x1*x2"])
    entry = catalogue("montgomery-problem-12-10-b")
    assert set(map(tuple, digits(design, names).tolist())) == printed(entry, names, CODED)
    assert set(design.defining_relation) == {"I=x1*x2^2*x3^2", "I=x1*x2*x4^2", "I=x1*x3*x4", "I=x2*x3^2*x4"}


# --------------------------------------------------------------------------- NIST, Schmidt and Launsby
@pytest.mark.dataset
def test_the_l9_of_nist_and_of_schmidt_and_launsby_is_one_fraction():
    """Both L9 tables are the 3^(4-2) with C = AB and D = A2B, of resolution III."""
    nist = fraction(["X1", "X2", "X3", "X4"], ["X3=X1*X2", "X4=X1²*X2"], levels=(1, 2, 3))
    nist_runs = printed(catalogue("nist-5-3-3-10-taguchi-l9"), ["X1", "X2", "X3", "X4"], (1, 2, 3))
    assert set(map(tuple, digits(nist, ["X1", "X2", "X3", "X4"]).tolist())) == nist_runs
    schmidt = fraction("ABCD", ["C=AB", "D=A^2B"])
    assert set(map(tuple, digits(schmidt, "ABCD").tolist())) == printed(
        catalogue("schmidt-table-3-5-l9"), "ABCD", CODED
    )
    assert schmidt.defining_relation == ["I=ABC2", "I=AB2D", "I=ACD2", "I=BCD"]
    assert (nist.resolution, schmidt.resolution) == (3, 3)


@pytest.mark.dataset
def test_schmidt_and_launsby_s_latin_square_derives_its_first_factor():
    """B and C cross, and A, the letter of the square, is B + C (mod 3): a derived factor need not be last."""
    design = fraction("ABC", ["A=BC"])
    expected = printed(catalogue("schmidt-table-3-19-latin-square"), "ABC", CODED)
    assert set(map(tuple, digits(design, "ABC").tolist())) == expected


# --------------------------------------------------------------------------- the algebra, by brute force
@pytest.mark.parametrize(
    "generators",
    [["C=AB2"], ["D=A2BC2"], ["C=A2B", "D=AB"], ["E=ABCD"], ["D=AB", "E=AC2"], ["D=ABC", "E=AB2", "F=BC2"]],
)
def test_the_relation_is_every_component_the_runs_hold_constant(generators):
    """A component is in the defining relation exactly when its contrast is 0 on every run."""
    names = up_to_the_last_derived(generators)
    design = fraction(names, generators)
    runs = digits(design, names)
    constant = [component(w, names) for w in all_components(len(names)) if not np.any(runs @ np.array(w) % 3)]
    assert sorted(design.defining_relation) == sorted("I=" + w for w in constant)
    assert len(constant) == (3 ** len(generators) - 1) // 2


@pytest.mark.parametrize("generators", [["D=A2BC2"], ["C=A2B", "D=AB"], ["D=AB", "E=AC2"]])
def test_an_effect_s_aliases_split_the_runs_as_the_effect_does(generators):
    """Two components are aliased exactly when their contrasts split the runs into the same three groups."""
    names = up_to_the_last_derived(generators)
    design = fraction(names, generators)
    runs = digits(design, names)

    def groups(word):
        values = runs @ np.array(word) % 3
        return {frozenset(np.flatnonzero(values == v)) for v in range(3)} - {frozenset()}

    for j, name in enumerate(names):
        effect = groups(np.eye(len(names), dtype=int)[j])
        same = {component(w, names) for w in all_components(len(names)) if groups(w) == effect} - {name}
        assert set(design.metadata["aliases"][name]) == same


def test_categorical_factors_take_their_own_levels():
    """The digits 0, 1 and 2 are each factor's levels in the order given."""
    factors = [Factor(name=n, type="categorical", levels=["lo", "mid", "hi"]) for n in "ABC"]
    design = generate_design(factors, design_type="fractional_factorial", generators=["C=AB2"], n_center_points=0)
    runs = design.design[list("ABC")].replace({"lo": 0, "mid": 1, "hi": 2}).to_numpy(dtype=int)
    assert design.n_runs == 9
    assert len({tuple(run) for run in runs.tolist()}) == 9
    assert np.all(runs[:, 2] == (runs[:, 0] + 2 * runs[:, 1]) % 3)


def test_a_saturated_27_run_fraction_reports_its_resolution_but_not_its_relation():
    """Thirteen factors in 27 runs (ten generators) is an orthogonal array; its 29,524 components are not listed."""
    words = ["AB", "AB2", "AC", "AC2", "BC", "BC2", "ABC", "ABC2", "AB2C", "AB2C2"]
    names = "ABCDEFGHJKLMN"
    design = fraction(names, [f"{name}={word}" for name, word in zip(names[3:], words, strict=True)])
    assert (design.n_runs, design.resolution, design.defining_relation) == (27, 3, None)
    assert "29524 components" in design.metadata["defining_relation_note"]
    runs = digits(design, names)
    for i, j in itertools.combinations(range(len(names)), 2):  # every pair of levels three times
        assert len({(a, b) for a, b in runs[:, [i, j]].tolist()}) == 9


def test_more_than_ten_generators_leave_the_resolution_unknown():
    words = ["AB", "AB2", "AC", "AC2", "BC", "BC2", "ABC", "ABC2", "AB2C", "AB2C2", "AB"]
    names = "ABCDEFGHJKLMN" + "P"
    design = fraction(names, [f"{name}={word}" for name, word in zip(names[3:], words, strict=True)])
    assert (design.n_runs, design.resolution) == (27, None)
    assert "too many to examine" in design.metadata["defining_relation_note"]


def test_the_metadata_names_the_arithmetic():
    design = fraction("ABC", ["C=AB2"])
    assert design.generators == ["C=AB2"]
    assert design.metadata["modulus"] == 3
    assert design.metadata["levels_per_factor"] == [3, 3, 3]


# --------------------------------------------------------------------------- evaluate_design reads it right
def test_evaluate_design_reads_the_recorded_relation_not_a_two_level_one():
    """Read modulo 2, D = A2BC2 would be I = ABCD and A = BCD; the design's own relation is used instead."""
    design = fraction("ABCD", ["D=A2BC2"])
    assert evaluate_design(design, metric="defining_relation")["defining_relation"] == ["I=AB2CD"]
    assert evaluate_design(design, metric="resolution")["resolution"] == 4
    assert "A = BCD" not in evaluate_design(design, metric="alias_structure")["alias_structure"]
    pattern = evaluate_design(design, metric="minimum_aberration")["minimum_aberration"]
    assert pattern["note"] == "Not a two-level fractional factorial design."


# --------------------------------------------------------------------------- what is unchanged, and refused
def test_without_generators_three_level_factors_still_give_the_two_level_fraction():
    """Only generators choose the three-level fraction; a resolution, or nothing, keeps the two-level one."""
    assert fraction("ABCD", None).n_runs == 8
    assert fraction("ABCD", None, resolution=4).n_runs == 8
    factors = [Factor(name=n, type="categorical", levels=["lo", "mid", "hi"]) for n in "ABCD"]
    with pytest.raises(ValueError, match="takes two-level categorical factors only"):
        generate_design(factors, design_type="fractional_factorial", n_center_points=0)


def test_a_generator_short_of_the_resolution_asked_for_is_refused():
    with pytest.raises(ValueError, match=r"give a resolution-3 design, short of the resolution=4"):
        fraction("ABC", ["C=AB2"], resolution=4)
    assert fraction("ABCD", ["D=A2BC2"], resolution=4).resolution == 4


@pytest.mark.parametrize(
    ("generators", "message"),
    [
        (["DAB"], "must have the form 'D=A2BC2'"),
        (["D2=AB"], "must have the form 'D=A2BC2'"),
        (["AD=B"], "must have the form 'D=A2BC2'"),
        (["D=A3B"], "cannot be read"),
        (["D=AE"], "names 'E', which is not a factor"),
        (["D="], "its right-hand side names no factor"),
        (["D=AB", "D=AC"], r"\['D'\] have more than one"),
        (["C=AB", "D=AC"], r"\['C'\] are derived factors themselves"),
    ],
)
def test_generators_that_cannot_make_a_fraction_are_refused(generators, message):
    with pytest.raises(ValueError, match=message):
        fraction("ABCD", generators)
