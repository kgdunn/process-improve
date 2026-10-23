"""Studies of more than 12 factors: the screening designs, runs as given, and what the page keeps to past 12.

The page takes up to 40 factors (``MAX_FACTORS``) in runs as given and the screening designs whose
construction is quick at any size, and a fraction up to 24. Past 12 factors
(``MAX_INTERACTION_FACTORS``) it models no interactions: Learn fits the main effects, or the terms
the reader names, and the parts that grow fastest with the factors keep their own lower caps.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from tests.web._support import coded, held_of, run_sheet

#: The numbers of factors the page is tested at past its old cap of 12: one more, a cap the issue weighed, the most.
MANY = (13, 24, 40)


def named(k: int, **more: object) -> list[dict]:
    """Return ``k`` numeric factors, X1 to Xk, each from 0 to 10, or as ``more`` says."""
    return [{"name": f"X{i + 1}", "low": 0, "high": 10, "units": "u", **more} for i in range(k)]


def screening(api, k: int, design_type: str = "plackett_burman", **more: object) -> dict:
    """Make a screening design in ``k`` factors, as the page holds it."""
    return held_of(api.ok("api_make_design", {"design_type": design_type, "factors": named(k), **more}))


def results(held: dict, effects: dict[str, float], seed: int = 0) -> list[float]:
    """Return a result for each run: 50 plus each named factor's effect times its coded setting, and a little noise."""
    x = coded(run_sheet(held), held["spec"]["factors"])
    y = 50 + sum(effect * x[name] for name, effect in effects.items())
    return (y + np.random.default_rng(seed).normal(0, 0.1, len(x))).round(4).tolist()


def saturated(base: int) -> list[list[int]]:
    """Return the regular fraction of 2^base runs in every product of its base factors: 2^(k-p) with k = 2^base - 1."""
    runs = list(itertools.product([-1, 1], repeat=base))
    subsets = [s for size in range(1, base + 1) for s in itertools.combinations(range(base), size)]
    return [[math.prod(run[j] for j in s) for s in subsets] for run in runs]


# --------------------------------------------------------------------------- the designs
@pytest.mark.parametrize("k", MANY)
@pytest.mark.parametrize("design_type", ["plackett_burman", "taguchi", "dsd"])
def test_a_screening_design_of_up_to_40_factors_is_made_for_its_main_effects(api, design_type, k):
    held = screening(api, k, design_type)
    assert [f["name"] for f in held["spec"]["factors"]] == [f"X{i + 1}" for i in range(k)]
    assert held["spec"]["model"] == "main_effects"
    assert {f"X{i + 1}" for i in range(k)} <= set(held["columns"])


@pytest.mark.parametrize(("k", "runs"), [(13, [16, 32]), (24, [32, 64])])
def test_a_fraction_of_up_to_24_factors_is_made_at_resolution_iii_or_iv(api, k, runs):
    made = [
        api.ok("api_make_design", {"design_type": "fractional_factorial", "factors": named(k), "resolution": r})
        for r in (3, 4)
    ]
    assert [m["n_runs"] - m["facts"]["center_runs"] for m in made] == runs
    assert [m["facts"]["resolution"] for m in made] == [3, 4]
    assert all(m["model"] == "main_effects" for m in made)


def test_a_taguchi_array_of_40_factors_at_three_levels_is_the_l81(api):
    made = api.ok("api_make_design", {"design_type": "taguchi", "factors": named(40), "n_levels": 3})
    assert made["n_runs"] == 81
    assert made["facts"]["notes"][0].startswith("Taguchi L81 (3^40): 81 runs")


# --------------------------------------------------------------------------- runs as given
@pytest.mark.parametrize("k", MANY)
def test_runs_as_given_of_up_to_40_factors_are_analysed_for_their_main_effects(api, k):
    """However many runs there are, past 12 factors the design's model is its main effects."""
    full = held_of(api.ok("api_make_design", {"design_type": "plackett_burman", "factors": named(k)}))
    rows = run_sheet(full)[[f"X{i + 1}" for i in range(k)]].to_numpy().tolist()
    given = api.ok("api_make_design", {"design_type": "given", "factors": named(k), "runs": rows + rows})
    assert given["model"] == "main_effects"
    held = held_of(given)
    analysis = api.ok("api_analyze_runs", {**held, "responses": results(held, {"X1": 3, "X2": -2})})
    assert analysis["model"] == "main_effects"
    assert len(analysis["coefficients"]) == k + 1


def test_a_saturated_fraction_given_names_its_shortest_words_and_counts_the_rest(api):
    """2^(15-11) has 2047 words in its defining relation: too many to list, so the note names the shortest."""
    letters = [f"F{j + 1}" for j in range(15)]
    factors = [{"name": n, "low": -1, "high": 1} for n in letters]
    made = api.ok("api_make_design", {"design_type": "given", "factors": factors, "runs": saturated(4)})
    assert made["facts"]["resolution"] == 3
    assert made["facts"]["defining_relation"] == [], (
        "the library lists a relation of up to 1023 words, and so does this"
    )
    note = next(n for n in made["facts"]["notes"] if n.startswith("Resolution III"))
    assert note.endswith(", and 2040 more words.")
    assert note.startswith("Resolution III: the runs make the defining relation I = F1·F2·F5 = ")


def test_a_fraction_of_up_to_1023_words_given_lists_its_whole_relation(api):
    """2^(13-9), the 16 runs of the saturated fraction in its first 13 columns, has 511 words, every one listed."""
    factors = [{"name": f"F{j + 1}", "low": -1, "high": 1} for j in range(13)]
    runs = [row[:13] for row in saturated(4)]
    made = api.ok("api_make_design", {"design_type": "given", "factors": factors, "runs": runs})
    assert len(made["facts"]["defining_relation"]) == 511
    assert made["facts"]["resolution"] == 3


def _every_word(coded_runs: np.ndarray, names: list[str]) -> list[str]:
    """Return the defining relation of runs by trying every set of factors, as the page read it before."""
    distinct = np.unique(np.sign(coded_runs), axis=0)
    words = []
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(range(len(names)), size):
            product = np.prod(distinct[:, list(subset)], axis=1)
            if np.all(product == product[0]):
                words.append(("-" if product[0] < 0 else "") + "·".join(names[j] for j in subset))
    return words


@pytest.mark.parametrize(
    "columns",
    [
        [0, 1, 2, 3],  # 2^(4-1) in a half of the 2^3, D = ABC
        [0, 1, 2, 3, 4, 5, 6],  # 2^(7-4), saturated
        [0, 1, 3, 6, 2, 5],  # 2^(6-3), its columns in another order
        [0, 1, 2, 6, 3, 4, 5, 9, 10],  # 2^(9-5) in 16 runs
    ],
    ids=["half fraction", "saturated", "reordered", "nine in sixteen"],
)
@pytest.mark.parametrize("flip", [False, True], ids=["as made", "a column negated"])
def test_the_relation_read_by_elimination_is_the_one_every_set_of_factors_gives(bootstrap, columns, flip):
    base = 3 if max(columns) < 7 else 4
    runs = np.array(saturated(base))[:, columns].astype(float)
    if flip:
        runs[:, -1] *= -1
    names = [f"F{j + 1}" for j in range(len(columns))]
    words, count = bootstrap._relation_of_runs(runs, names)
    assert words == _every_word(runs, names)
    assert count == len(words)


def test_runs_that_are_no_regular_fraction_have_no_relation(api, bootstrap):
    """A fraction made three times over is still one, of one word; a 12-run Plackett-Burman design is none."""
    repeated = np.array(saturated(2) * 3, dtype=float)
    assert bootstrap._relation_of_runs(repeated, ["A", "B", "C"]) == (["A·B·C"], 1)
    held = screening(api, 11)
    runs = coded(run_sheet(held), held["spec"]["factors"]).to_numpy()
    assert bootstrap._relation_of_runs(runs, [f["name"] for f in held["spec"]["factors"]]) == ([], 0)


# --------------------------------------------------------------------------- past 12 factors
@pytest.mark.parametrize("k", MANY)
def test_past_12_factors_learn_fits_main_effects_and_the_interactions_named(api, k):
    held = screening(api, k)
    responses = results(held, {"X1": 3, "X2": -2, "X3": 1})
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    assert analysis["model"] == "main_effects"
    terms = [["X1"], ["X2"], ["X1", "X2"]]  # two factors that stand out, and their interaction
    chosen = api.ok("api_analyze_runs", {**held, "responses": responses, "model": {"terms": terms}})
    assert [c["term"] for c in chosen["coefficients"]] == ["Intercept", "X1", "X2", "X1:X2"]


@pytest.mark.parametrize("k", MANY)
def test_past_12_factors_the_designs_are_compared_on_their_runs_spare_runs_and_power(api, k):
    listed = api.ok("api_candidates", {"factors": named(k), "model": "main_effects"})
    assert listed["result"]["measures"] == []
    assert listed["result"]["fds"] == {}
    labels = {m["label"] for m in listed["made"]}
    assert {"Plackett-Burman", "Taguchi", "DSD"} <= labels
    assert ("Fractional factorial" in labels) == (k <= 24)
    power = [v["power_main"] for v in listed["result"]["values"].values()]
    assert all(p is None or 0 <= p <= 1 for p in power)
    corners = f"{k} factors have {2**k:,} of them."
    assert any(note.endswith(corners) for note in listed["notes"]), listed["notes"]


def test_12_factors_are_still_compared_on_every_measure(api):
    listed = api.ok("api_candidates", {"factors": named(12), "model": "main_effects"})
    assert listed["result"]["measures"]
    assert not any(note.startswith("Past 12 factors") for note in listed["notes"])


@pytest.mark.slow
@pytest.mark.parametrize("k", [13, 24])
def test_past_12_factors_the_next_step_reads_the_main_effects(api, k):
    held = screening(api, k)
    spec = {**held["spec"], "response_info": {"goal": "maximize"}}
    reply = api.ok("api_next_runs", {**held, "spec": spec, "responses": results(held, {"X1": 3, "X2": -2})})
    assert reply["waiting"] is None
    assert reply["recommendation"]["judged_by"]
    assert set(reply["recommendation"]["active"]) >= {"X1", "X2"}
    assert all(":" not in term for term in reply["recommendation"]["active"]), "no interaction is judged"


def test_past_24_factors_the_next_step_is_not_worked_out_and_says_why(api):
    held = screening(api, 40)
    spec = {**held["spec"], "response_info": {"goal": "maximize"}}
    reply = api.ok("api_next_runs", {**held, "spec": spec, "responses": results(held, {"X1": 3})})
    assert reply["recommendation"] is None
    assert reply["waiting"].startswith("Past 24 factors, the page does not work out what to run next:")
    assert "with 40 factors that takes minutes here" in reply["waiting"]


@pytest.mark.slow
def test_the_best_settings_of_40_factors_put_each_where_its_effect_points(api):
    held = screening(api, 40)
    effects = {"X1": 3.0, "X2": -2.0, "X3": 1.5}
    spec = {**held["spec"], "response_info": {"goal": "maximize"}}
    best = api.ok("api_best_settings", {**held, "spec": spec, "responses": results(held, effects)})
    assert [best["coded"][n] for n in effects] == [1.0, -1.0, 1.0]


@pytest.mark.parametrize("k", [13, 40])
def test_the_levels_of_many_categorical_factors_are_each_chosen_at_its_best(api, k):
    """Past 4096 combinations of levels (13 factors of two have 8192), they are chosen a factor at a time."""
    factors = [{"name": f"S{i + 1}", "type": "categorical", "levels": ["a", "b"], "units": ""} for i in range(k)]
    held = held_of(api.ok("api_make_design", {"design_type": "plackett_burman", "factors": factors}))
    sheet = run_sheet(held)
    effects = {"S1": 3.0, "S2": -2.0, "S3": 1.5}
    y = 50 + sum(e * np.where(sheet[n] == "b", 1.0, -1.0) for n, e in effects.items())
    spec = {**held["spec"], "response_info": {"goal": "maximize"}}
    best = api.ok("api_best_settings", {**held, "spec": spec, "responses": y.round(4).tolist()})
    levels = {s["name"]: s["value"] for s in best["settings"]}
    assert [levels[n] for n in effects] == ["b", "a", "b"]
