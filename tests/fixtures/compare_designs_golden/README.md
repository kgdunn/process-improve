# compare_designs golden output, without a region

Equivalence fixture for `compare_designs`. It was recorded from the implementation that
scored designs over the cube only, immediately before the `region` argument was added.
`tests/test_compare_designs.py` checks that a comparison made without a region still
returns exactly what it did.

## Files

| File | Role |
| --- | --- |
| `prepare_fixture.py` | Defines the cases, runs them on the installed `process_improve`, and writes `golden.json`. Not run in CI. |
| `golden.json` | One case per line: the designs as exact coded runs, the model, the keyword settings, and the comparison's `to_dict()`. `generated_from` records the tree it ran on (`git describe --dirty`) and the library versions. |

## What is pinned

Every entry of `to_dict()`: the model, terms, settings, labels, the measures' definitions,
every value (counts, flags, the five criteria, max |r| and the three powers), every note,
`g_points` and the book link. The correlation maps are pinned for the three-factor cases;
for six and eleven factors they are large, and `max_r` already reads them.

The cases cover the book's twelve-point design with one, three and five centre runs; four
three-factor designs under the full quadratic model at a non-default effect size and level,
one with runs outside the cube and one that cannot fit the model; a saturated screening
design; and six and eleven factors, where G is searched over a 5-level grid and over the
corners plus a fixed random sample.

The test compares numbers to `rtol = atol = 1e-12` and everything else exactly. On the
platform that recorded the fixture (see `generated_from`) the result is bit-identical; the
tolerance allows for a last-bit difference in another platform's linear algebra.

## Regenerating

Only regenerate when a change to the numbers without a region is intended (a bug fix,
say); the point of the fixture is that adding features must not move them. From the
repository root:

```bash
uv run python tests/fixtures/compare_designs_golden/prepare_fixture.py
```
