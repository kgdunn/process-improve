- **`recommend_next_step` is many times faster on a large screening study.** Listing the alternatives, it chose
  the runs of the response-surface option by a D-optimal search only to count them; the search always adds the
  runs it is asked for, so the count now skips it. An 11-factor Plackett-Burman design with 3 centre runs took
  15 s, nearly all of it on that count, and now takes under 1 s, with the same alternatives.
- **`recommend_next_step(..., response=None)` says the runs have no response column `None`**, as for any other
  name not among the runs, not a `KeyError` for a column `'y'` the caller never named.
