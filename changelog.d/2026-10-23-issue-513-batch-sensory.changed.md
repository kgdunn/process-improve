- **`find_elbow_point`, and `f_elbow` with it, scale both axes to span 0 to 1 before looking for the
  elbow (#513).** The elbow was the point nearest the lines' intersection by a distance that added
  x units to y units, so it moved when the units did: on one test curve it was at index 24 with time
  in hours, 26 in seconds, and 38 with y multiplied by 1000. It is now 24 for all three. The elbow
  you get can move by a few points wherever the two axes span very different ranges.
- **`panel_scorecard` measures each panelist's `agreement` against the other panelists only (#513).**
  The consensus was the mean of every panelist, the one being scored included, and a mean that holds
  a panelist's own scores correlates with them by construction. This changes the numbers you get:
  `agreement` falls, by most on a small panel. With two panelists both now get
  the correlation between them (0.665 in the new test, where they got 0.928 and 0.896), and a panel
  of one gets NaN instead of 1.0. Because the flags compare `agreement` with fixed levels (0.5, and
  0 for anti-correlation), a panelist can now be flagged, and dropped by `drop_panelists="auto"` in
  `analyze_descriptive`, where it was not before. The same applies to the `sensory_panel_check` and
  `sensory_analyze_descriptive` tools.
- **`mixed_assessor_model` gives NaN F-tests, with a `RuntimeWarning`, for an attribute where a
  panelist did not score every product (#513).** Its sums of squares add up only on a complete
  panelist-by-product table. With a missing cell the four F-test columns were sometimes NaN and
  sometimes a finite number on the wrong degrees of freedom, with no warning; a finite value you got
  before for such an attribute is now NaN. The warning names the attributes and points to
  `mixed_assessor_model_reml`, which handles missing cells. The scaling coefficients are unchanged.
