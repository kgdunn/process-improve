- **`find_elbow_point`, and `f_elbow` with it, scale both axes to span 0 to 1 before looking for the
  elbow (#513).** The elbow was the point nearest the lines' intersection by a distance that added
  x units to y units, so it moved when the units did: on one test curve it was at index 24 with time
  in hours, 26 in seconds, and 38 with y multiplied by 1000. It is now 24 for all three. The elbow
  you get can move by a few points wherever the two axes span very different ranges.
