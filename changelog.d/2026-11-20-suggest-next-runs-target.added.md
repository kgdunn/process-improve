- **`suggest_next_runs` and its tool take a target as the goal** (#1374): `goal="target", target=1200` proposes the
  runs that are expected to bring the response closest to 1200. The expected improvement is in the distance from
  the target, worked exactly; `best` is the result closest to the target so far, and `improvement_ratio` is over
  its distance. A batch is chosen as before, each run taken to give its prediction.
