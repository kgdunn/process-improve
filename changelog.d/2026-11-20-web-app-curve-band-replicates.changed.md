- **A curve's 99% band in the browser app (factori.al) rests on 1,000 bootstrap replicates** (#1381), where it
  rested on 200: the simultaneous band is read from the 99% quantile of the replicates' largest departures, which
  200 replicates leave to about the second largest. The page now takes ten replicates past the quantile: 200 at a
  significance level of 0.05 or 0.10, as before, and 1,000 at 0.01, which takes about twice as long to refit.
