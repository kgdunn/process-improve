- **The browser app (factori.al) keeps a factor's possible range when a design is picked, and makes the design
  with it** (#1322). Picking a design from the comparison, as the page does itself when its comparison ends, reset
  the possible ranges and the hard-to-change ticks typed meanwhile. A possible range also did not count as a change
  to the plan, so making the run sheet could reuse a design the comparison had made without it: its runs could go
  past the range, and the study did not keep it. Both are kept now, and a typed range makes the design again.
