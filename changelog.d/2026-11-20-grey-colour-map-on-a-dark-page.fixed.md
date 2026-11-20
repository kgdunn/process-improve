- **The grey colour map starts at a dark page's own grey, not at black.** `colorscale("grayC", dark_page=True)` and the
  browser app's Grey map began at `#000000`, darker than the page (#141517), so the lowest values stood out from it
  instead of fading into it. The map is now sampled from 6% of the way along, `#131313`; its light end is unchanged.
