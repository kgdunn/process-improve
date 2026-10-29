- **A one-component mixture region with a constraint no longer crashes.** `DesignRegion` with a single
  mixture component and any constraint raised `ValueError: cannot reshape array of size 0` while
  enumerating its vertices. Its only blend is the pure component, so its share of the blends is now 1.0
  when the constraints allow that blend and 0.0 when they do not (#999).
