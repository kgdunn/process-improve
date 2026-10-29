- **`ControlChart.calculate_limits` keeps a Holt-Winters weight pinned on its own, and searches only the other.**
  Pinning one of `ld_1` and `ld_2` was ignored: both were searched over the grid, the pinned value was replaced,
  and an out-of-range value such as `ld_2=5.0` was accepted without a word. It is now kept, checked to lie in
  0 to 1, and, when the data cannot choose the other weight, kept beside its default. Results change only for a
  call that pins exactly one of the two. Found by mutation testing.
