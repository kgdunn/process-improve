- **Concurrent safe tool calls no longer deadlock on free-threaded Python.** The worker pool of
  `safe_execute_tool_call` forked its workers from whichever thread asked, and on a free-threaded
  build a lock another thread held at that moment stayed held in the child, so concurrent calls
  timed out. On a free-threaded build the pool now starts its workers with `forkserver`, which
  forks from a single-threaded server process; other builds keep `fork`.
