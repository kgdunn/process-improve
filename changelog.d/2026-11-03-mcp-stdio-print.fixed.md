- **A tool's `print` no longer reaches the stdio MCP server's protocol stream.** While it serves,
  the MCP SDK points file descriptor 1 at stderr, so output written there from C (osqp's "Polishing
  not needed ...", HiGHS) goes to stderr. A Python `print` stays in `sys.stdout`'s buffer when stdout
  is a pipe, and was written when the server exited, after the SDK had pointed fd 1 back at the
  protocol stream; in safe mode it was lost with the worker. `execute_tool_call` now flushes
  `sys.stdout` as each call ends, so the print goes to stderr. (#1095)
