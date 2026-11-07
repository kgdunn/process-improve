- **A hosted tool call's worker start-up no longer counts against the tool's timeout.** In safe
  mode each call starts a fresh worker, which under `spawn` (the start method on macOS and Windows)
  imports the package and its tools again: 2 to 3.5 s of the 10 s default timeout, measured on
  Linux, and more on a slow machine, before any tool code ran. `safe_execute_tool_call` now waits
  for the worker to start, with its own limit of 60 s, and only then starts the tool's timeout, so
  `PROCESS_IMPROVE_TOOL_TIMEOUT` measures the tool's own work. A worker that does not start in
  time is ended, and the call raises `ToolTimeoutError` with a message that says the tool did not
  run; its `details["stage"]` is `"startup"`, and `"tool"` for a tool that ran too long. The total
  time of a call does not change. (#1130)
