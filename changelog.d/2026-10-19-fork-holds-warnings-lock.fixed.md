- **Concurrent `safe_execute_tool_call` calls no longer hang on Python 3.14.** From 3.14 every
  warning takes one lock for the whole process. A worker forked while another thread held that lock
  inherited it held, waited forever at its first warning, and the call ended in `ToolTimeoutError`.
  Since tool calls record their warnings (`library_warnings`), several calls at once on the
  free-threaded build often failed this way. The fork of a worker now holds the lock, so no other
  thread holds it at that instant.
