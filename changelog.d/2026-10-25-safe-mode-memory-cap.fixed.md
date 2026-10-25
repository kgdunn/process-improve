- **Safe mode's memory cap no longer counts the imports against a tool call.** `max_memory_mb`
  (`PROCESS_IMPROVE_MAX_MEMORY_MB`) is now the address space a worker may map beyond what it mapped on importing
  the package, as the worker's own documentation said. It was an absolute cap on `RLIMIT_AS`, which counts address
  space reserved rather than memory in use: on the free-threaded build the imports alone reserve about 1.5 GiB, so
  every tool that allocated failed with "out of memory", even on a 7 kB CSV. On the build with the GIL the imports
  reserve about 0.8 GiB, so the total a worker may reserve rises from 1 GiB to about 1.8 GiB at the default.
