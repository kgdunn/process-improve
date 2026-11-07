- **`process-improve-mcp` without the `mcp` extra says which extra to install.** It stopped at
  `No module named 'mcp'`; it now raises the `ImportError` that every other optional surface raises, which
  names `pip install 'process-improve[mcp]'`. (#1100)
