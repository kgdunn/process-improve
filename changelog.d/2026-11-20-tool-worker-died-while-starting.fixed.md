- **A hosted tool call whose worker dies while it starts says so** (#1130). It was reported as "worker died (likely
  exceeded memory limit)", but the memory limit applies only once the worker has started. It now raises
  `ToolSafetyError` saying the worker died while starting, and names a likely cause: a main script started with
  `spawn` (macOS, Windows) without an `if __name__ == "__main__":` guard.
