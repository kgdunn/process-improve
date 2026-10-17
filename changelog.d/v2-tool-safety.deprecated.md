- **The module-level `DEFAULT_TIMEOUT_S`, `DEFAULT_MAX_CELLS`, `DEFAULT_MAX_STRING`,
  `DEFAULT_MAX_DEPTH` and `DEFAULT_MEMORY_MB` names in `process_improve.tool_safety` are deprecated
  since 2.0.0 and will be removed in 3.0; use `settings.tool_timeout`, `settings.max_cells`,
  `settings.max_string`, `settings.max_depth` and `settings.max_memory_mb` from
  `process_improve.config`.** A read still returns the current settings value, and now issues a
  `DeprecationWarning`. The names were marked for removal in 2.0 but never warned, so the
  deprecation cycle starts with this release.
