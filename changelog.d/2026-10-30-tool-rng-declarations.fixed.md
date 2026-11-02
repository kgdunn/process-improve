- **`recommend_next_step`, `sensory_analyze_descriptive` and `create_simulator` declare their seed.**
  Each draws random numbers, but its tool spec had no `rng` entry, so a host that reads `rng` to
  reproduce a call took it for deterministic. `recommend_next_step` and `sensory_analyze_descriptive`
  now declare `random_state`, with its defaults 42 and 0. `create_simulator` declares its
  Python-only `seed` with no default seed: with no seed it draws a new one, so the hidden model
  cannot be guessed. No output changes. (#1066)
