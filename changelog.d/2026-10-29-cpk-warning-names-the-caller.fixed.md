- **The warning `calculate_cpk` gives for a side with no spread now names the line that called it.** It named a
  line inside `calculate_cpk` instead, because the helper that raises it sits one frame deeper than its
  `stacklevel` allowed for. Found by mutation testing.
