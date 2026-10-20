- **How to evaluate an agent that designs experiments.** A development page sets out the protocol
  and the first results: an LLM agent runs whole studies with the process-improve tools and without
  them, on processes whose truth is known. Its runs come from `scripts/doe_lab.py`, a blind
  laboratory on localhost that charges each run against a budget, keeps the true model to itself,
  and scores the agent's answer afterwards. In twelve sessions on three problems, the agents with
  the tools used 36 runs on average against 77 without them.
