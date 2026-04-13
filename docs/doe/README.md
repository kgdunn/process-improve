# DOE Module — Architecture & Requirements

Developer reference for the `process_improve.experiments` module.
These docs capture the tool architecture and question bank that drive development.

## At a Glance

- **8 tools** defined (6 implemented, 2 not started)
- **162 questions** across 16 categories the module must answer
- **Dominant workflow:** screening → optimization → confirmation

## Files

| File | What's in it |
|---|---|
| [tools.md](tools.md) | The 8-tool architecture — what each tool does, its inputs/outputs |
| [questions.md](questions.md) | All 162 questions organized by category (A–P) |
| [tool-question-mapping.md](tool-question-mapping.md) | Which tool(s) answer which question |
| [workflows.md](workflows.md) | Common workflow patterns, design type usage, multi-tool chains |
| [coverage.md](coverage.md) | Implementation status and gap analysis |

## Related

- Source: [`process_improve/experiments/`](../../process_improve/experiments/)
- API docs: [`docs/api/experiments.rst`](../api/experiments.rst)
- Tool specs: [`process_improve/experiments/tools.py`](../../process_improve/experiments/tools.py)
