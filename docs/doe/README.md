# DOE Module - Architecture & Requirements

Developer reference for the `process_improve.experiments` module.
These docs capture the tool architecture and question bank that drive development.

## At a Glance

- **162 questions** across 16 categories the module must answer
- **Dominant workflow:** screening → optimization → confirmation

## Files

| File | What's in it |
|---|---|
| [questions.md](questions.md) | All 162 questions organized by category (A–P) |
| [tool-question-mapping.md](tool-question-mapping.md) | Which tool(s) answer which question |
| [workflows.md](workflows.md) | Common workflow patterns, design type usage, multi-tool chains |
| [../user_guide/doe_coverage.rst](../user_guide/doe_coverage.rst) | Design families and the conformance test that checks each one |

## Related

- Source: [`process_improve/experiments/`](../../process_improve/experiments/)
- API docs: [`docs/api/experiments.rst`](../api/experiments.rst)
- Tool specs: [`process_improve/experiments/tools.py`](../../process_improve/experiments/tools.py)
