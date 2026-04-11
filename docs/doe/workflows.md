# DOE Workflows & Patterns

## Dominant Workflow

Most DOE projects follow this sequence:

```
Risk assessment (Ishikawa / FMEA)
    ↓
Screening DOE (PB or fractional factorial, 8–20 runs)
    ↓
Identify 2–4 significant factors
    ↓
RSM optimization (BBD or CCD, 15–30 runs)
    ↓
Desirability function (multi-response)
    ↓
Confirmation runs (3–5 replicates at optimum)
    ↓
Design space mapping / control strategy
```

## Design Type Usage

From literature analysis across pharma, fermentation, food science, and bioprocess domains.

| Design | Usage Rank | Typical Stage |
|---|---|---|
| Box-Behnken (BBD) | #1 in RSM | Optimization (3–4 factors) |
| Central Composite (CCD) | #2 in RSM | Optimization (3–5 factors) |
| Plackett-Burman (PB) | #1 in screening | Screening (6–15 factors) |
| Full factorial 2^k | Common | Small-scale (2–4 factors) |
| D-optimal / I-optimal | Growing | Constrained / irregular regions |
| Mixture designs | Niche | Formulation / composition |
| Definitive Screening (DSD) | Emerging | Screening + some quadratic (5–12 factors) |
| Taguchi / Orthogonal Array | Legacy | Screening (declining use) |
| Doehlert | Rare | RSM (niche, European literature) |

## Multi-Tool Chains

Common sequences of tool calls for typical tasks.

### Screening Experiment

```
doe_knowledge  →  recommend_strategy  →  generate_design  →  evaluate_design
```

Example: "Design a screening experiment for my 7 fermentation factors"

1. `doe_knowledge` — retrieve design selection guidance for 7 factors
2. `recommend_strategy` — propose staged plan (screening → RSM)
3. `generate_design` — create PB design in 12 runs
4. `evaluate_design` — check alias structure and resolution

### Factorial Analysis + Optimization

```
analyze_experiment  →  visualize_doe  →  optimize_responses  →  visualize_doe
```

Example: "Analyze my 2^4 factorial data and find the optimum"

1. `analyze_experiment` — ANOVA, effects, residual diagnostics
2. `visualize_doe` — Pareto plot and normal probability plot
3. `optimize_responses` — find optimal factor settings
4. `visualize_doe` — contour plot at optimum

### Troubleshooting

```
doe_knowledge  →  analyze_experiment
```

Example: "My confirmation runs don't match — what went wrong?"

1. `doe_knowledge` — retrieve troubleshooting guidance
2. `analyze_experiment` — run confirmation test analysis with observed vs predicted

## Agent Orchestration

The LLM agent routes questions through tools in this general order:

1. **Classify** user intent from the question
2. **Retrieve knowledge** via `doe_knowledge` for conceptual grounding
3. **Compute** via `generate_design`, `analyze_experiment`, etc.
4. **Visualize** via `visualize_doe` to produce charts
5. **Synthesize** outputs into a coherent answer
