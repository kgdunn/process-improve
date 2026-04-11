# Tool → Question Mapping

Traceability matrix: which tool(s) answer each question.
Use this when implementing a tool to see exactly what it must handle.

---

## Tool-Centric View

### `generate_design`

46 questions as primary tool. Creates design matrices for factorial, fractional, RSM, optimal, mixture, and screening designs.

**Primary (46):** 1–3, 5, 9, 11–12, 15–18, 91–98, 105–107, 112, 115–116, 118–120, 124–127, 131–133, 136, 140–141, 144–146, 148

**Secondary (12):** 56, 59, 63–64, 66, 111, 113, 119, 135, 151, 159

### `evaluate_design`

7 questions as primary tool. Computes alias structure, confounding, resolution, power, efficiency metrics.

**Primary (7):** 22–23, 25, 29–30, 20, 16

**Secondary (8):** 5, 17, 28, 72, 77, 100, 130, 142, 156

### `analyze_experiment`

22 questions as primary tool. The analytical workhorse — ANOVA, effects, diagnostics, model fitting.

**Primary (22):** 31–43, 86, 101, 103, 108, 114, 121, 128–130, 147, 152–153, 157, 160–161

**Secondary (14):** 50, 60–61, 67, 70–71, 108, 114, 147, 152–153, 157, 160–161

### `optimize_responses`

15 questions as primary tool. Finds optimal factor settings via desirability, steepest ascent, ridge analysis.

**Primary (15):** 45–46, 49, 53, 99, 103, 108, 119, 121, 127, 129, 139, 155, 158

**Secondary (5):** 47–48, 64, 66, 103

### `augment_design`

7 questions as primary tool. Extends designs with foldover, axial points, or optimal augmentation.

**Primary (7):** 10, 19, 73, 75, 77, 100, 143, 156

**Secondary (2):** 97, 109

### `visualize_doe`

3 questions as primary tool. Generates all DOE plot types — also supports many questions as secondary.

**Primary (3):** 47, 76, 121

**Secondary (7):** 32–33, 36, 38, 68, 128

### `doe_knowledge`

63 questions as primary tool. The most-used tool — provides conceptual grounding for 65% of all questions.

**Primary (63):** 4, 6–8, 13–14, 21, 24, 26–28, 44, 48, 50–52, 54–55, 57–59, 62, 67–72, 74, 78–85, 87–90, 102, 109–111, 113–114, 122–123, 128, 130, 134–135, 137–138, 142, 145, 150–154, 157, 159, 161–162

**Secondary (42):** 1–3, 11–12, 15, 18, 20, 30, 34, 36, 38, 43, 53, 56, 59, 73, 75–77, 87–88, 97, 101, 109, 125, 131, 134, 143, 145, 147, 152–153, 156–158

### `recommend_strategy`

10 questions as primary tool. Multi-stage experimental strategy advisor.

**Primary (10):** 1, 63–66, 97, 104, 117, 131, 134, 149, 162

**Secondary (4):** 9, 91, 109, 145

---

## Reverse Lookup

Every question with its primary and secondary tools.

| Q# | Primary | Secondary |
|---|---|---|
| 1 | `recommend_strategy` | `doe_knowledge` |
| 2 | `generate_design` | `doe_knowledge` |
| 3 | `generate_design` | `doe_knowledge` |
| 4 | `doe_knowledge` | — |
| 5 | `generate_design` | `evaluate_design` |
| 6 | `doe_knowledge` | — |
| 7 | `doe_knowledge` | — |
| 8 | `doe_knowledge` | — |
| 9 | `generate_design` | `recommend_strategy` |
| 10 | `augment_design` | — |
| 11 | `generate_design` | `doe_knowledge` |
| 12 | `generate_design` | `doe_knowledge` |
| 13 | `doe_knowledge` | — |
| 14 | `doe_knowledge` | — |
| 15 | `generate_design` | `doe_knowledge` |
| 16 | `generate_design` | `evaluate_design` |
| 17 | `generate_design` | `evaluate_design` |
| 18 | `generate_design` | `doe_knowledge` |
| 19 | `augment_design` | — |
| 20 | `evaluate_design` | `doe_knowledge` |
| 21 | `doe_knowledge` | — |
| 22 | `evaluate_design` | — |
| 23 | `evaluate_design` | — |
| 24 | `doe_knowledge` | — |
| 25 | `evaluate_design` | — |
| 26 | `doe_knowledge` | — |
| 27 | `doe_knowledge` | — |
| 28 | `evaluate_design` | `doe_knowledge` |
| 29 | `evaluate_design` | — |
| 30 | `evaluate_design` | `doe_knowledge` |
| 31 | `analyze_experiment` | — |
| 32 | `analyze_experiment` | `visualize_doe` |
| 33 | `analyze_experiment` | `visualize_doe` |
| 34 | `analyze_experiment` | `doe_knowledge` |
| 35 | `analyze_experiment` | — |
| 36 | `analyze_experiment` | `visualize_doe`, `doe_knowledge` |
| 37 | `analyze_experiment` | — |
| 38 | `analyze_experiment` | `visualize_doe`, `doe_knowledge` |
| 39 | `analyze_experiment` | — |
| 40 | `analyze_experiment` | — |
| 41 | `analyze_experiment` | — |
| 42 | `analyze_experiment` | — |
| 43 | `analyze_experiment` | `doe_knowledge` |
| 44 | `doe_knowledge` | — |
| 45 | `optimize_responses` | — |
| 46 | `optimize_responses` | — |
| 47 | `visualize_doe` | `optimize_responses` |
| 48 | `doe_knowledge` | `optimize_responses` |
| 49 | `optimize_responses` | — |
| 50 | `doe_knowledge` | `analyze_experiment` |
| 51 | `doe_knowledge` | — |
| 52 | `doe_knowledge` | — |
| 53 | `optimize_responses` | `doe_knowledge` |
| 54 | `doe_knowledge` | — |
| 55 | `doe_knowledge` | — |
| 56 | `doe_knowledge` | `generate_design` |
| 57 | `doe_knowledge` | — |
| 58 | `doe_knowledge` | — |
| 59 | `doe_knowledge` | `generate_design` |
| 60 | `doe_knowledge` | `analyze_experiment` |
| 61 | `doe_knowledge` | `analyze_experiment` |
| 62 | `doe_knowledge` | — |
| 63 | `recommend_strategy` | `generate_design` |
| 64 | `recommend_strategy` | `generate_design` |
| 65 | `recommend_strategy` | — |
| 66 | `recommend_strategy` | `generate_design`, `optimize_responses` |
| 67 | `doe_knowledge` | `analyze_experiment` |
| 68 | `doe_knowledge` | `visualize_doe` |
| 69 | `doe_knowledge` | — |
| 70 | `doe_knowledge` | `analyze_experiment` |
| 71 | `doe_knowledge` | `analyze_experiment` |
| 72 | `doe_knowledge` | `evaluate_design` |
| 73 | `augment_design` | `doe_knowledge` |
| 74 | `doe_knowledge` | — |
| 75 | `augment_design` | `doe_knowledge` |
| 76 | `visualize_doe` | `doe_knowledge` |
| 77 | `augment_design` | `evaluate_design` |
| 78 | `doe_knowledge` | — |
| 79 | `doe_knowledge` | — |
| 80 | `doe_knowledge` | — |
| 81 | `doe_knowledge` | — |
| 82 | `doe_knowledge` | — |
| 83 | `doe_knowledge` | — |
| 84 | `doe_knowledge` | — |
| 85 | `doe_knowledge` | — |
| 86 | `analyze_experiment` | — |
| 87 | `doe_knowledge` | — |
| 88 | `doe_knowledge` | — |
| 89 | `doe_knowledge` | — |
| 90 | `doe_knowledge` | — |
| 91 | `generate_design` | `recommend_strategy` |
| 92 | `generate_design` | — |
| 93 | `generate_design` | — |
| 94 | `generate_design` | — |
| 95 | `generate_design` | — |
| 96 | `generate_design` | — |
| 97 | `recommend_strategy` | `augment_design` |
| 98 | `generate_design` | — |
| 99 | `optimize_responses` | — |
| 100 | `augment_design` | `evaluate_design` |
| 101 | `analyze_experiment` | `doe_knowledge` |
| 102 | `doe_knowledge` | — |
| 103 | `optimize_responses` | `analyze_experiment` |
| 104 | `recommend_strategy` | — |
| 105 | `generate_design` | — |
| 106 | `generate_design` | — |
| 107 | `generate_design` | — |
| 108 | `optimize_responses` | `analyze_experiment` |
| 109 | `doe_knowledge` | `recommend_strategy` |
| 110 | `doe_knowledge` | — |
| 111 | `doe_knowledge` | `generate_design` |
| 112 | `generate_design` | — |
| 113 | `doe_knowledge` | `generate_design` |
| 114 | `doe_knowledge` | `analyze_experiment` |
| 115 | `generate_design` | — |
| 116 | `generate_design` | — |
| 117 | `recommend_strategy` | — |
| 118 | `generate_design` | — |
| 119 | `optimize_responses` | `generate_design` |
| 120 | `generate_design` | — |
| 121 | `visualize_doe` | `optimize_responses` |
| 122 | `doe_knowledge` | — |
| 123 | `doe_knowledge` | — |
| 124 | `generate_design` | — |
| 125 | `generate_design` | `doe_knowledge` |
| 126 | `generate_design` | — |
| 127 | `optimize_responses` | `generate_design` |
| 128 | `doe_knowledge` | `visualize_doe` |
| 129 | `optimize_responses` | — |
| 130 | `doe_knowledge` | `evaluate_design` |
| 131 | `recommend_strategy` | `doe_knowledge` |
| 132 | `generate_design` | — |
| 133 | `generate_design` | — |
| 134 | `recommend_strategy` | `doe_knowledge` |
| 135 | `doe_knowledge` | `generate_design` |
| 136 | `generate_design` | — |
| 137 | `doe_knowledge` | — |
| 138 | `doe_knowledge` | — |
| 139 | `optimize_responses` | — |
| 140 | `generate_design` | — |
| 141 | `generate_design` | — |
| 142 | `doe_knowledge` | `evaluate_design` |
| 143 | `augment_design` | `doe_knowledge` |
| 144 | `generate_design` | — |
| 145 | `doe_knowledge` | `recommend_strategy` |
| 146 | `generate_design` | — |
| 147 | `analyze_experiment` | `doe_knowledge` |
| 148 | `generate_design` | — |
| 149 | `recommend_strategy` | — |
| 150 | `doe_knowledge` | — |
| 151 | `doe_knowledge` | `generate_design` |
| 152 | `doe_knowledge` | `analyze_experiment` |
| 153 | `doe_knowledge` | `analyze_experiment` |
| 154 | `doe_knowledge` | — |
| 155 | `optimize_responses` | — |
| 156 | `augment_design` | `evaluate_design`, `doe_knowledge` |
| 157 | `doe_knowledge` | — |
| 158 | `optimize_responses` | `doe_knowledge` |
| 159 | `doe_knowledge` | `generate_design` |
| 160 | `analyze_experiment` | — |
| 161 | `doe_knowledge` | `analyze_experiment` |
| 162 | `doe_knowledge` | — |
