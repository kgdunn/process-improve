# DOE Question Bank

162 questions across 16 categories. These define what the module must be able to answer.
Questions span classical/academic DOE (A–G) and applied scientific literature (H–P).

---

## A. Design Creation & Selection (20 questions)

| # | Question |
|---|---|
| 1 | I have 7 factors I want to test — how do I even start planning a DOE? |
| 2 | How many runs do I need? I have 5 factors and a budget for about 20 experiments. |
| 3 | What's the minimum number of experiments I need with 4 factors? |
| 4 | What is the difference between a full factorial and a fractional factorial? |
| 5 | I have 6 factors at 2 levels but can only afford 16 runs — design a fractional factorial and show the generators. |
| 6 | When should I use Plackett-Burman vs. a regular fractional factorial for screening? |
| 7 | What is a definitive screening design and when should I use one instead of Plackett-Burman? |
| 8 | When should I use CCD vs. Box-Behnken? What are the tradeoffs? |
| 9 | I have 3 factors and want to optimize yield and purity — design an experiment for me. |
| 10 | How do I augment my screening design to add runs for response surface modeling? |
| 11 | How do I set up a DOE with both continuous and categorical factors? |
| 12 | I need to screen 12 factors quickly with minimal runs — what design should I use? |
| 13 | What is the difference between a classical design (CCD) and an optimal design (D-optimal, I-optimal)? |
| 14 | How do I choose between D-optimal and I-optimal designs? |
| 15 | I have a mixture of 4 ingredients summing to 100% plus 2 process factors — what design do I need? |
| 16 | Design a DOE for 2^6 in 16 runs — what resolution will I get? |
| 17 | Construct the design matrix for a 2^(4-1) with generator D=ABC, list all 8 runs, and identify the defining relation. |
| 18 | I want a full quadratic model in 3 factors — how many runs for CCD vs. BBD, and what is alpha for a rotatable CCD? |
| 19 | How do I augment a 2^k factorial with center and axial points to create a CCD? |
| 20 | How does a face-centered CCD compare to a rotatable CCD in prediction variance? |

## B. Design Properties & Evaluation (10 questions)

| # | Question |
|---|---|
| 21 | What is design resolution and why does it matter? Difference between Resolution III, IV, and V? |
| 22 | Compute the alias structure for my 2^(5-2) with generators D=AB and E=AC. |
| 23 | Show me the confounding pattern for a Resolution IV design with 6 factors in 16 runs. |
| 24 | Explain the difference between confounding and aliasing. |
| 25 | For a 2^(7-4) design in 8 runs, what are the generators, defining relation, and full aliasing? |
| 26 | What is minimum aberration and why might I prefer it over maximizing resolution? |
| 27 | What does it mean for a design to be orthogonal? What happens when orthogonality is lost? |
| 28 | What are "clear" effects in a fractional factorial and how do I identify them? |
| 29 | How do I calculate statistical power — is my experiment big enough? |
| 30 | What is D-efficiency and how do I use it to compare designs? |

## C. Analysis of Designed Experiments (13 questions)

| # | Question |
|---|---|
| 31 | I ran a 2^3 factorial with 3 replicates — walk me through the ANOVA table. |
| 32 | I ran a single-replicate 2^4 — how do I estimate error? How does a half-normal plot help? |
| 33 | Use a Pareto plot to identify significant effects from my unreplicated 2^5. |
| 34 | My 2^(4-1) estimated effect [AB+CD]=10.4 — why is this aliased and how do I separate them? |
| 35 | Show me how to calculate main effects and interactions by hand for a 2^3. |
| 36 | Residuals vs. fitted shows a funnel shape — what does that mean and how do I fix it? |
| 37 | How do I use Box-Cox to choose a transformation? |
| 38 | Normal probability plot of residuals shows an S-curve — what remedial action? |
| 39 | Given model output with coefficients, SEs, and p-values — which effects are significant? |
| 40 | How do I use Lenth's method (PSE) to determine active effects in an unreplicated factorial? |
| 41 | Center point test: factorial avg=72.3, center avg=65.8 — is curvature significant? |
| 42 | Analyze 16 runs from my 2^4 — identify significant effects and give a reduced model. |
| 43 | Full factorial in 3 factors, 8 runs, 8 parameters, zero residual df — how to judge importance? |

## D. Response Surface Methodology (10 questions)

| # | Question |
|---|---|
| 44 | Describe the sequential nature of RSM — screening to steepest ascent to second-order model. |
| 45 | Given a fitted quadratic model, find the stationary point — is it a max, min, or saddle? |
| 46 | Explain steepest ascent — given b1=5.45 and b2=-3.20, what direction and step size? |
| 47 | How do I draw and interpret a contour plot? Which direction maximizes the response? |
| 48 | How do I optimize multiple conflicting responses? What is a desirability function? |
| 49 | Compute desirability for three responses: maximize yield >90%, minimize cost <$5, target viscosity 50+/-5. |
| 50 | What is a lack-of-fit test and how do center point replicates provide pure error? |
| 51 | What is rotatability and why is it desirable? |
| 52 | How do I know when to switch from first-order to second-order model? |
| 53 | Stationary point is outside my experimental region — what is ridge analysis? |

## E. Practical & Applied (13 questions)

| # | Question |
|---|---|
| 54 | Should I use DOE or one-factor-at-a-time? |
| 55 | How do I handle a hard-to-change factor like oven temperature? |
| 56 | How do I handle constraints where some factor combinations are impossible? |
| 57 | How do I set factor ranges (high/low levels)? How wide should I go? |
| 58 | Do I need to randomize? What if randomization is inconvenient? |
| 59 | How do I block my experiment across days or batches? |
| 60 | How do I handle missing/failed runs? |
| 61 | How do I perform and analyze confirmation runs? How many? |
| 62 | How do I use DOE when my response is binary (pass/fail)? |
| 63 | 8 factors — describe a screening strategy to narrow to 2–3 important factors in 16 runs. |
| 64 | Chemical engineer wants to maximize yield with T, P, catalyst% in ~20 runs — propose full strategy. |
| 65 | Experiments cost $5,000 each, budget for 25 runs — how to stage this? |
| 66 | Soap manufacturing with constraint 3T+5D<=600 — use RSM to maximize profit. |

## F. Interpretation (12 questions)

| # | Question |
|---|---|
| 67 | ANOVA shows significant lack of fit — what does that mean? |
| 68 | How do I interpret an interaction plot? What does it mean when lines cross? |
| 69 | Main effect A is not significant but AB interaction is — should I keep A? |
| 70 | R^2=0.95 but predicted R^2=0.40 — what does that mean? |
| 71 | Confirmation runs don't match predictions — what went wrong? |
| 72 | PB screening gave significant effects — are these real or aliased interactions? |
| 73 | Factor C is largest effect but aliased with ABD — how to de-alias? What is foldover? |
| 74 | Significant p-value but tiny effects — is this practically significant? |
| 75 | Significant curvature test from center points — what next? |
| 76 | Walk me through reading a Pareto plot and half-normal plot from my 2^5. |
| 77 | 2^(7-4) Res III, largest effect C aliased with AE+BF+DG — how to resolve? |
| 78 | After dropping insignificant effects, why don't remaining coefficients change? |

## G. Statistical Concepts (12 questions)

| # | Question |
|---|---|
| 79 | Difference between replication and repetition? |
| 80 | Why is randomization important? Concrete example of what goes wrong without it. |
| 81 | What are degrees of freedom and how do they partition in a 2^k factorial? |
| 82 | Explain sparsity-of-effects (effect heredity). |
| 83 | What is the hierarchy principle — keep main effect if its interaction is significant? |
| 84 | Difference between split-plot and blocked design? |
| 85 | Why use the highest-order interaction as generator for a fractional factorial? |
| 86 | Convert real-world units to coded units (-1 to +1). |
| 87 | What is projectivity and what happens when a factor is unimportant? |
| 88 | Why don't coefficients change when removing insignificant terms from a full factorial? |
| 89 | What is a split-plot design? Why two error terms? How does the ANOVA change? |
| 90 | Why does a 2^2 in 4 runs give more information than 4 OFAT experiments? |

## H. Pharma Formulation & Drug Delivery (13 questions)

| # | Question |
|---|---|
| 91 | Optimize nanoparticle formulation — screen polymer concentration, surfactant %, sonication time, phase ratio. |
| 92 | 2^3 full factorial for solid lipid nanoparticles: lipid content, emulsifier, homogenization time. |
| 93 | 5 variables for lipid nanoparticles — design a definitive screening design. |
| 94 | Optimize solid dispersion with 3^2 factorial: carrier:drug ratio and adsorbent:SD ratio. |
| 95 | 2^4 factorial for transdermal transfersome: phospholipid ratio, edge activator type, hydration volume, sonication. |
| 96 | BBD for hydrogel microspheres: polymer %, crosslinker, drug loading. |
| 97 | 2^3 factorial found significant effects for dropping pills — how to move to optimization? |
| 98 | CCD for nanocapsules: polymer concentration, active ingredient, drip rate, phase ratio. |
| 99 | Desirability function for particle size <100nm, PDI <=0.30, zeta >=30mV, EE >=70%. |
| 100 | Hybrid DOE: 2^4 screening then CCD optimization — how to combine data? |
| 101 | NLC optimization: factorial screening + CCD, lack-of-fit is significant — what to do? |
| 102 | QbD for oral suspension: build Ishikawa/FMEA to identify CMAs and CPPs. |
| 103 | Surfactant affects all three responses — find conditions satisfying all simultaneously. |

## I. Fermentation & Bioprocess (11 questions)

| # | Question |
|---|---|
| 104 | Optimize fermentation medium (7 factors) — too many for full factorial, what's my screening strategy? |
| 105 | PB experiment to screen 7 fermentation variables in 12 runs. |
| 106 | After PB identified 3 factors, design BBD for optimal concentrations. |
| 107 | BBD with temperature, pH, culture time for OD600. |
| 108 | Optimize solid-state fermentation — maximize xylanase and cellulase simultaneously. |
| 109 | PB identified significant factors but I suspect interactions — go straight to RSM or run Res IV first? |
| 110 | Compare RSM predictions with ANN — is this valid? |
| 111 | Screening 11 variables — is PB in 12 runs sufficient or do I need 20? |
| 112 | BBD for cucumber fermentation: salt, temperature, brine filling temperature. |
| 113 | Can I use an orthogonal array for second-stage fermentation optimization? |
| 114 | RSM model R^2=0.98 but confirmation run is 15% off — what went wrong? |

## J. Food Science & Beverage (10 questions)

| # | Question |
|---|---|
| 115 | D-optimal mixture experiment for composite flour blend. |
| 116 | BBD for fruit wine: temperature, sugar content, SO2 addition. |
| 117 | Screen and optimize brewing parameters: pH, Brix, time, inoculum, temperature. |
| 118 | CCD for polyphenol extraction: pressure, temperature, CO2 flow, co-solvent. |
| 119 | Microwave-assisted extraction: optimize power, time, water for yield, polyphenols, cannabinoids. |
| 120 | Mixture design for extruded snack: millet, sorghum, soy proportions. |
| 121 | BBD for pickle fermentation — contour plots and optimal salt/temperature/time. |
| 122 | CCD vs. BBD vs. Doehlert for extraction optimization — when to choose each? |
| 123 | Compare PB, factorial, and Taguchi OA for screening 8 extraction parameters. |
| 124 | DOE for enzymatic hydrolysis: enzyme concentration, pH, temperature, time. |

## K. Analytical Method Development / QbD (6 questions)

| # | Question |
|---|---|
| 125 | HPLC method QbD: screen critical method parameters with factorial design. |
| 126 | Fractional factorial to screen 6 HPLC parameters in 16 runs. |
| 127 | After screening, CCD to optimize mobile phase ratio, flow rate, column temperature. |
| 128 | Establish a design space for my analytical method from DOE data. |
| 129 | Desirability approach for HPLC with 4 factors and 3 responses. |
| 130 | Demonstrate method robustness using DOE instead of OFAT. |

## L. Cell Culture, Stem Cells & Biologics (5 questions)

| # | Question |
|---|---|
| 131 | Optimize iPSC differentiation (6 conditions, 21-day runs) — most efficient DOE approach? |
| 132 | DSD for CHO cell culture: temperature, pH, DO, glucose feed rate, amino acid supplement. |
| 133 | Factorial for transfection: lipid:DNA ratio, cell density, incubation time, serum. |
| 134 | Stem cell experiments are expensive/slow — sequential DOE strategy with minimal runs? |
| 135 | Continuous + categorical factors in cell culture — what design handles this? |

## M. mRNA/Vaccine & Biopharma (4 questions)

| # | Question |
|---|---|
| 136 | DSD for mRNA-LNP production with categorical and continuous factors. |
| 137 | Compare RSM vs. ML (XGBoost, ANN) with only ~30 DOE runs. |
| 138 | DSD screening + ANN hybrid model — how to combine? |
| 139 | Optimize mRNA-LNP vaccine: particle size 40–100nm, PDI <=0.30, zeta >=30mV, EE >=70%. |

## N. Natural Product Extraction (5 questions)

| # | Question |
|---|---|
| 140 | CCD for supercritical CO2 extraction: pressure, temperature, flow rate, co-solvent. |
| 141 | BBD for ultrasound-assisted extraction: power, time, solvent ratio, temperature. |
| 142 | Compare CCD, BBD, and Doehlert for extraction — which predicts better at edges? |
| 143 | CCD optimal at boundary — augment or shift ranges? |
| 144 | Taguchi L9 for screening before RSM optimization. |

## O. Bioprocess Scale-up & Environmental (4 questions)

| # | Question |
|---|---|
| 145 | PHB production scale-up — hard-to-change vs. easy-to-change factors. |
| 146 | BBD for gold nanoparticle biosynthesis after PB screening. |
| 147 | BBD model F=70.23, lack-of-fit p=0.13 — is model adequate? Interpret the ANOVA. |
| 148 | CCD for enzymatic saccharification with alpha=1.68 for rotatability. |

## P. Cross-Cutting Strategy (14 questions)

| # | Question |
|---|---|
| 149 | Two-stage strategy: PB screening then RSM for significant factors. |
| 150 | BBD vs. CCD vs. Doehlert for RSM optimization — when to use each? |
| 151 | Mixture components + process variables — what combined design? |
| 152 | ANOVA model adequacy: R^2, adj-R^2, pred-R^2, adequate precision, LOF, CV. |
| 153 | BBD shows significant lack-of-fit — add center points, transform, or different model? |
| 154 | ANN vs. RSM for prediction accuracy — when is ANN-DOE hybrid worthwhile? |
| 155 | Multiple conflicting responses — walk through Derringer's desirability function. |
| 156 | PB identified 3 factors but I suspect interactions — resolve aliases before RSM. |
| 157 | How many center point replicates for reliable pure error estimate? |
| 158 | Stationary point outside region — what does ridge analysis tell me? |
| 159 | Constraint: two factors can't both be high — how to handle in DOE? |
| 160 | After optimization, how many confirmation runs and how to verify statistically? |
| 161 | Coded vs. actual levels for fitting — does it matter for interpreting coefficients? |
| 162 | How to report DOE results in a publication — what tables, plots, statistics? |

---

## Summary

| Category | Count | Source |
|---|---|---|
| A. Design Creation & Selection | 20 | Classical/academic |
| B. Design Properties & Evaluation | 10 | Classical/academic |
| C. Analysis of Designed Experiments | 13 | Classical/academic |
| D. Response Surface Methodology | 10 | Classical/academic |
| E. Practical & Applied | 13 | Classical/academic |
| F. Interpretation | 12 | Classical/academic |
| G. Statistical Concepts | 12 | Classical/academic |
| H. Pharma Formulation & Drug Delivery | 13 | PubMed literature |
| I. Fermentation & Bioprocess | 11 | PubMed literature |
| J. Food Science & Beverage | 10 | PubMed literature |
| K. Analytical Method Development / QbD | 6 | PubMed literature |
| L. Cell Culture, Stem Cells & Biologics | 5 | PubMed literature |
| M. mRNA/Vaccine & Biopharma | 4 | PubMed literature |
| N. Natural Product Extraction | 5 | PubMed literature |
| O. Bioprocess Scale-up & Environmental | 4 | PubMed literature |
| P. Cross-Cutting Strategy | 14 | PubMed literature |
| **Total** | **162** | |
