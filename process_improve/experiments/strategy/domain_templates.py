# (c) Kevin Dunn, 2010-2026. MIT License.

"""Domain-specific strategy templates for DOE recommendations.

Each domain template provides preferred design choices, budget weight
adjustments, and domain-specific advice.  Templates are Python dicts
(not YAML) because they encode algorithmic adjustments, not reference data.

Sources:
    - ICH Q8/Q9/Q10 for pharma QbD
    - Stat-Ease SCOR framework
    - NIST Engineering Statistics Handbook section 5.3.3
    - Montgomery, *Design and Analysis of Experiments*, 10th ed.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Template structure
# ---------------------------------------------------------------------------
#
# Each template is a dict with the following keys:
#   screening_preference : str — preferred screening design type
#   rsm_preference : str — preferred RSM design type
#   budget_weights : dict — stage -> fraction of total budget
#   min_confirmation : int — minimum confirmation runs
#   min_center_points : int — minimum center points for variability estimation
#   prefer_curvature_detection : bool — prefer DSD over PB when possible
#   notes : dict[str, str] — detail_level -> domain-specific advice
#   extra_stages : list[str] — additional stages specific to this domain
#   special_considerations : list[str] — domain-specific warnings/notes

DOMAIN_TEMPLATES: dict[str, dict[str, Any]] = {
    "pharma_formulation": {
        "screening_preference": "definitive_screening",
        "rsm_preference": "ccd_face_centered",
        "budget_weights": {"screening": 0.25, "optimization": 0.45, "confirmation": 0.15},
        "min_confirmation": 5,
        "min_center_points": 5,
        "prefer_curvature_detection": True,
        "notes": {
            "novice": (
                "Pharmaceutical formulation studies follow Quality by Design (QbD) principles. "
                "The strategy includes a design space definition stage for regulatory submissions. "
                "Factors are called Critical Process Parameters (CPPs) or Critical Material "
                "Attributes (CMAs), and responses are Critical Quality Attributes (CQAs)."
            ),
            "intermediate": (
                "Per ICH Q8, the experimental strategy should map the design space — the "
                "multidimensional region of factor settings that provides acceptable quality. "
                "Face-centered CCD is preferred for RSM because it stays within physical bounds, "
                "which is important for regulatory acceptance. DSD screening efficiently detects "
                "curvature in drug release or dissolution profiles."
            ),
        },
        "extra_stages": ["risk_assessment", "design_space", "robustness"],
        "special_considerations": [
            "Regulatory requirements (ICH Q8/Q9/Q10) may require additional documentation.",
            "Design space definition is needed for process validation.",
            "Consider mixture constraints if formulation components must sum to 100%.",
        ],
    },
    "fermentation": {
        "screening_preference": "plackett_burman",
        "rsm_preference": "ccd",
        "budget_weights": {"screening": 0.30, "optimization": 0.50, "confirmation": 0.10},
        "min_confirmation": 3,
        "min_center_points": 5,
        "prefer_curvature_detection": False,
        "notes": {
            "novice": (
                "Fermentation processes typically involve many factors (pH, temperature, carbon "
                "source, nitrogen source, agitation, aeration, inoculum size). Plackett-Burman "
                "screening is the standard first step to identify which factors matter most."
            ),
            "intermediate": (
                "Fermentation has high biological variability, so extra center points (5+) are "
                "recommended for reliable error estimation. Sequential CCD augmentation from "
                "the factorial base is preferred over BBD because it reuses screening runs. "
                "Long experiment times (days to weeks) make run efficiency critical."
            ),
        },
        "extra_stages": ["validation", "scaleup"],
        "special_considerations": [
            "Biological variability requires extra center points (5+) for reliable error estimation.",
            "Long fermentation times (days to weeks) make run minimisation critical.",
            "Scale-up effects may require a separate verification stage.",
        ],
    },
    "food_science": {
        "screening_preference": "fractional_factorial",
        "rsm_preference": "box_behnken",
        "budget_weights": {"screening": 0.30, "optimization": 0.50, "confirmation": 0.10},
        "min_confirmation": 3,
        "min_center_points": 3,
        "prefer_curvature_detection": False,
        "notes": {
            "novice": (
                "Food science experiments often involve formulation (mixture) components. "
                "Box-Behnken designs are popular because they avoid extreme factor combinations "
                "that may produce unphysical products."
            ),
            "intermediate": (
                "If formulation components must sum to a constant (e.g. 100%), use mixture "
                "designs (simplex lattice or D-optimal mixture) instead of factorial designs. "
                "BBD is preferred for RSM because it avoids corners where food products may be "
                "unacceptable (e.g. too salty, too dry). Scheffe polynomials model mixture responses."
            ),
        },
        "extra_stages": [],
        "special_considerations": [
            "Check for mixture constraints — formulation components summing to a constant.",
            "Sensory responses may require larger replication for subjective variability.",
            "Extreme factor combinations may produce unacceptable products.",
        ],
    },
    "extraction": {
        "screening_preference": "fractional_factorial",
        "rsm_preference": "ccd",
        "budget_weights": {"screening": 0.30, "optimization": 0.50, "confirmation": 0.10},
        "min_confirmation": 3,
        "min_center_points": 3,
        "prefer_curvature_detection": False,
        "notes": {
            "novice": (
                "Extraction processes (supercritical CO2, solvent extraction, microwave-assisted) "
                "typically have 4-6 key factors: temperature, pressure, time, flow rate, "
                "co-solvent percentage, and particle size."
            ),
            "intermediate": (
                "CCD with rotatable alpha provides uniform prediction variance across the "
                "design region, which is important when the optimum may be near the region "
                "boundary. Extraction yield curves are often strongly quadratic."
            ),
        },
        "extra_stages": [],
        "special_considerations": [
            "Extraction processes often have strongly quadratic responses.",
            "CCD with rotatable alpha provides good prediction at region boundaries.",
        ],
    },
    "analytical_method": {
        "screening_preference": "fractional_factorial",
        "rsm_preference": "ccd",
        "budget_weights": {"screening": 0.25, "optimization": 0.45, "confirmation": 0.15},
        "min_confirmation": 5,
        "min_center_points": 3,
        "prefer_curvature_detection": False,
        "notes": {
            "novice": (
                "Analytical method development follows Analytical Quality by Design (AQbD) "
                "principles per ICH Q2/Q14. Factors include mobile phase composition, flow rate, "
                "column temperature, pH, and injection volume."
            ),
            "intermediate": (
                "A robustness study is typically the final stage, verifying that the method "
                "performs within specification when factors vary within normal operating ranges. "
                "Fractional factorials at resolution IV are standard for robustness screening."
            ),
        },
        "extra_stages": ["scouting", "robustness"],
        "special_considerations": [
            "AQbD / ICH Q2/Q14 framework requires method robustness verification.",
            "Robustness study is typically a fractional factorial at resolution IV.",
            "System suitability criteria should be defined before starting.",
        ],
    },
    "cell_culture": {
        "screening_preference": "definitive_screening",
        "rsm_preference": "box_behnken",
        "budget_weights": {"screening": 0.35, "optimization": 0.45, "confirmation": 0.10},
        "min_confirmation": 3,
        "min_center_points": 3,
        "prefer_curvature_detection": True,
        "notes": {
            "novice": (
                "Cell culture experiments are expensive and time-consuming (often 14-21 days "
                "per run). Definitive Screening Designs are recommended because they screen "
                "factors and detect curvature simultaneously, saving an entire experimental stage."
            ),
            "intermediate": (
                "With potentially 50-100+ media components, a two-phase approach is typical: "
                "first a library screen or PB to reduce to a manageable set, then a DSD or BBD "
                "for fine-tuning. Run minimisation is the top priority given the long timelines "
                "and high per-run costs."
            ),
        },
        "extra_stages": ["library_screen", "scaleup"],
        "special_considerations": [
            "Experiments are expensive and slow (14-21 days per run).",
            "Run minimisation is critical — every run saved is 2-3 weeks.",
            "DSD allows screening and curvature detection in a single stage.",
            "High-dimensionality media screening may need a pre-screening step.",
        ],
    },
    "bioprocess": {
        "screening_preference": "plackett_burman",
        "rsm_preference": "ccd",
        "budget_weights": {"screening": 0.30, "optimization": 0.50, "confirmation": 0.10},
        "min_confirmation": 3,
        "min_center_points": 5,
        "prefer_curvature_detection": False,
        "notes": {
            "novice": (
                "Bioprocess optimisation (upstream and downstream) typically involves many "
                "factors. Plackett-Burman screening identifies the vital few before investing "
                "in detailed response surface studies."
            ),
            "intermediate": (
                "Scale-up from bench to pilot to production is a key consideration. Factors "
                "that are hard to change at production scale (e.g. reactor geometry, impeller "
                "type) should be flagged early. The strategy should consider whether results "
                "will transfer across scales."
            ),
        },
        "extra_stages": ["validation", "scaleup"],
        "special_considerations": [
            "Scale-up effects may invalidate bench-scale optimisation.",
            "Hard-to-change factors are common (reactor geometry, media preparation).",
            "Biological variability requires extra center points for error estimation.",
        ],
    },
    "general": {
        "screening_preference": None,  # Use base rule engine defaults
        "rsm_preference": None,
        "budget_weights": {"screening": 0.30, "optimization": 0.50, "confirmation": 0.10},
        "min_confirmation": 3,
        "min_center_points": 3,
        "prefer_curvature_detection": False,
        "notes": {
            "novice": (
                "A general DOE strategy follows the screening-optimisation-confirmation "
                "sequence. The specific design choices depend on the number of factors, "
                "budget, and whether you have prior knowledge about which factors matter."
            ),
            "intermediate": (
                "The strategy follows the Stat-Ease SCOR framework: Screening, "
                "Characterisation, Optimisation, Ruggedness/Robustness. Budget is allocated "
                "roughly 25-40% screening, 40-55% optimisation, 5-15% confirmation."
            ),
        },
        "extra_stages": [],
        "special_considerations": [],
    },
}


def get_domain_template(domain: str) -> dict[str, Any]:
    """Return the domain template for the given domain string.

    Parameters
    ----------
    domain : str
        Domain key (e.g. ``"fermentation"``).  Falls back to ``"general"``
        if the key is not recognised.

    Returns
    -------
    dict
        The domain template dictionary.
    """
    return DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["general"])
