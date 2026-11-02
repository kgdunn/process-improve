# (c) Kevin Dunn, 2010-2026. MIT License.
"""Simulate T-shaped data (D, F, Z, Y) from a known truth, to test prediction with new materials.

The generative model, for two groups of materials ("A", where new materials appear, and "B"):

* Every material ``j`` has latent traits ``theta_j`` (3 for group A, 2 for group B) and one
  unmeasured trait ``u_j``. Its measured properties are a noisy linear function of the
  latent traits only: ``d_j = theta_j @ L.T + e_j``. So ``u_j`` is invisible in D.
* A blend uses two or three materials of group A, with Dirichlet fractions that sum to 1,
  and one material of group B (fraction 1). Process conditions Z are independent N(0, 1).
* Quality, for blend ``i`` with fractions ``f_ij``::

      y_i = tau_i @ G_A + theta_B(i) @ G_B + z_i @ G_Z            (ideal mixing of traits)
            + hidden * (sum_j f_ij u_j) * g_u                    (unmeasured material trait)
            + nonideal * (sum_j f_ij |theta_j - tau_i|^2) * g_n  (dissimilar materials do not mix ideally)
            + noise * eps_i

  where ``tau_i = sum_j f_ij theta_j`` is the ideal-mixing average of the group-A traits.

``z_effect`` and ``b_effect`` scale the effects of the process conditions and of group B; ``usage_skew`` makes some materials
far more popular than others, as in a real formulation history; ``hidden`` sets how much of a material's effect its properties cannot explain; ``nonideal``
sets how far mixing departs from the ideal (linear) rule; ``spread`` places one material
("A-out") far from the others in trait space, to test extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Simulated:
    """Simulated blocks plus the noise-free truth used to score predictions."""

    D: dict[str, pd.DataFrame]
    F: dict[str, pd.DataFrame]
    Z: dict[str, pd.DataFrame]
    Y: dict[str, pd.DataFrame]
    y_true: pd.DataFrame  # noise-free quality
    noise: float  # standard deviation of the noise on every Y column


def simulate(
    seed: int,
    n_blends: int = 80,
    n_materials_a: int = 14,
    n_materials_b: int = 5,
    hidden: float = 0.0,
    nonideal: float = 0.0,
    noise: float = 0.25,
    outlier_shift: float = 0.0,
    z_effect: float = 1.0,
    b_effect: float = 1.0,
    usage_skew: float = 0.0,
) -> Simulated:
    """Draw one data set. See the module docstring for the model."""
    rng = np.random.default_rng(seed)
    traits = {"A": rng.standard_normal((n_materials_a, 3)), "B": rng.standard_normal((n_materials_b, 2))}
    if outlier_shift:
        traits["A"][-1] += outlier_shift / np.sqrt(3)  # the last group-A material sits away from the rest
    hidden_trait = rng.standard_normal(n_materials_a)
    n_props = {"A": 8, "B": 4}
    names = {
        "A": [f"A{j + 1:02d}" for j in range(n_materials_a)],
        "B": [f"B{j + 1:02d}" for j in range(n_materials_b)],
    }
    if outlier_shift:
        names["A"][-1] = "A-out"
    d_blocks = {}
    for g in ("A", "B"):
        loadings = rng.standard_normal((n_props[g], traits[g].shape[1]))
        props = 10 + traits[g] @ loadings.T + 0.3 * rng.standard_normal((traits[g].shape[0], n_props[g]))
        d_blocks[g] = pd.DataFrame(props, index=names[g], columns=[f"{g}-prop{k + 1}" for k in range(n_props[g])])

    blend_names = [f"blend{i + 1:03d}" for i in range(n_blends)]
    f_a = np.zeros((n_blends, n_materials_a))
    popularity = np.exp(usage_skew * rng.standard_normal(n_materials_a))
    for i in range(n_blends):
        used = rng.choice(n_materials_a, size=rng.integers(2, 4), replace=False, p=popularity / popularity.sum())
        f_a[i, used] = rng.dirichlet(2 * np.ones(len(used)))
    f_b = np.eye(n_materials_b)[rng.integers(0, n_materials_b, n_blends)]
    z = rng.standard_normal((n_blends, 3))

    n_quality = 4
    g_a, g_b, g_z = (rng.standard_normal((k, n_quality)) for k in (3, 2, 3))
    g_u, g_n = rng.standard_normal(n_quality), rng.standard_normal(n_quality)
    tau = f_a @ traits["A"]
    dispersion = np.sum(f_a[:, :, None] * (traits["A"][None, :, :] - tau[:, None, :]) ** 2, axis=(1, 2))
    y_true = (
        tau @ g_a
        + b_effect * f_b @ traits["B"] @ g_b
        + z_effect * z @ g_z
        + hidden * np.outer(f_a @ hidden_trait, g_u)
        + nonideal * np.outer(dispersion, g_n)
    )
    y = y_true + noise * rng.standard_normal(y_true.shape)
    q_names = [f"Q{k + 1}" for k in range(n_quality)]
    return Simulated(
        D=d_blocks,
        F={
            "A": pd.DataFrame(f_a, index=blend_names, columns=names["A"]),
            "B": pd.DataFrame(f_b, index=blend_names, columns=names["B"]),
        },
        Z={"Z": pd.DataFrame(z, index=blend_names, columns=["z1", "z2", "z3"])},
        Y={"Y": pd.DataFrame(y, index=blend_names, columns=q_names)},
        y_true=pd.DataFrame(y_true, index=blend_names, columns=q_names),
        noise=noise,
    )
