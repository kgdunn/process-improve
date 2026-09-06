"""Build the SBR batch-PCA reference fixture and the openmv.net upload workbook.

The source files are not vendored. They live in the archived
`kgdunn/gsk-teaching <https://github.com/kgdunn/gsk-teaching>`_ repository at
commit ``87cfb3bbb02fa623bd19bd20e2813bff4a0cf5d4``:

- ``tests/SBRDATA.mat``: ``X`` (10600 x 9; 53 batches of 200 samples stacked
  vertically), ``Y`` (53 x 5), and the space-padded ``Xnames`` / ``Ynames``.
- ``tests/SBR-expected.mat``: ``t`` (53 x 2) and ``p`` (1800 x 2), the scores
  and loadings of the batch PCA fitted by the legacy MATLAB toolkit
  (``unit_tests.m::PCA_batch_data``: ``lvm`` with ``min_lv = 2`` on all nine
  tags; mean-centred and scaled to unit variance per unfolded column; a single
  block, so no block weighting).

This script writes, next to itself:

- ``reference_scores.csv``: columns ``batch_id, t1, t2``.
- ``reference_loadings.csv``: columns ``tag, sequence, p1, p2``. MATLAB unfolds
  time-major (all tags at sample 0, then all tags at sample 1, and so on), so
  row ``k * 9 + j`` of ``p`` is tag ``j`` at sample ``k``. The two cells with
  zero variance (the feed flows at sample 0) carry exactly zero loadings.
- ``reference_metadata.json``: provenance, the name mapping, and the R2 values
  the legacy unit test asserts.

and, at ``--workbook``, the ``sbr-batch-reactor.xlsx`` workbook that is hosted
on openmv.net: sheet ``X_batch`` (``batch_id``, ``time``, nine tags) and sheet
``Y_quality`` (``batch_id``, five quality attributes), the same layout as the
``batch-dryer.xlsx`` workbook.

Run from the repository root::

    uv run python tests/fixtures/sbr_batch_pca/prepare_fixture.py \
        --sbrdata /path/to/SBRDATA.mat \
        --expected /path/to/SBR-expected.mat \
        --workbook /path/to/sbr-batch-reactor.xlsx
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np
import pandas as pd
import scipy.io

HERE = pathlib.Path(__file__).parent
SCORES_CSV = HERE / "reference_scores.csv"
LOADINGS_CSV = HERE / "reference_loadings.csv"
METADATA_JSON = HERE / "reference_metadata.json"

N_BATCHES = 53
N_SAMPLES = 200
N_COMPONENTS = 2

# MATLAB tag name (space padding stripped) -> column name in the workbook.
TAG_NAMES = {
    "F STRYRENE": "StyreneFlow",
    "F BUTADIEN": "ButadieneFlow",
    "T FEED": "FeedTemp",
    "T REACTOR": "ReactorTemp",
    "T COOLING": "CoolingTemp",
    "T R. JACKT": "JacketTemp",
    "LATEX DENS": "LatexDensity",
    "CONVERSION": "Conversion",
    "ENERGY REL": "EnergyReleased",
}
QUALITY_NAMES = {
    "Compositon": "Composition",
    "Part. Size": "ParticleSize",
    "Branching": "Branching",
    "Cross Link": "CrossLinking",
    "Polydisper": "Polydispersity",
}
SOURCE = {
    "repository": "https://github.com/kgdunn/gsk-teaching",
    "commit": "87cfb3bbb02fa623bd19bd20e2813bff4a0cf5d4",
    "files": {
        "tests/SBRDATA.mat": "d3d4fd5161dc42929b9f5d416f28c9254705efcf",
        "tests/SBR-expected.mat": "f579145f60926bbc0c285b36298e6652dc55b588",
    },
}
# From ``unit_tests.m::PCA_batch_data``: assertEAE([.17085, .100531], R2Xb_a, 5).
LEGACY_R2_PER_COMPONENT = [0.17085, 0.100531]


def _rename(raw_names: np.ndarray, mapping: dict[str, str]) -> list[str]:
    """Strip MATLAB's space padding and map each name through ``mapping``."""
    names = [str(name).strip() for name in raw_names]
    unknown = [name for name in names if name not in mapping]
    if unknown:
        msg = f"Unexpected MATLAB variable names {unknown}; expected {list(mapping)}."
        raise ValueError(msg)
    return [mapping[name] for name in names]


def load_source(sbrdata: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the melted trajectory table and the quality table from ``SBRDATA.mat``."""
    mat = scipy.io.loadmat(sbrdata, squeeze_me=True)
    tags = _rename(mat["Xnames"], TAG_NAMES)
    qualities = _rename(mat["Ynames"], QUALITY_NAMES)
    x = np.asarray(mat["X"], dtype=float)
    y = np.asarray(mat["Y"], dtype=float)
    if x.shape != (N_BATCHES * N_SAMPLES, len(tags)) or y.shape != (N_BATCHES, len(qualities)):
        msg = f"Unexpected shapes X={x.shape}, Y={y.shape}."
        raise ValueError(msg)
    melted = pd.DataFrame(x, columns=tags)
    melted.insert(0, "time", np.tile(np.arange(1, N_SAMPLES + 1), N_BATCHES))
    melted.insert(0, "batch_id", np.repeat(np.arange(1, N_BATCHES + 1), N_SAMPLES))
    quality = pd.DataFrame(y, columns=qualities)
    quality.insert(0, "batch_id", np.arange(1, N_BATCHES + 1))
    return melted, quality


def load_expected(expected: pathlib.Path, tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the legacy scores and loadings, the loadings labelled by (tag, sequence)."""
    mat = scipy.io.loadmat(expected, squeeze_me=True)
    t = np.asarray(mat["t"], dtype=float)
    p = np.asarray(mat["p"], dtype=float)
    if t.shape != (N_BATCHES, N_COMPONENTS) or p.shape != (N_SAMPLES * len(tags), N_COMPONENTS):
        msg = f"Unexpected shapes t={t.shape}, p={p.shape}."
        raise ValueError(msg)
    scores = pd.DataFrame(t, columns=["t1", "t2"])
    scores.insert(0, "batch_id", np.arange(1, N_BATCHES + 1))
    loadings = pd.DataFrame(p, columns=["p1", "p2"])
    # Time-major unfolding: row k * n_tags + j is tag j at sample k.
    loadings.insert(0, "sequence", np.repeat(np.arange(N_SAMPLES), len(tags)))
    loadings.insert(0, "tag", np.tile(np.asarray(tags, dtype=object), N_SAMPLES))
    return scores, loadings


def write_workbook(melted: pd.DataFrame, quality: pd.DataFrame, path: pathlib.Path) -> None:
    """Write the two-sheet workbook hosted on openmv.net."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        melted.to_excel(writer, sheet_name="X_batch", index=False)
        quality.to_excel(writer, sheet_name="Y_quality", index=False)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sbrdata", type=pathlib.Path, required=True, help="path to SBRDATA.mat")
    parser.add_argument("--expected", type=pathlib.Path, required=True, help="path to SBR-expected.mat")
    parser.add_argument("--workbook", type=pathlib.Path, required=True, help="where to write sbr-batch-reactor.xlsx")
    args = parser.parse_args(argv)

    melted, quality = load_source(args.sbrdata)
    tags = [c for c in melted.columns if c not in ("batch_id", "time")]
    scores, loadings = load_expected(args.expected, tags)

    scores.to_csv(SCORES_CSV, index=False, float_format="%.10f")
    loadings.to_csv(LOADINGS_CSV, index=False, float_format="%.10f")
    metadata = {
        "source": SOURCE,
        "n_batches": N_BATCHES,
        "n_samples": N_SAMPLES,
        "n_components": N_COMPONENTS,
        "tag_names": TAG_NAMES,
        "quality_names": QUALITY_NAMES,
        "legacy_model": (
            "unit_tests.m::PCA_batch_data: lvm({'X', batch_X}, min_lv=2) on all nine tags; "
            "columns mean-centred and scaled to unit variance (N-1); single block, no block weighting"
        ),
        "legacy_r2_per_component": LEGACY_R2_PER_COMPONENT,
        "legacy_tolerance": "assertEAE: relative 1e-2 on |t| and |p| (sign ignored), 1e-5 on R2",
        "loadings_layout": "time-major: MATLAB row k * 9 + j is tag j at sample k",
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000"),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n")
    write_workbook(melted, quality, args.workbook)

    print(f"wrote {SCORES_CSV} {scores.shape}, {LOADINGS_CSV} {loadings.shape}, {METADATA_JSON}")
    print(f"wrote {args.workbook}: X_batch {melted.shape}, Y_quality {quality.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
