"""Build the reference catalogue's designs with another library, and compare each printed number.

The catalogue's own checker (``_reproduce``) measures every printed number that depends on how a
design is built, twice: once on the design process-improve builds, and once on the design the other
library builds. Only the construction changes, so a difference is a difference in the design.

Run it from the repository root, naming the library in ``DOE_LIB``::

    DOE_LIB=R uv run python scripts/design_crosscheck/compare.py all

``DOE_LIB`` is ``pyDOE3``, ``pydoe``, ``R`` (FrF2, DoE.base, rsm and daewr), ``dexpy``,
``oapackage`` or ``dsdpkg``. The argument is ``all``, or a seed for a random sample of 20 entries.
``DOE_OUT`` names the folder for the result (default: the working directory). See the README beside
this file for the environment each library needs.
"""

import contextlib
import functools
import importlib
import itertools
import json
import operator
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import process_improve.experiments.reference_catalogue as rc
from process_improve.experiments.designs_utils import build_design_result

HERE = Path(__file__).resolve().parent
LIB = os.environ.get("DOE_LIB", "pyDOE3")
OUT = Path(os.environ.get("DOE_OUT", "."))
DEXPY_PYTHON = os.environ.get("DEXPY_PYTHON", "python")  # dexpy needs pandas 1.x: an interpreter of its own
ARG = sys.argv[1] if len(sys.argv) > 1 else "20"  # a seed for a sample of 20, or "all"
SEED, N_SAMPLE, TIMEOUT_S = (0 if ARG == "all" else int(ARG)), (None if ARG == "all" else 20), 60

pyDOE3 = importlib.import_module(LIB) if LIB in ("pyDOE3", "pydoe") else None  # noqa: N816 - the package's name
_star = (getattr(pyDOE3, "star", None) or pyDOE3.doe_star.star) if pyDOE3 else None
_ENGINE = [LIB]  # the package that built the design of the check in hand


class Unsupported(Exception):  # noqa: N818 - pyDOE3 has no construction for this request
    pass


# --------------------------------------------------------------------------- pyDOE3 constructions
def _gen_string(names: list[str], generators: list[str]) -> str:
    """Translate generators such as ``E=-ABCD`` into pyDOE3's ``"a b c d -abcd"``."""
    lhs = {g.split("=")[0].strip(): g.split("=")[1].strip() for g in generators}
    base = [n for n in names if n not in lhs]
    letter = {n: chr(97 + i) for i, n in enumerate(base)}
    pattern = re.compile("|".join(sorted(map(re.escape, names), key=len, reverse=True)))
    tokens = []
    for n in names:
        if n in letter:
            tokens.append(letter[n])
            continue
        rhs = lhs[n].replace("\u2212", "-")
        sign = "-" if rhs.startswith("-") else ""
        tokens.append(sign + "".join(letter[m] for m in pattern.findall(rhs)))
    return " ".join(tokens)


def _relation_from_matrix(matrix: np.ndarray, names: list[str]) -> tuple[list[str], list[str], int | None]:  # noqa: C901
    """Read generators, defining relation and resolution off a two-level matrix (independent of process-improve)."""

    def as_product(col: np.ndarray, base: list[int]) -> tuple[tuple[int, ...], int] | None:
        for r in range(1, len(base) + 1):
            for subset in itertools.combinations(base, r):
                prod = np.prod(matrix[:, list(subset)], axis=1)
                if np.array_equal(prod, col):
                    return subset, 1
                if np.array_equal(prod, -col):
                    return subset, -1
        return None

    base, gens = [], []
    for j in range(matrix.shape[1]):  # a column that is no product of earlier base columns is a base factor
        found = as_product(matrix[:, j], base)
        if found is None:
            base.append(j)
        else:
            gens.append((j, *found))
    if 2 ** len(base) != matrix.shape[0]:
        raise Unsupported("pyDOE3 matrix is not a regular fraction")
    generators = [f"{names[j]}={'-' if s < 0 else ''}{''.join(names[i] for i in sub)}" for j, sub, s in gens]
    if len(gens) > 16:  # a relation this long is worked out on request, from the generators
        return generators, None, None
    words = []
    for r in range(1, len(gens) + 1):
        for combo in itertools.combinations(gens, r):
            letters, sign = [], 1
            for j, sub, s in combo:
                letters += [j, *sub]
                sign *= s
            word = sorted(i for i in set(letters) if letters.count(i) % 2)
            words.append((sign, word))
    relation = [f"I={'-' if s < 0 else ''}{''.join(names[i] for i in w)}" for s, w in words]
    return generators, relation, min((len(w) for _, w in words), default=None)


def _fraction(names: list[str], generators: list[str] | None, resolution: int | None) -> np.ndarray:
    if generators:
        return pyDOE3.fracfact(_gen_string(names, generators))
    if resolution:
        return pyDOE3.fracfact_by_res(len(names), resolution)
    # neither given: the half fraction, last factor the product of the others (process-improve's default)
    return pyDOE3.fracfact(
        " ".join([chr(97 + i) for i in range(len(names) - 1)] + ["".join(chr(97 + i) for i in range(len(names) - 1))])
    )


def _oa(levels: list[int]) -> np.ndarray:
    """Return the smallest pyDOE3 orthogonal array with enough columns of each level count."""
    need = {n: levels.count(n) for n in set(levels)}
    for name in sorted(pyDOE3.list_orthogonal_arrays(), key=lambda s: int(re.match(r"L(\d+)", s)[1])):
        arr = np.asarray(pyDOE3.get_orthogonal_array(name), dtype=float)
        col_levels = [len(np.unique(arr[:, j])) for j in range(arr.shape[1])]  # the names do not always say
        if all(col_levels.count(n) >= c for n, c in need.items()):
            picked, used = [], set()
            for n in levels:
                j = next(j for j, cl in enumerate(col_levels) if cl == n and j not in used)
                used.add(j)
                col = arr[:, j]
                vals = np.unique(col)
                picked.append(np.interp(col, vals, np.linspace(-1, 1, n)))
            return np.column_stack(picked)
    raise Unsupported(f"no pyDOE3 orthogonal array for levels {levels}")


def _ccd(k: int, opts: dict, names: list[str]) -> tuple[np.ndarray, float]:
    alpha = opts.get("alpha", "rotatable")
    if not isinstance(alpha, str):
        raise Unsupported("pyDOE3 ccdesign takes no numeric axial distance")
    if alpha not in ("rotatable", "orthogonal", "face_centered", "inscribed"):
        raise Unsupported(f"no {alpha!r} axial distance in ccdesign or star")
    face = {"face_centered": "faced", "inscribed": "inscribed"}.get(alpha, "circumscribed")
    kind = alpha if alpha in ("rotatable", "orthogonal") else "rotatable"
    nc = int(opts.get("n_center_points") or 0)
    center = (nc // 2, nc - nc // 2)
    if opts.get("cube") == "fractional":  # pyDOE3 has no fractional-cube CCD: compose its own pieces
        gens, res = opts.get("generators"), opts.get("resolution")
        if not gens and not res:  # the same default as process-improve: the half fraction, last = all others
            gens = [f"{names[-1]}={''.join(names[:-1])}"]
        cube = _fraction(names, gens, res)
        star = _star(k, alpha={"faced": "faced"}.get(face, kind), center=(1, 1))[0]
        if face == "inscribed":
            cube = cube / np.abs(star).max()
            star = star / np.abs(star).max()
        m = np.vstack([cube, np.zeros((center[0], k)), star, np.zeros((center[1], k))])
    else:
        m = pyDOE3.ccdesign(k, center=center, alpha=kind, face=face)
    return m, float(np.abs(m).max() if face == "circumscribed" else 1.0)


def pydoe3_generate_design(factors, design_type=None, **opts):  # noqa: C901
    """generate_design's signature, built with pyDOE3; the result is packed in a DesignResult unchanged."""
    _BUILT[0] = True
    names, k = [f.name for f in factors], len(factors)
    blocked = opts.get("n_blocks") or opts.get("block_generators")
    if blocked:
        return _blocked(factors, design_type, opts)
    nc = int(opts.get("n_center_points") or 0)
    gens = rel = res = alpha = None
    centers_inside = False
    if design_type == "full_factorial":
        lv = [len(f.levels) if f.levels else 2 for f in factors]
        m = pyDOE3.ff2n(k) if set(lv) == {2} else pyDOE3.fullfact(lv)
        if set(lv) != {2}:
            m = np.column_stack([np.linspace(-1, 1, n)[m[:, j].astype(int)] for j, n in enumerate(lv)])
    elif design_type == "fractional_factorial":
        m = _fraction(names, opts.get("generators"), opts.get("resolution"))
        gens, rel, res = _relation_from_matrix(m, names)
    elif design_type == "plackett_burman":
        m = pyDOE3.pbdesign(k)
    elif design_type == "box_behnken":
        m, centers_inside = pyDOE3.bbdesign(k, center=nc), True
    elif design_type == "ccd":
        m, alpha = _ccd(k, opts, names)
        centers_inside = True
    elif design_type == "dsd" and hasattr(pyDOE3, "definitive_screening_design"):
        if opts.get("budget"):
            raise Unsupported("harness does not map a DSD budget to fake factors")
        # process-improve counts centre runs in total, the DSD's own one included
        m, nc = pyDOE3.definitive_screening_design(k), max(nc - 1, 0)
    elif design_type == "taguchi":
        m = _oa([len(f.levels) if f.levels else 2 for f in factors])
    else:
        raise Unsupported(f"pyDOE3 has no {design_type!r} design")
    return build_design_result(
        np.asarray(m, dtype=float),
        factors,
        design_type,
        n_center_points=0 if centers_inside else nc,
        n_replicates=int(opts.get("n_replicates") or 1),
        generators=gens,
        defining_relation=rel,
        resolution=res,
        alpha=alpha,
        randomize=False,
    )


def _blocked(factors, design_type, opts):  # noqa: ANN202
    """Pydoe's blocked designs: block_full_factorial (generators given), block_ccdesign (two blocks, full cube)."""
    names, k = [f.name for f in factors], len(factors)
    nc = int(opts.get("n_center_points") or 0)
    if design_type == "full_factorial" and opts.get("block_generators") and hasattr(pyDOE3, "block_full_factorial"):
        m, labels = pyDOE3.block_full_factorial(k, [tuple(names.index(c) for c in g) for g in opts["block_generators"]])
    elif design_type == "full_factorial":
        raise Unsupported("block_full_factorial needs the block generators; the book's design does not give them")
    elif (
        design_type == "ccd"
        and opts.get("n_blocks") == 2
        and opts.get("cube", "full") == "full"
        and opts.get("alpha") in (None, "orthogonal")
        and hasattr(pyDOE3, "block_ccdesign")
    ):
        m, labels = pyDOE3.block_ccdesign(k, center=(nc // 2, nc - nc // 2))
        nc = 0
    else:
        raise Unsupported(f"no blocking for a {design_type} design with these options")
    m = np.asarray(m, float)
    res = build_design_result(m, factors, design_type, n_center_points=nc, randomize=False)
    labels = np.asarray(labels)
    # effects confounded with blocks: the factor products constant within every block, read off pydoe's labels
    words = []
    for r in range(1, k + 1):
        for cols in itertools.combinations(range(k), r):
            prod = np.prod(m[:, list(cols)], axis=1)
            if all(np.ptp(prod[labels == b]) == 0 for b in np.unique(labels)):
                words.append("".join(names[c] for c in cols))
    res.blocks = labels.tolist()
    res.metadata["blocking"] = {"confounded_with": words if design_type == "full_factorial" else []}
    if design_type == "ccd":
        res.alpha = float(np.abs(m).max())
    return res


def other_generate_design(factors, design_type=None, **opts):
    """generate_design's signature, built with OApackage or the definitive_screening_design package."""
    _BUILT[0] = True
    if opts.get("n_blocks") or opts.get("block_generators"):
        raise Unsupported(f"{LIB} has no blocking")
    k, nc = len(factors), int(opts.get("n_center_points") or 0)
    deadline = time.monotonic() + TIMEOUT_S - 5
    if LIB == "dsdpkg" and design_type == "dsd":
        import definitive_screening_design as dsdp  # noqa: PLC0415

        fake = 0
        if opts.get("budget"):  # process-improve fills a budget with fake factors; so does the package
            while len(dsdp.generate(n_num=k, n_fake_factors=fake + 1, min_13=False, verbose=False)) <= opts["budget"]:
                fake += 1
        m = dsdp.generate(n_num=k, n_fake_factors=fake, min_13=False, verbose=False).to_numpy(float)
        nc = max(nc - 1, 0)  # process-improve counts centre runs in total, the DSD's own one included
    elif LIB == "oapackage" and design_type == "dsd":
        import oa_builders  # noqa: PLC0415
        import oapackage  # noqa: PLC0415

        if opts.get("budget"):
            raise Unsupported("harness does not map a DSD budget for OApackage")
        c = oa_builders.conference_design(k if k % 2 == 0 else k + 1, k, deadline)
        m = np.array(oapackage.conference2DSD(oapackage.array_link(c.astype(int))), float)
        nc = max(nc - 1, 0)
    elif LIB == "oapackage" and design_type == "taguchi":
        import oa_builders  # noqa: PLC0415

        levels = [len(f.levels) if f.levels else 2 for f in factors]
        a = oa_builders.orthogonal_array(levels, deadline)
        m = np.column_stack([np.linspace(-1, 1, n)[a[:, j].astype(int)] for j, n in enumerate(levels)])
    else:
        raise Unsupported(f"{LIB} has no {design_type!r} design")
    return build_design_result(m, factors, design_type, n_center_points=nc, randomize=False)


_DEXPY_CACHE: dict = {}
_DEXPY_LAST: dict = {}  # the fraction dexpy chose for a run count, until the catalogue asks for it


def dexpy(request: dict) -> np.ndarray:
    """Dexpy runs on pandas 1.x in its own environment: ask it there, once per request."""
    key = json.dumps(request, sort_keys=True)
    if key not in _DEXPY_CACHE:
        out = subprocess.run(  # noqa: S603 - our own worker script
            [DEXPY_PYTHON, "-I", str(HERE / "dexpy_worker.py")],
            input=key,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
        if out.returncode:
            raise ValueError(f"dexpy: {out.stderr.strip().splitlines()[-1][:120]}")
        _DEXPY_CACHE[key] = np.array(json.loads(out.stdout), dtype=float)
    return _DEXPY_CACHE[key]


def dexpy_generate_design(factors, design_type=None, **opts):  # noqa: C901
    """generate_design's signature, built with dexpy (Stat-Ease)."""
    _BUILT[0] = True
    if opts.get("n_blocks") or opts.get("block_generators"):
        raise Unsupported("dexpy has no blocking")
    names, k, nc = [f.name for f in factors], len(factors), int(opts.get("n_center_points") or 0)
    gens = rel = res = alpha = None
    inside = False
    if any(f.levels for f in factors):
        raise Unsupported("dexpy builds two-level factorials only")
    if design_type == "full_factorial":
        m = dexpy({"kind": "full", "k": k})
    elif design_type == "fractional_factorial":
        if opts.get("generators") and _DEXPY_LAST.get("k") == k:  # the fraction dexpy itself chose by run count
            m = _DEXPY_LAST.pop("matrix")
            _DEXPY_LAST.clear()
        elif opts.get("generators") or opts.get("resolution"):
            raise Unsupported("dexpy chooses the fraction by run count; it takes no generators or resolution")
        else:
            m = dexpy({"kind": "fraction", "k": k, "runs": 2 ** (k - 1)})
        gens, rel, res = _relation_from_matrix(m, names)
    elif design_type == "ccd":
        a = opts.get("alpha", "rotatable")
        if opts.get("cube") == "fractional":
            raise Unsupported("dexpy builds a CCD on the full cube only")
        a = {"face_centered": "face centered"}.get(a, a)
        if a == "inscribed":
            raise Unsupported("dexpy has no inscribed CCD")
        m, inside = dexpy({"kind": "ccd", "k": k, "alpha": a, "centers": nc}), True
        alpha = float(np.abs(m).max())
    elif design_type == "box_behnken":
        m, inside = dexpy({"kind": "bbd", "k": k, "centers": nc}), True
    else:
        raise Unsupported(f"dexpy has no {design_type!r} design")
    return build_design_result(
        m,
        factors,
        design_type,
        n_center_points=0 if inside else nc,
        n_replicates=int(opts.get("n_replicates") or 1),
        generators=gens,
        defining_relation=rel,
        resolution=res,
        alpha=alpha,
        randomize=False,
    )


def dexpy_min_aberration(n_runs: int, k: int) -> list[str]:
    _BUILT[0] = True
    m = dexpy({"kind": "fraction", "k": k, "runs": n_runs})
    _DEXPY_LAST.update(k=k, matrix=m)
    gens, _, _ = _relation_from_matrix(m, list(rc._LETTERS[:k]))
    return gens


# --------------------------------------------------------------------------- R: FrF2, DoE.base, rsm, daewr
_R: dict = {"proc": None, "cache": {}}
FRF2_LETTERS = list("ABCDEFGHJKLMNOPQRSTUVWXYZ") + list("abcdefghjklmnopqrstuvwxyz")  # FrF2's Letters: no I
R_PACKAGE = {
    "fractional_factorial": "FrF2", "plackett_burman": "FrF2", "full_factorial": "FrF2",
    "ccd": "rsm", "box_behnken": "rsm", "dsd": "daewr", "taguchi": "DoE.base",
}  # fmt: skip


def r(request: dict) -> dict:
    """Ask the R worker (one process, kept open) for a design; an R error means the package cannot build it."""
    key = json.dumps(request, sort_keys=True)
    if key not in _R["cache"]:
        if _R["proc"] is None or _R["proc"].poll() is not None:
            _R["proc"] = subprocess.Popen(  # noqa: S603
                ["Rscript", "--vanilla", str(HERE / "r_worker.R")],  # noqa: S607
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        proc = _R["proc"]
        try:
            proc.stdin.write(key + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            while line and not line.startswith("@@"):
                line = proc.stdout.readline()
        except TimeoutError:
            proc.kill()  # the reply would arrive late and answer the next request
            _R["proc"], _R["cache"][key] = None, {"timeout": True}
            raise
        if not line:
            _R["proc"] = None
            raise ValueError("the R worker stopped")
        _R["cache"][key] = json.loads(line[2:])
    hit = _R["cache"][key]
    if "timeout" in hit:
        raise TimeoutError
    if "error" in hit:
        raise Unsupported(f"R: {' '.join(hit['error'].split())[:110]}")
    return hit


def _signed_generators(names: list[str], generators: list[str]) -> tuple[list[list[int]], list[int], list[int], int]:
    """Return generators such as ``E=-ABCD`` as FrF2 takes them: base-factor positions and signs, base factors first."""
    lhs = {g.split("=")[0].strip(): g.split("=")[1].strip().replace("\u2212", "-") for g in generators}
    base = [n for n in names if n not in lhs]
    made = [n for n in names if n in lhs]
    pattern = re.compile("|".join(sorted(map(re.escape, names), key=len, reverse=True)))
    positions = [[base.index(x) + 1 for x in pattern.findall(lhs[n])] for n in made]
    signs = [-1 if lhs[n].startswith("-") else 1 for n in made]
    order = [base.index(n) if n in base else len(base) + made.index(n) for n in names]  # R's columns: base, then made
    return positions, signs, order, len(base)


def _block_words(m: np.ndarray, labels: np.ndarray, names: list[str], asked: list[str] | None) -> list[str]:
    """Return the effects confounded with blocks: the factor products that are constant within every block.

    In a fraction a block effect is an alias class, so one word stands for each class, as a source prints
    it: the word the request asked for when the class holds one (a product of the block generators), and
    the shortest word otherwise. A word that is constant over the whole design is in the defining relation,
    not a block effect.
    """
    k = len(names)
    constant = {}  # word, as a bit mask -> whether it is constant over the whole design
    for size in range(1, k + 1):
        for cols in itertools.combinations(range(k), size):
            prod = np.prod(m[:, list(cols)], axis=1)
            if all(np.ptp(prod[labels == b]) == 0 for b in np.unique(labels)):
                constant[sum(1 << c for c in cols)] = np.ptp(prod) == 0
    defining = [0] + [word for word, whole in constant.items() if whole]
    masks = [sum(1 << names.index(c) for c in word) for word in asked or []]
    wanted = {
        functools.reduce(operator.xor, group)
        for r in range(1, len(masks) + 1)
        for group in itertools.combinations(masks, r)
    }
    seen, chosen = set(), []
    for word, whole in constant.items():
        if whole or word in seen:
            continue
        alias_class = {word ^ d for d in defining}
        seen |= alias_class
        chosen.append(min(alias_class, key=lambda w: (w not in wanted, w.bit_count(), w)))
    return [
        "".join(names[c] for c in range(k) if word >> c & 1)
        for word in sorted(chosen, key=lambda w: (w.bit_count(), w))
    ]


def _r_ccd(k: int, opts: dict, names: list[str], nc: int) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Rsm's ccd: cube (full, or from generators) and star as two blocks, the cube split when three are asked."""
    n_blocks = opts.get("n_blocks")
    alpha = opts.get("alpha", "orthogonal" if n_blocks else "rotatable")  # as process-improve, when blocked
    cube_blocks = (n_blocks or 2) - 1
    if nc % (cube_blocks + 1) == 0:  # rsm takes centre runs per block: equal counts when they divide
        n0 = [nc // (cube_blocks + 1)] * 2
    else:  # else process-improve's split: half to the cube portion, the rest to the star
        n0 = [(nc // 2) // cube_blocks, nc - ((nc // 2) // cube_blocks) * cube_blocks]
    req = {"kind": "ccd", "k": k, "n0": n0, "inscribed": alpha == "inscribed"}
    req["alpha"] = (
        alpha if not isinstance(alpha, str) else {"face_centered": "faces", "inscribed": "rotatable"}.get(alpha, alpha)
    )
    order, nb = list(range(k)), k
    if opts.get("cube") == "fractional":
        gens = opts.get("generators")
        if not gens and opts.get("resolution"):  # rsm takes generators only: FrF2 picks them for the resolution
            cube = np.array(r({"kind": "fraction", "k": k, "resolution": int(opts["resolution"])})["matrix"], float)
            gens = _relation_from_matrix(cube, names)[0]
        gens = gens or [f"{names[-1]}={''.join(names[:-1])}"]  # process-improve's default: the half fraction
        positions, signs, order, nb = _signed_generators(names, gens)
        req.update(positions=positions, signs=signs)
    if n_blocks == 3:  # two cube blocks and the star block: split the cube on its highest-order base interaction
        req["cube_blocks"] = "*".join(f"x{i + 1}" for i in range(nb))
    elif n_blocks not in (None, 2):
        raise Unsupported(f"harness maps 2 or 3 CCD blocks to rsm, not {n_blocks}")
    declined = False
    try:
        out = r(req)
    except Unsupported as exc:
        if "not achievable" not in str(exc):
            raise
        # rsm declines "rotatable" on a cube below resolution V. Build the same runs with the star on the faces,
        # so that run counts can still be read, and report no axial distance.
        out, declined = r({**req, "alpha": "faces"}), True
    m = np.array(out["matrix"], float)[:, order]
    blocks = np.array(out["blocks"]) if n_blocks and out.get("blocks") else None
    return m, float("nan") if declined else float(np.abs(m).max()), blocks


def r_generate_design(factors, design_type=None, **opts):  # noqa: C901, PLR0912, PLR0915
    """generate_design's signature, built with FrF2, DoE.base, rsm or daewr."""
    _BUILT[0] = True
    names, k = [f.name for f in factors], len(factors)
    nc = int(opts.get("n_center_points") or 0)
    lv = [len(f.levels) if f.levels else 2 for f in factors]
    gens = rel = res = alpha = blocks = None
    inside = False
    mixed_levels = design_type == "full_factorial" and set(lv) != {2}
    _ENGINE[0] = "DoE.base" if mixed_levels else R_PACKAGE.get(design_type, "R")
    if design_type in ("full_factorial", "fractional_factorial") and set(lv) == {2}:
        req, order = {"kind": "fraction", "k": k}, list(range(k))
        if design_type == "full_factorial":
            req["runs"] = 2**k
        elif opts.get("generators"):
            positions, signs, order, nb = _signed_generators(names, opts["generators"])
            req.update(runs=2**nb, positions=positions, signs=signs)
        elif opts.get("resolution"):
            req["resolution"] = int(opts["resolution"])
        else:
            req["runs"] = 2 ** (k - 1)  # as process-improve: the half fraction
        if opts.get("block_generators"):
            req["block_words"] = ["".join(FRF2_LETTERS[names.index(c)] for c in w) for w in opts["block_generators"]]
        elif opts.get("n_blocks"):
            req["n_blocks"] = int(opts["n_blocks"])
        try:
            out = r(req)
        except Unsupported as exc:
            if "alias.block.2fis" not in str(exc):
                raise
            out = r({**req, "allow_2fi": True})  # no blocking keeps every two-factor interaction clear
        m = np.array(out["matrix"], float)[:, order]
        if out.get("blocks"):
            blocks = np.array(out["blocks"])
        if design_type == "fractional_factorial":
            gens, rel, res = _relation_from_matrix(m, names)
    elif design_type == "full_factorial":
        m = np.array(r({"kind": "fullmixed", "levels": lv})["matrix"])
        m = np.column_stack([np.linspace(-1, 1, n)[m[:, j]] for j, n in enumerate(lv)])
    elif design_type == "plackett_burman":
        m = np.array(r({"kind": "pb", "k": k, "runs": 4 * ((k + 1 + 3) // 4)})["matrix"], float)
    elif design_type == "box_behnken":
        m, inside = np.array(r({"kind": "bbd", "k": k, "n0": nc})["matrix"], float), True
    elif design_type == "ccd":
        m, alpha, blocks = _r_ccd(k, opts, names, nc)
        inside = True
    elif design_type == "dsd":
        if opts.get("budget"):
            raise Unsupported("harness does not map a DSD budget to DefScreen")
        # process-improve counts centre runs in total, the DSD's own one included
        m, nc = np.array(r({"kind": "dsd", "k": k})["matrix"], float), max(nc - 1, 0)
    elif design_type == "taguchi":
        m = np.array(r({"kind": "oa", "levels": lv})["matrix"])
        m = np.column_stack([np.linspace(-1, 1, n)[m[:, j]] for j, n in enumerate(lv)])
    else:
        raise Unsupported(f"no R package here builds a {design_type!r} design")
    result = build_design_result(
        np.asarray(m, dtype=float),
        factors,
        design_type,
        n_center_points=0 if inside else nc,
        n_replicates=int(opts.get("n_replicates") or 1),
        generators=gens,
        defining_relation=rel,
        resolution=res,
        alpha=alpha,
        randomize=False,
    )
    if blocks is not None:
        result.blocks = blocks.tolist()
        two_level = design_type != "ccd"
        words = _block_words(m, blocks, names, opts.get("block_generators")) if two_level else []
        result.metadata["blocking"] = {"confounded_with": words}
    return result


def r_min_aberration(n_runs: int, k: int) -> list[str]:
    """FrF2's design for a run count: the first of its catalogue, which is ordered by aberration."""
    _BUILT[0], _ENGINE[0] = True, "FrF2"
    m = np.array(r({"kind": "fraction", "k": k, "runs": n_runs})["matrix"], float)
    return _relation_from_matrix(m, list(rc._LETTERS[:k]))[0]


def pydoe3_min_aberration(n_runs: int, k: int) -> list[str]:
    _BUILT[0] = True
    p = k - int(np.log2(n_runs))
    gen = pyDOE3.fracfact_opt(k, p)[0]
    letters = list(rc._LETTERS[:k])
    gens, _, _ = _relation_from_matrix(pyDOE3.fracfact(gen), letters)
    return gens


def unsupported(*_a, **_k):
    _BUILT[0] = True
    raise Unsupported("pyDOE3 has no such construction")


_BUILT = [False]


@contextlib.contextmanager
def built_by(engine: str):
    """Swap the catalogue's design constructors for the other library's, or watch process-improve's."""
    saved = {n: getattr(rc, n) for n in ("generate_design", "minimum_aberration_generators", "generate_omars")}

    def watch(fn):  # noqa: ANN202
        def inner(*a, **k):  # noqa: ANN202
            _BUILT[0] = True
            return fn(*a, **k)

        return inner

    if engine == "other":
        if LIB == "R":
            rc.generate_design, rc.minimum_aberration_generators = r_generate_design, r_min_aberration
        elif LIB == "dexpy":
            rc.generate_design, rc.minimum_aberration_generators = dexpy_generate_design, dexpy_min_aberration
        elif pyDOE3 is None:
            rc.generate_design, rc.minimum_aberration_generators = other_generate_design, unsupported
        else:
            rc.generate_design, rc.minimum_aberration_generators = pydoe3_generate_design, pydoe3_min_aberration
        rc.generate_omars = unsupported
    else:
        for n, fn in saved.items():
            setattr(rc, n, watch(fn))
    rc._made.cache_clear()
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(rc, n, fn)
        rc._made.cache_clear()


def _alarm(*_):  # noqa: ANN202
    raise TimeoutError


def check_row(entry: dict, expected: dict, check: dict) -> tuple[bool, str, bool | None]:
    """Return (a design was built, value found, agrees) for one check."""
    _BUILT[0], _ENGINE[0] = False, LIB
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT_S)
    try:
        found, ok = rc._reproduce(entry, expected, check, rc._Fits())
    except Unsupported as exc:
        found, ok = f"unsupported: {exc}", None
    except TimeoutError:
        found, ok = f"timeout > {TIMEOUT_S} s", None
    except Exception as exc:  # noqa: BLE001
        found, ok = f"{type(exc).__name__}: {str(exc)[:80]}", False
    finally:
        signal.alarm(0)
    return _BUILT[0], found, ok


def builds_a_design(check: dict) -> bool:
    return "kind" in check and ("design" in check or check["kind"] in rc._DESIGN_KINDS)


OUTCOMES = {
    (True, True): "agree",
    (True, False): "other differs",
    (True, None): "other cannot build",
    (False, True): "process-improve differs",
    (False, False): "neither gives the printed number",
    (False, None): "neither gives the printed number",
}


def main() -> None:
    """Check every design-building number with both libraries; write all rows, and the rows that do not agree."""
    entries = [
        e for e in rc.reference_designs() if any(builds_a_design(c) for x in e["expected"] for c in rc._checks(x))
    ]
    sample = entries if N_SAMPLE is None else random.Random(SEED).sample(entries, N_SAMPLE)  # noqa: S311
    rows = []
    for entry in sample:
        for expected in entry["expected"]:
            for check in rc._checks(expected):
                if "kind" not in check:
                    continue
                with built_by("process_improve"):
                    built, pi_found, pi_ok = check_row(entry, expected, check)
                if not built:
                    continue  # read off printed runs, not a construction: nothing for another library to build
                with built_by("other"):
                    _, other_found, other_ok = check_row(entry, expected, check)
                spec = check.get("design") or {"design_type": "fractional_factorial", "k": check.get("k")}
                rows.append(
                    {
                        "source": entry["source"],
                        "entry": entry["name"],
                        "design": f"{spec.get('design_type')} k={spec.get('k')}",
                        "kind": check["kind"] + (f":{check.get('property') or check.get('metric') or ''}"),
                        "printed": str(check.get("printed", expected["value"]))[:40],
                        "process_improve": str(pi_found)[:40],
                        "other": str(other_found)[:110],
                        "built_by": _ENGINE[0],
                        "outcome": OUTCOMES[bool(pi_ok), None if other_ok is None else bool(other_ok)],
                        "catalogue_cause": check.get("cause"),
                    }
                )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"rows_{LIB}_{ARG}.csv", index=False)
    df[df["outcome"] != "agree"].to_csv(OUT / f"differences_{LIB}_{ARG}.csv", index=False)
    df["design_type"] = df["design"].str.split(" ").str[0]
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
    print(f"{LIB}: {df['entry'].nunique()} entries, {len(df)} checks that build a design")
    print(pd.crosstab(df["built_by"], df["outcome"], margins=True).to_string(), "\n")
    print(pd.crosstab(df["design_type"], df["outcome"], margins=True).to_string(), "\n")
    differs = df[df["outcome"].isin(["other differs", "process-improve differs"])]
    print(differs[["entry", "design", "kind", "printed", "process_improve", "other", "outcome"]].to_string())
    cannot = df[df["outcome"] == "other cannot build"]
    print("\nCannot build, by reason:")
    print(cannot.groupby(["design_type", "other"]).size().to_string())


if __name__ == "__main__":
    main()
