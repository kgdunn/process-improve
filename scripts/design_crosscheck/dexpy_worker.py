"""Build one dexpy design from a JSON request on stdin; print the coded matrix as JSON."""

import json
import sys
import warnings

warnings.simplefilter("ignore")
import dexpy.box_behnken as bb  # noqa: E402
import dexpy.factorial as fac  # noqa: E402
from dexpy import ccd  # noqa: E402

req = json.loads(sys.stdin.read())
kind, k = req["kind"], req["k"]
if kind == "full":
    d = fac.build_full_factorial(k)
elif kind == "fraction":
    d = fac.build_factorial(k, req["runs"])
elif kind == "ccd":
    d = ccd.build_ccd(k, alpha=req["alpha"], center_points=req["centers"])
elif kind == "bbd":
    d = bb.build_box_behnken(k, center_points=req["centers"])
print(
    json.dumps(
        (d.to_numpy(dtype=float) if hasattr(d, "to_numpy") else __import__("numpy").asarray(d, dtype=float)).tolist()
    )
)
