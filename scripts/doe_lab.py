"""A blind laboratory for evaluating an agent that designs experiments: it makes runs and keeps its process to itself.

The lab holds one benchmark case (see :mod:`process_improve.experiments.benchmark`) in memory
and answers over HTTP on localhost, so the agent under test learns the process only through the
runs it pays for, as at a real bench. Every request is logged; ``score`` reads the log afterwards
and says how the agent did.

Usage::

    uv run python scripts/doe_lab.py serve --case "adhesion, DSD" --port 8765 --log lab.jsonl
    uv run python scripts/doe_lab.py score lab.jsonl
    uv run python scripts/doe_lab.py score runs/*/lab.jsonl     # one row per log

Endpoints (JSON in and out):

``GET /problem``
    The factors (name, low, high, units, description), the response, the goal, the run budget
    and the runs used so far.
``POST /runs``
    ``{"runs": [{"<factor>": value, ...}, ...]}``: the settings in the factors' own units, each
    inside its range. Returns the same runs with the measured response. Runs outside a range,
    or beyond the budget, are refused and not charged.
``GET /history``
    Every run made so far, in order.
``POST /answer``
    ``{"settings": {...}, "predicted": number, "summary": "..."}``: the agent's final answer.
    Only the last answer counts. The lab does not say how good it is.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_improve.experiments.benchmark import CASES, BenchmarkCase
from process_improve.experiments.simulate import _loss, _true_best

#: What the agent is told about each case's factors and response, beyond their names and ranges.
DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "adhesion, DSD": {
        "response": "Adhesion of a coating [arbitrary units]",
        "factors": {name: f"Process factor {name}, coded from -1 to +1" for name in "ABCDEFGH"},
    },
    "hardness (02), hypothesised": {
        "response": "Koenig pendulum hardness of a cured 2K polyurethane clearcoat [s]",
        "factors": {
            "X01": "NCO:OH index [-]",
            "X02": "Polyol OH number of the blend [mg KOH/g]",
            "X03": "Catalyst, DBTDL [ppm]",
            "X04": "Flow additive [%]",
            "X05": "Matting agent [%]",
            "X06": "Solvent dilution [%]",
            "X07": "Flash-off time [min]",
            "X08": "Cure temperature [degC]",
            "X09": "Cure time [min]",
            "X10": "Dry film thickness [um]",
            "X11": "Booth humidity [% r.h.]",
            "X12": "Substrate temperature [degC]",
        },
    },
    "impact (03), hypothesised": {
        "response": "Reverse impact to first crack of a coil coating [in-lb]",
        "factors": {
            "X01": "Binder blend ratio, soft share [%]",
            "X02": "Elastomer modifier [%]",
            "X03": "Plasticizer [%]",
            "X04": "Silane coupling agent [%]",
            "X05": "Filler median particle size [um]",
            "X06": "Pigment volume concentration [%]",
            "X07": "Dispersant [%]",
            "X08": "Dry film thickness [um]",
            "X09": "Peak metal temperature [degC]",
            "X10": "Oven dwell time [s]",
        },
    },
}


class Lab:
    """One case, its run budget, and the log of everything asked of it."""

    def __init__(self, case: BenchmarkCase, seed: int, budget: int, log: Path) -> None:
        self.case, self.budget, self.log = case, budget, log
        self.rng = np.random.default_rng(seed)
        self.runs: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self._write({"event": "start", "case": case.name, "seed": seed, "budget": budget})

    def _write(self, record: dict[str, Any]) -> None:
        with self.log.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"time": time.time(), **record}) + "\n")

    def problem(self) -> dict[str, Any]:
        """Return what the agent may know: the factors and their ranges, the response, the goal, the budget."""
        described = DESCRIPTIONS.get(self.case.name, {})
        return {
            "factors": [
                {
                    "name": f.name,
                    "low": f.low,
                    "high": f.high,
                    "units": f.units,
                    "description": described.get("factors", {}).get(f.name, ""),
                }
                for f in self.case.factors
            ],
            "response": described.get("response", "y"),
            "goal": self.case.goal,
            "target": self.case.target,
            "settings_must_stay_inside_the_ranges": True,
            "budget": self.budget,
            "runs_used": len(self.runs),
        }

    def make(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Measure ``runs`` on the process and charge them, or refuse them all and charge nothing."""
        names = [f.name for f in self.case.factors]
        try:
            frame = pd.DataFrame(runs)[names].astype(float)
        except (KeyError, ValueError, TypeError) as exc:
            return {"error": f"Each run needs a number for every factor {names}: {exc}"}
        outside = [
            f"{f.name}={v:g}"
            for f in self.case.factors
            for v in frame[f.name]
            if not (f.low - 1e-9 <= v <= f.high + 1e-9)  # type: ignore[operator]
        ]
        if outside or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
            return {"error": f"Every setting must be a number inside its range; refused: {outside[:5]}"}
        with self.lock:
            if len(self.runs) + len(frame) > self.budget:
                return {"error": f"Only {self.budget - len(self.runs)} runs are left of the budget of {self.budget}."}
            y = self.case.truth.simulate(self.case.coded(frame), random_state=self.rng).to_numpy()
            made = [{**row, "y": round(float(v), 3)} for row, v in zip(frame.to_dict(orient="records"), y, strict=True)]
            self.runs.extend(made)
            self._write({"event": "runs", "runs": made})
        return {"runs": made, "runs_used": len(self.runs), "runs_left": self.budget - len(self.runs)}

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        """Log the final answer, without saying how good it is."""
        self._write({"event": "answer", **body})
        return {"received": True}


def _handler(lab: Lab) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:  # the log file records the requests
            return

        def do_GET(self) -> None:
            if self.path == "/problem":
                self._send(lab.problem())
            elif self.path == "/history":
                self._send({"runs": lab.runs})
            else:
                self._send({"error": "unknown path"}, 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send({"error": "send JSON"}, 400)
                return
            if self.path == "/runs":
                result = lab.make(body.get("runs", []))
                self._send(result, 400 if "error" in result else 200)
            elif self.path == "/answer":
                self._send(lab.answer(body))
            else:
                self._send({"error": "unknown path"}, 404)

    return Handler


def score(log: Path) -> dict[str, Any]:
    """Return how the agent in a lab log did: runs used, its answer's true response, and the shortfall."""
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(r for r in records if r["event"] == "start")
    case = next(c for c in CASES if c.name == start["case"])
    runs = sum(len(r["runs"]) for r in records if r["event"] == "runs")
    answers = [r for r in records if r["event"] == "answer"]
    names = [f.name for f in case.factors]
    best = _true_best(case.truth, names, case.goal, case.target)
    out: dict[str, Any] = {
        "case": case.name,
        "runs_used": runs,
        "true_best": best["response"],
        "answered": bool(answers),
    }
    if answers:
        settings = pd.DataFrame([answers[-1]["settings"]])[names].astype(float)
        inside = all(f.low - 1e-9 <= settings[f.name].iloc[0] <= f.high + 1e-9 for f in case.factors)  # type: ignore[operator]
        value = float(case.truth.mean(case.coded(settings))[0])
        short = max(float(_loss(np.array([value]), case.goal, case.target)[0]) - best["loss"], 0.0)
        out |= {
            "inside_ranges": inside,
            "true_value": value,
            "predicted": answers[-1].get("predicted"),
            "shortfall": short,
            "shortfall_in_sigma": short / case.truth.sigma,
        }
    else:
        out |= {"shortfall_in_sigma": math.nan}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="hold a case and answer runs on localhost")
    serve.add_argument("--case", required=True, help="a benchmark case name")
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--budget", type=int, default=150)
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--log", type=Path, required=True)
    scored = sub.add_parser("score", help="score lab logs: one in full, several as a table")
    scored.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "score":
        if len(args.logs) == 1:
            print(json.dumps(score(args.logs[0]), indent=2))
        else:
            table = pd.DataFrame([{"log": str(log.parent), **score(log)} for log in args.logs])
            with pd.option_context("display.width", 200, "display.precision", 3):
                print(table[["log", "case", "runs_used", "answered", "shortfall_in_sigma"]].to_string(index=False))
        return 0
    case = next((c for c in CASES if c.name == args.case), None)
    if case is None:
        parser.error(f"no case {args.case!r}; the cases are: {', '.join(c.name for c in CASES)}")
    lab = Lab(case, args.seed, args.budget, args.log)
    ThreadingHTTPServer(("127.0.0.1", args.port), _handler(lab)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
