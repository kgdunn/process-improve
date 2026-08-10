#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["process-improve[expt,plotting]"]
# ///
"""Generic dispatcher for the process-improve tool registry.

Every ``@tool_spec``-decorated function in ``process_improve`` is reachable
here by name, with JSON in and JSON out. Prefer this over writing bespoke
analysis code: the registry is versioned, tested, and validated by pydantic
at the boundary, so a malformed call fails loudly instead of silently
producing a plausible number.

Usage
-----
List the tools, optionally filtered by category::

    python doe_tool.py list
    python doe_tool.py list --category experiments

Show one tool's full input schema before calling it::

    python doe_tool.py spec generate_design

Call a tool. The input is JSON, read from a file, from a literal string, or
from stdin::

    python doe_tool.py call generate_design --input spec.json
    python doe_tool.py call doe_knowledge --input '{"query": "resolution IV"}'
    echo '{"query": "aliasing"}' | python doe_tool.py call doe_knowledge

Write the result somewhere instead of stdout::

    python doe_tool.py call generate_design --input spec.json --output design.json

Exit codes
----------
0   the tool ran and returned a result
1   bad usage (unknown tool, unreadable input, malformed JSON)
2   the tool ran but reported an error in its result payload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_USAGE = 1
EXIT_TOOL_ERROR = 2


def _load_registry() -> Any:
    """Import the tool registry, with an actionable message if it is absent."""
    try:
        from process_improve.tool_spec import discover_tools, execute_tool_call, get_tool_specs
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.exit(
            f"Could not import process_improve ({exc}).\n"
            "Install it with:  pip install 'process-improve[expt,plotting]'\n"
            "or run this script with:  uv run --script doe_tool.py ..."
        )
    discover_tools()
    return execute_tool_call, get_tool_specs


def _read_input(raw: str | None) -> dict[str, Any]:
    """Resolve ``--input`` from a path, a literal JSON string, or stdin."""
    if raw is None:
        if sys.stdin.isatty():
            return {}
        raw = sys.stdin.read()
    else:
        candidate = Path(raw)
        # A literal JSON object starts with '{'; anything else is a path.
        if not raw.lstrip().startswith("{"):
            if not candidate.is_file():
                sys.exit(f"Input file not found: {raw}")
            raw = candidate.read_text()

    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"Input is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        sys.exit(f"Input must be a JSON object, got {type(payload).__name__}.")
    return payload


def _emit(payload: Any, output: str | None) -> None:
    """Write the result to a file or to stdout."""
    text = json.dumps(payload, indent=2, default=str)
    if output:
        Path(output).write_text(text)
        print(f"Wrote {output}")
    else:
        print(text)


def cmd_list(args: argparse.Namespace) -> int:
    """Print the registered tools, one per line, with a one-line description."""
    _, get_tool_specs = _load_registry()
    specs = get_tool_specs(category=args.category) if args.category else get_tool_specs()
    if not specs:
        print(f"No tools registered under category {args.category!r}.")
        return EXIT_USAGE
    width = max(len(spec["name"]) for spec in specs)
    for spec in sorted(specs, key=lambda s: s["name"]):
        summary = spec.get("description", "").split(". ")[0].strip()
        print(f"{spec['name']:<{width}}  {summary}")
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    """Print one tool's full Anthropic-format spec, including its JSON schema."""
    _, get_tool_specs = _load_registry()
    for spec in get_tool_specs():
        if spec["name"] == args.tool:
            _emit(spec, args.output)
            return 0
    known = ", ".join(sorted(spec["name"] for spec in get_tool_specs()))
    sys.exit(f"Unknown tool {args.tool!r}.\nAvailable: {known}")


def cmd_call(args: argparse.Namespace) -> int:
    """Execute one tool and print its JSON result."""
    execute_tool_call, get_tool_specs = _load_registry()
    known = {spec["name"] for spec in get_tool_specs()}
    if args.tool not in known:
        sys.exit(f"Unknown tool {args.tool!r}.\nAvailable: {', '.join(sorted(known))}")

    payload = _read_input(args.input)
    try:
        result = execute_tool_call(args.tool, payload)
    except (ValueError, TypeError) as exc:
        # Pydantic validation failures land here. The message names the field,
        # so pass it through verbatim rather than paraphrasing it.
        sys.exit(f"{args.tool} rejected the input:\n{exc}")

    _emit(result, args.output)
    if isinstance(result, dict) and "error" in result:
        return EXIT_TOOL_ERROR
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="doe_tool.py",
        description="Call any process-improve registry tool with JSON in and JSON out.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list the registered tools")
    p_list.add_argument("--category", help="filter by category, e.g. experiments")
    p_list.set_defaults(func=cmd_list)

    p_spec = sub.add_parser("spec", help="print one tool's input schema")
    p_spec.add_argument("tool")
    p_spec.add_argument("--output", help="write to this file instead of stdout")
    p_spec.set_defaults(func=cmd_spec)

    p_call = sub.add_parser("call", help="execute one tool")
    p_call.add_argument("tool")
    p_call.add_argument("--input", help="JSON file path, literal JSON object, or omit to read stdin")
    p_call.add_argument("--output", help="write to this file instead of stdout")
    p_call.set_defaults(func=cmd_call)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
