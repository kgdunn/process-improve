"""Wait until the published browser app is the one built from a given commit.

For a few minutes after a deploy, GitHub Pages can still serve the app before it from its
cache. The app's ``wheels/manifest.json`` names the commit it was built from
(``scripts/build_web_app.py`` writes it), so this reads it, past the cache, until it names the
commit given. It exits 0 then, or 1 when the time allowed runs out. Standard library only.

    python tools/wait_for_web_app.py https://kgdunn.github.io/process-improve/app/ <commit>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

#: How long to wait for the deploy to be served, and how long between two looks (seconds).
TIMEOUT = 900.0
INTERVAL = 15.0


def published_commit(app: str) -> str | None:
    """Return the commit the published app's manifest names, or None while it cannot be read."""
    url = f"{app}wheels/manifest.json?at={time.time_ns()}"  # a new query each time, so no cache answers it
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - the app's own https URL
            return json.load(response).get("commit")
    except (OSError, ValueError):  # not served yet, or not a manifest yet
        return None


def wait(app: str, commit: str, timeout: float = TIMEOUT, interval: float = INTERVAL) -> str | None:
    """Look at the published app until it is the one built from ``commit``; return the last commit seen."""
    deadline = time.monotonic() + timeout
    while True:
        seen = published_commit(app)
        if seen == commit or time.monotonic() >= deadline:
            return seen
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("app", help="the app's URL")
    parser.add_argument("commit", help="the commit the app should be built from")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="seconds to wait (default: %(default)s)")
    args = parser.parse_args()
    app = args.app if args.app.endswith("/") else f"{args.app}/"
    seen = wait(app, args.commit, args.timeout)
    if seen != args.commit:
        print(f"{app} serves the app built from {seen or 'no readable manifest'}, not {args.commit}", file=sys.stderr)
        return 1
    print(f"{app} serves the app built from {args.commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
