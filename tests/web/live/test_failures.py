"""What the page says when PyPI is refused: Pyodide starts, and the packages it installs from PyPI do not.

A network that reaches Pyodide's CDN but not PyPI stops Python inside micropip, whose error is
a Python traceback. The status line gives the reason in one line, and the page stops waiting.
"""

from __future__ import annotations

import re

from tests.web._browser import BOOT_TIMEOUT, Console
from tests.web.e2e._page import status


def test_a_refused_package_index_is_reported_in_one_line(own_context, app_url):
    own_context.route(
        re.compile(r"^https://(pypi\.org|files\.pythonhosted\.org)/"), lambda route: route.abort("blockedbyclient")
    )
    console = Console()
    console.allow(r"ERR_BLOCKED_BY_CLIENT")  # the browser reports each refused download
    page = own_context.new_page()
    console.watch(page)
    page.goto(app_url)
    page.wait_for_function("document.querySelector('#status').dataset.state === 'error'", timeout=BOOT_TIMEOUT)
    text, _ = status(page)
    assert text.startswith("Could not start Python: "), text
    assert "\n" not in text, "one line, not a traceback"
    assert page.is_disabled("#run-example")
    assert not console.unexpected()
