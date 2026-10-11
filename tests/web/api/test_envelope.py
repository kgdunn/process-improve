"""The reply every entry point gives: ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": ...}``, with warnings.

The page reads each reply with ``JSON.parse`` and shows an error or a warning as it is, so
refusals come back as data, and what the library warns about comes back with the result.
"""

from __future__ import annotations

from tests.web._support import factors_named


def test_library_warnings_reach_the_page(api):
    """An even-k DSD with only its own center run leaves the staged analysis no error df, and says so."""
    spec = {"design_type": "dsd", "factors": factors_named("A", "B", "C", "D"), "n_center_points": 1}
    assert any("degrees of freedom" in w for w in api.reply("api_make_design", spec)["warnings"])
    spec["n_center_points"] = 3
    assert api.reply("api_make_design", spec)["warnings"] == []
