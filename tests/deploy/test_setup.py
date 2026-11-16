"""``deploy/factorial/setup.sh``: what it says is left to do for Caddy, once it has installed the server's files.

The script itself needs root and a server. Its last words come from one function, ``caddy_next``, which
reads the Caddyfile it is given: these tests run that function, as the script wrote it, on Caddyfiles
of their own.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "setup.sh"
IMPORT = "import /etc/caddy/factorial.caddy\n"

SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(SH is None, reason="the server's setup script is run by sh")


def said(tmp_path: Path, caddyfile: str | None, block: str) -> str:
    """Run ``caddy_next`` on a Caddyfile holding this text (None: there is no file), and return what it says."""
    setup = SETUP.read_text(encoding="utf-8")
    function = re.search(r"^caddy_next\(\) \{\n.+?^\}\n", setup, flags=re.MULTILINE | re.DOTALL)[0]
    path = tmp_path / "Caddyfile"
    if caddyfile is not None:
        path.write_text(caddyfile, encoding="utf-8")
    script = f'set -eu\n{function}caddy_next "$1" "$2"\n'
    command = [SH, "-c", script, "sh", str(path), block]
    done = subprocess.run(command, capture_output=True, encoding="utf-8", check=True)  # noqa: S603
    return done.stdout.replace(str(path), "Caddyfile")


def test_a_first_setup_says_what_to_put_in_the_caddyfile_and_to_reload(tmp_path):
    for caddyfile in (None, "factori.al {\n\treverse_proxy localhost:8000\n}\n"):
        words = said(tmp_path, caddyfile, "changed")
        assert "replace the whole factori.al { ... } block" in words
        assert words.endswith("sudo caddy validate --config Caddyfile && sudo systemctl reload caddy\n")


@pytest.mark.parametrize("off", ["{\n\tadmin off\n}\n", "{\n    admin   off\n}\n"])
def test_a_caddyfile_that_turns_the_admin_endpoint_off_is_restarted(tmp_path, off):
    """A reload asks Caddy through that endpoint, and fails without it; the site's other servers stay up."""
    words = said(tmp_path, off + IMPORT, "changed")
    assert "sudo systemctl restart caddy\n" in words
    assert "reload" not in words
    assert "the server's other sites drop for a second" in words


def test_an_admin_endpoint_only_named_in_a_comment_is_reloaded(tmp_path):
    commented = "# admin off, once\n{\n\tadmin localhost:2019\n}\n"
    assert "systemctl reload caddy" in said(tmp_path, commented + IMPORT, "changed")


def test_run_again_with_the_same_block_it_asks_nothing_of_caddy(tmp_path):
    """The usage report or the relay is what changed: Caddy reads nothing new, so it is left alone."""
    words = said(tmp_path, "{\n\tadmin off\n}\n" + IMPORT, "same")
    assert words == "Caddy already serves the site with this block: nothing to do there.\n"


def test_run_again_with_a_new_block_it_names_the_command_and_no_edit(tmp_path):
    words = said(tmp_path, IMPORT, "changed")
    assert "replace the whole" not in words
    assert words.endswith("sudo systemctl reload caddy\n")


def test_the_script_compares_the_block_before_it_installs_it():
    setup = SETUP.read_text(encoding="utf-8")
    steps = [
        'cmp -s "$here/factorial.caddy"',
        'install -o root -g root -m 644 "$here/factorial.caddy"',
        'caddy_next /etc/caddy/Caddyfile "$block"',
    ]
    places = [setup.index(step) for step in steps]
    assert places == sorted(places)
