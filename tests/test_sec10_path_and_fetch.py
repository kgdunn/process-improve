"""SEC-10: path-traversal guard in the knowledge YAML loader, and clear errors
for the remote sample-dataset fetch.
"""

from __future__ import annotations

import pytest

from process_improve.experiments import datasets
from process_improve.experiments.knowledge import engine


class TestLoadYamlTraversal:
    @pytest.mark.parametrize(
        "bad",
        [
            "../../../etc/passwd",
            "../secrets.yaml",
            "/etc/passwd",
        ],
    )
    def test_traversal_paths_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="escapes the data directory"):
            engine._load_yaml(bad)

    def test_legitimate_filename_loads(self) -> None:
        # A real data file ships with the package and loads as a list.
        result = engine._load_yaml("design_types.yaml")
        assert isinstance(result, list)

    def test_safe_missing_filename_returns_empty(self) -> None:
        assert engine._load_yaml("does_not_exist.yaml") == []


class TestRemoteCsvErrorHandling:
    def test_network_failure_raises_clear_error(self, monkeypatch) -> None:
        def _boom(_url):
            raise OSError("name resolution failed")

        monkeypatch.setattr(datasets.pd, "read_csv", _boom)
        with pytest.raises(RuntimeError, match="Could not download the sample dataset"):
            datasets._read_remote_csv("https://openmv.net/file/distillate-flow.csv")
