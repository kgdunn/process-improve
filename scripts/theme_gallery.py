"""Render the four Plotly base themes side by side for visual assessment.

Fits a small PCA on the bundled Kamyr dataset and renders a score plot, an
SPE plot and a loadings plot under each registered theme
(``pi_tufte``, ``pi_economist``, ``pi_journal``, ``pi_brand``).

Usage
-----
    python scripts/theme_gallery.py [output_dir]

Writes ``index.html`` (always) and one PNG per theme-plot combination (when
the optional ``kaleido`` package is installed) into ``output_dir`` - by
default ``./theme_gallery``.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

from process_improve.multivariate.methods import PCA, MCUVScaler
from process_improve.multivariate.plots import loading_plot, score_plot, spe_plot
from process_improve.visualization.themes import THEME_NAMES

PLOTS = ("score", "spe", "loading")


def _fit_model() -> PCA:
    folder = pathlib.Path(__file__).resolve().parents[1] / "process_improve" / "datasets" / "multivariate"
    raw = pd.read_csv(folder / "kamyr.csv", index_col=None, header=None)
    x_scaled = MCUVScaler().fit_transform(raw)
    return PCA(n_components=3).fit(x_scaled)


def _render(model: PCA, theme: str, plot: str):
    settings = {"template": theme}
    if plot == "score":
        return score_plot(model, settings=settings)
    if plot == "spe":
        return spe_plot(model, settings=settings)
    return loading_plot(model, settings=settings)


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("theme_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _fit_model()
    sections: list[str] = []
    png_count = 0
    for theme in THEME_NAMES:
        blocks: list[str] = []
        for plot in PLOTS:
            fig = _render(model, theme, plot)
            blocks.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            try:
                fig.write_image(out_dir / f"{theme}_{plot}.png", scale=2)
                png_count += 1
            except Exception as exc:  # noqa: BLE001 - kaleido is optional
                print(f"  (PNG skipped for {theme}/{plot}: {exc})")
        sections.append(f"<h2>{theme}</h2>\n" + "\n".join(blocks))
        print(f"Rendered theme: {theme}")

    index = out_dir / "index.html"
    index.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>process-improve theme gallery</title></head><body>"
        "<h1>process-improve Plotly theme gallery</h1>\n" + "\n".join(sections) + "</body></html>",
        encoding="utf-8",
    )
    print(f"\nWrote {index}")
    print(f"Wrote {png_count} PNG file(s) to {out_dir}")


if __name__ == "__main__":
    main()
