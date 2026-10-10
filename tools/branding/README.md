# Branding images

The images in [`docs/_static/readme/`](../../docs/_static/readme/) are drawn by process-improve
itself, from data that ships with the package, so they can be regenerated after the library
changes. Run the commands from the repository root with the development environment active.

| Image | Script | Command |
| --- | --- | --- |
| `banner-light.png`, `banner-dark.png` | `design_gallery.py` | `python tools/branding/design_gallery.py hero light docs/_static/readme/banner-light.png` (then `dark`) |
| `design-gallery-light.png`, `design-gallery-dark.png` | `design_gallery.py` | `python tools/branding/design_gallery.py gallery light docs/_static/readme/design-gallery-light.png` (then `dark`) |
| `social-preview.png` | `social_card.py` | `python tools/branding/social_card.py tools/branding/social-card-base.png docs/_static/readme/social-preview.png` |
| `app-demo.gif` | `app_demo.js`, `make_demo_gif.py` | `node tools/branding/app_demo.js FRAMES_DIR`, then `python tools/branding/make_demo_gif.py FRAMES_DIR docs/_static/readme/app-demo.gif` |

Notes:

- Every design in the banner and gallery comes from one `generate_design` call. The two colour
  themes were checked for colour-vision-deficiency separation and contrast on GitHub's light and
  dark page colours; the README picks one with a `<picture>` element.
- `social_card.py` starts from `social-card-base.png`, the card uploaded before, and redraws two
  of its six panels: "Designed experiments" (a factorial, the path of steepest ascent from
  `optimize_responses`, and a central composite design that reuses runs) and "PLS with
  prediction intervals" (out-of-sample predictions of pectin yield from FTIR spectra, with the
  intervals of `PLS.prediction_interval`). The card is uploaded by hand under Settings, General,
  Social preview: GitHub has no API for it.
- `app_demo.js` drives the live app at <https://kgdunn.github.io/process-improve/app/> with
  Playwright: it generates a definitive screening design, downloads the run sheet, fills it in
  with `fill_workbook.py`, and uploads it for the analysis. `PYTHON` names the interpreter for
  that step, and `PLAYWRIGHT_PROXY`, if set, routes the browser through a proxy.
