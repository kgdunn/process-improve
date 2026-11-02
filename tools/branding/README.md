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
| The browser app's mark, icons and link card, in `web/` | `app_mark.py` | `python tools/branding/app_mark.py web` |

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
- `app_demo.js` drives the live app at <https://factori.al/> with Playwright, or the build
  `APP_URL` names: it runs the worked example (the reactor's yield and the factors that may change
  it, the designs that can learn its curvature, a Box-Behnken design with results simulated from a
  known process, and the analysis beside the truth), capturing Design, the detailed comparison,
  Run and Learn. `PLAYWRIGHT_PROXY`, if set, routes the browser through a proxy.
- `app_mark.py` draws Factorial's mark, the browser app's, from one `generate_design` call: the
  2³ factorial and its centre run, seen down the cube's diagonal. Six corners land on a hexagon;
  the other two fall behind the centre run, which is red as the banner draws centre runs. The
  package's own logo is the same idea in two factors. `tests/web/page/test_brand.py` checks that
  `web/favicon.svg` is what the script draws, and that the page's inline copies agree with it.
