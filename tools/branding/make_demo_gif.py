"""Join the frames captured by ``app_demo.js`` into a captioned GIF: design, run, analyse.

Usage::

    python tools/branding/make_demo_gif.py FRAMES_DIR docs/_static/readme/app-demo.gif
"""

import sys
from pathlib import Path

import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont

WIDTH, BAR = 760, 64
NAVY, WHITE, SOFT = (44, 95, 124), (255, 255, 255), (214, 228, 236)
FONTS = Path(mpl.get_data_path()) / "fonts" / "ttf"

STEPS = [  # frame, step number, headline, detail, milliseconds on screen
    ("demo-1-design.png", "1", "Design", "pick factors and a design type", 2800),
    ("demo-2-runs.png", "2", "Run", "download the run sheet, fill in your results", 2800),
    ("demo-3-analysis.png", "3", "Analyse", "upload it: the model is fitted in your browser", 4200),
]


def captioned(shot: Image.Image, number: str, headline: str, detail: str) -> Image.Image:
    """Scale the screenshot to the GIF width and add a navy caption bar below it."""
    bold = ImageFont.truetype(str(FONTS / "DejaVuSans-Bold.ttf"), 21)
    regular = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), 19)
    shot = shot.convert("RGB").resize((WIDTH, round(shot.height * WIDTH / shot.width)), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (WIDTH, shot.height + BAR), NAVY)
    frame.paste(shot, (0, 0))
    draw = ImageDraw.Draw(frame)
    y = shot.height + BAR // 2
    draw.ellipse((18, y - 17, 52, y + 17), fill=WHITE)
    draw.text((35, y), number, font=bold, fill=NAVY, anchor="mm")
    draw.text((66, y), headline, font=bold, fill=WHITE, anchor="lm")
    draw.text((78 + draw.textlength(headline, font=bold), y), detail, font=regular, fill=SOFT, anchor="lm")
    return frame


def main(frames_dir: str, output: str) -> None:
    """Write the GIF, looping forever, with each stage held long enough to read."""
    frames = [
        captioned(Image.open(Path(frames_dir) / name), number, headline, detail).quantize(
            colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        for name, number, headline, detail, _ in STEPS
    ]
    durations = [ms for *_, ms in STEPS]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
