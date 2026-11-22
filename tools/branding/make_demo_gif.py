"""Join the frames captured by ``app_demo.js`` into a captioned GIF: design, compare, run, learn, next.

Usage::

    python tools/branding/make_demo_gif.py FRAMES_DIR docs/_static/readme/app-demo.gif
"""

import sys
from pathlib import Path

import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont

WIDTH, BAR = 760, 64
WHITE, SOFT = (255, 255, 255), (242, 242, 240)
# Each stage's color in the app's light theme (--stage-* in web/style.css): Design, Run, Learn and Next.
DESIGN, RUN, LEARN, NEXT = (44, 95, 124), (163, 95, 0), (30, 123, 58), (169, 50, 38)
FONTS = Path(mpl.get_data_path()) / "fonts" / "ttf"

STEPS = [  # frame, stage number, headline, detail, the stage's color, milliseconds on screen
    ("demo-1-plan.png", "1", "Design", "your response, its factors, and the designs that can learn it", DESIGN, 3600),
    ("demo-2-compare.png", "1", "Design", "every design for the brief, on nine measures", DESIGN, 3600),
    ("demo-3-run.png", "2", "Run", "simulate the results from a known process, or type in your own", RUN, 3200),
    ("demo-4-learn.png", "3", "Learn", "fitted in your browser, beside the truth", LEARN, 4200),
    ("demo-5-next.png", "4", "Next", "what the results call for, and the next round", NEXT, 3600),
]


def captioned(shot: Image.Image, number: str, headline: str, detail: str, colour: tuple[int, int, int]) -> Image.Image:
    """Scale the screenshot to the GIF width and add a caption bar below it, in its stage's color."""
    bold = ImageFont.truetype(str(FONTS / "DejaVuSans-Bold.ttf"), 21)
    regular = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), 19)
    shot = shot.convert("RGB").resize((WIDTH, round(shot.height * WIDTH / shot.width)), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (WIDTH, shot.height + BAR), colour)
    frame.paste(shot, (0, 0))
    draw = ImageDraw.Draw(frame)
    y = shot.height + BAR // 2
    draw.ellipse((18, y - 17, 52, y + 17), fill=WHITE)
    draw.text((35, y), number, font=bold, fill=colour, anchor="mm")
    draw.text((66, y), headline, font=bold, fill=WHITE, anchor="lm")
    draw.text((78 + draw.textlength(headline, font=bold), y), detail, font=regular, fill=SOFT, anchor="lm")
    return frame


def main(frames_dir: str, output: str) -> None:
    """Write the GIF, looping forever, with each stage held long enough to read."""
    frames = [
        captioned(Image.open(Path(frames_dir) / name), number, headline, detail, colour).quantize(
            colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        for name, number, headline, detail, colour, _ in STEPS
    ]
    durations = [ms for *_, ms in STEPS]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
