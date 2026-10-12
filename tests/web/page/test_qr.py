"""``web/qr.js``, the run sheet's QR code, held to two independent implementations.

segno, an encoder, draws the same symbol module for module at every version from 1 to 40 and
every error-correction level, when the text fills the version and the mask is fixed: one table
entry wrong, one bit of the format or version information, or one step of the placement, and a
matrix differs. Below capacity segno pads differently from the standard (it adds a zero byte
when the stream is already byte-aligned), so there the padding is checked against the
standard's rule as written here, and, where zxing-cpp is installed, every symbol is decoded back
to its text by a decoder that shares no code with either.
"""

from __future__ import annotations

import pytest

from tests.web._support import javascript, source

segno = pytest.importorskip("segno", reason="segno is the reference the encoder is held to")

LEVELS = "LMQH"


def payload(n: int, seed: int) -> str:
    """Return n bytes of URL-like text, different for each seed."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    return "".join(alphabet[(i * 7 + seed * 13 + i * i) % len(alphabet)] for i in range(n))


def qr(node: str, calls: list[tuple[str, list]]) -> list:
    return javascript(node, source("qr.js"), calls)


def capacities(node: str, level: str) -> list[int]:
    return qr(node, [("capacity", [v, level]) for v in range(1, 41)])


@pytest.mark.parametrize("level", LEVELS)
def test_every_version_draws_the_symbol_segno_draws(node, level):
    held = capacities(node, level)
    texts = [payload(n, v) for v, n in enumerate(held, start=1)]
    drawn = qr(
        node,
        [("qrCode", [text, {"level": level, "version": v, "mask": v % 8}]) for v, text in enumerate(texts, start=1)],
    )
    for v, (text, code) in enumerate(zip(texts, drawn, strict=True), start=1):
        reference = segno.make(text, version=v, error=level.lower(), mask=v % 8, mode="byte", boost_error=False)
        expected = [[bool(m) for m in row] for row in reference.matrix]
        assert code["size"] == len(expected) == 4 * v + 17, (level, v)
        assert code["dark"] == expected, f"version {v}, level {level}"


@pytest.mark.parametrize("level", LEVELS)
def test_each_version_holds_what_segno_says_it_holds(node, level):
    """One byte more than the capacity does not fit: segno refuses it at that version."""
    for v, n in enumerate(capacities(node, level), start=1):
        segno.make(payload(n, v), version=v, error=level.lower(), mode="byte", boost_error=False)
        with pytest.raises(segno.DataOverflowError):
            segno.make(payload(n + 1, v), version=v, error=level.lower(), mode="byte", boost_error=False)


def test_the_smallest_version_that_holds_the_text_is_chosen(node):
    lengths = [1, 14, 15, 100, 700, 1000, 1500, 2300]
    drawn = qr(node, [("qrCode", [payload(n, n), {"level": "M"}]) for n in lengths])
    for n, code in zip(lengths, drawn, strict=True):
        expected = segno.make(payload(n, n), error="m", mode="byte", boost_error=False, micro=False).version
        assert code["version"] == expected, n


def standard_codewords(text: str, version: int, data_codewords: int) -> list[int]:
    """Return the data codewords ISO/IEC 18004 7.4 prescribes for byte mode, written out independently."""
    data = text.encode()
    bits = "0100" + format(len(data), "08b" if version <= 9 else "016b") + "".join(format(b, "08b") for b in data)
    room = data_codewords * 8
    bits += "0" * min(4, room - len(bits))  # the terminator, shortened if the symbol is full
    bits += "0" * (-len(bits) % 8)  # padding bits, only when the stream ends inside a codeword
    pads = ["11101100", "00010001"]
    bits += "".join(pads[i % 2] for i in range((room - len(bits)) // 8))
    return [int(bits[i : i + 8], 2) for i in range(0, room, 8)]


@pytest.mark.parametrize("level", LEVELS)
def test_a_text_shorter_than_the_version_is_padded_as_the_standard_says(node, level):
    held = capacities(node, level)
    cases = [(v, n) for v in (1, 2, 9, 10, 26, 27, 40) for n in (0, 1, held[v - 1] // 2, held[v - 1] - 1)]
    calls = [("dataBits", [list(payload(n, v).encode()), v, level]) for v, n in cases]
    for (v, n), codewords in zip(cases, qr(node, calls), strict=True):
        assert codewords == standard_codewords(payload(n, v), v, len(codewords)), (level, v, n)


def test_every_symbol_decodes_to_its_text(node):
    """A decoder sharing no code with the encoder reads each symbol back, padding and chosen mask included."""
    zxingcpp = pytest.importorskip("zxingcpp", reason="zxing-cpp decodes the symbols")
    image = pytest.importorskip("PIL.Image")
    cases = [(level, n) for level in LEVELS for n in (1, 17, 60, 300, 900)]
    cases = [(level, n) for level, n in cases if n <= capacities(node, level)[39]]
    drawn = qr(node, [("qrCode", [payload(n, n), {"level": level}]) for level, n in cases])
    for (level, n), code in zip(cases, drawn, strict=True):
        scale, quiet = 4, 4
        side = (code["size"] + 2 * quiet) * scale
        picture = image.new("L", (side, side), 255)
        for y, row in enumerate(code["dark"]):
            for x, dark in enumerate(row):
                if dark:
                    left, top = (x + quiet) * scale, (y + quiet) * scale
                    picture.paste(0, (left, top, left + scale, top + scale))
        read = zxingcpp.read_barcodes(picture)
        assert [r.text for r in read] == [payload(n, n)], (level, n)
