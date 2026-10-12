// A QR code for the printed run sheet: scanned at the plant, it opens the design on a phone, where
// the results can be typed in and sent back as a link. Byte mode, versions 1 to 40, error
// correction L, M, Q or H, the mask with the lowest penalty, as ISO/IEC 18004 sets them out.
// Nothing here touches the document; tests/web/page/test_qr.py holds every version and level to
// segno, an independent encoder, module for module.

/** Codewords of error correction per block, by level and version (index 0 unused). */
const ECC_PER_BLOCK = {
  L: [0, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
  M: [0, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
  Q: [0, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
  H: [0, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
};

/** Blocks of error correction, by level and version (index 0 unused). */
const BLOCKS = {
  L: [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
  M: [0, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
  Q: [0, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
  H: [0, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81],
};

/** The two bits each level writes in the format information. */
const FORMAT_BITS = { L: 1, M: 0, Q: 3, H: 2 };

/** The modules a version leaves for data and error correction, once its function patterns are drawn. */
function rawModules(version) {
  let n = (16 * version + 128) * version + 64;
  if (version >= 2) {
    const align = Math.floor(version / 7) + 2;
    n -= (25 * align - 10) * align - 55;
    if (version >= 7) n -= 36;
  }
  return n;
}

/** The data codewords a version holds at a level. */
const dataCodewords = (version, level) =>
  Math.floor(rawModules(version) / 8) - ECC_PER_BLOCK[level][version] * BLOCKS[level][version];

/** The most bytes a QR code of this version and level holds in byte mode. */
export function capacity(version, level) {
  const countBits = version <= 9 ? 8 : 16;
  return Math.floor((dataCodewords(version, level) * 8 - 4 - countBits) / 8);
}

// ---------------------------------------------------------------- Reed-Solomon over GF(256), x^8 + x^4 + x^3 + x^2 + 1
function multiply(x, y) {
  let z = 0;
  for (let i = 7; i >= 0; i--) {
    z = (z << 1) ^ ((z >>> 7) * 0x11d);
    z ^= ((y >>> i) & 1) * x;
  }
  return z;
}

/** The generator polynomial of a degree, its leading 1 left out. */
function divisor(degree) {
  const result = new Array(degree).fill(0);
  result[degree - 1] = 1;
  let root = 1;
  for (let i = 0; i < degree; i++) {
    for (let j = 0; j < degree; j++) {
      result[j] = multiply(result[j], root);
      if (j + 1 < degree) result[j] ^= result[j + 1];
    }
    root = multiply(root, 0x02);
  }
  return result;
}

function remainder(data, generator) {
  const result = generator.map(() => 0);
  for (const byte of data) {
    const factor = byte ^ result.shift();
    result.push(0);
    generator.forEach((coefficient, i) => {
      result[i] ^= multiply(coefficient, factor);
    });
  }
  return result;
}

// ---------------------------------------------------------------- the codewords
/** The data codewords: mode, count, the bytes, a terminator and padding (exported for the tests). */
export function dataBits(bytes, version, level) {
  const bits = [];
  const put = (value, length) => {
    for (let i = length - 1; i >= 0; i--) bits.push((value >>> i) & 1);
  };
  put(0b0100, 4);
  put(bytes.length, version <= 9 ? 8 : 16);
  for (const b of bytes) put(b, 8);
  const room = dataCodewords(version, level) * 8;
  put(0, Math.min(4, room - bits.length));
  put(0, (8 - (bits.length % 8)) % 8);
  for (let pad = 0xec; bits.length < room; pad ^= 0xec ^ 0x11) put(pad, 8);
  const codewords = [];
  for (let i = 0; i < bits.length; i += 8) codewords.push(bits.slice(i, i + 8).reduce((a, b) => (a << 1) | b, 0));
  return codewords;
}

/** Split the data into blocks, add each block's error correction, and interleave them. */
function withCorrection(data, version, level) {
  const blocks = BLOCKS[level][version];
  const ecc = ECC_PER_BLOCK[level][version];
  const raw = Math.floor(rawModules(version) / 8);
  const short = blocks - (raw % blocks);
  const shortLength = Math.floor(raw / blocks);
  const generator = divisor(ecc);
  const parts = [];
  for (let i = 0, at = 0; i < blocks; i++) {
    const length = shortLength - ecc + (i < short ? 0 : 1);
    const block = data.slice(at, at + length);
    at += length;
    parts.push({ data: block, ecc: remainder(block, generator) });
  }
  const out = [];
  for (let i = 0; i <= shortLength - ecc; i++) {
    for (const part of parts) if (i < part.data.length) out.push(part.data[i]);
  }
  for (let i = 0; i < ecc; i++) for (const part of parts) out.push(part.ecc[i]);
  return out;
}

// ---------------------------------------------------------------- the matrix
function alignmentPositions(version) {
  if (version === 1) return [];
  const count = Math.floor(version / 7) + 2;
  const step = version === 32 ? 26 : Math.ceil((version * 4 + 4) / (count * 2 - 2)) * 2;
  const positions = [6];
  for (let pos = version * 4 + 10; positions.length < count; pos -= step) positions.splice(1, 0, pos);
  return positions;
}

/** The format information: level and mask, BCH-coded and masked as the standard sets it. */
function formatBits(level, mask) {
  const data = (FORMAT_BITS[level] << 3) | mask;
  let rem = data;
  for (let i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >>> 9) * 0x537);
  return ((data << 10) | rem) ^ 0x5412;
}

function versionBits(version) {
  let rem = version;
  for (let i = 0; i < 12; i++) rem = (rem << 1) ^ ((rem >>> 11) * 0x1f25);
  return (version << 12) | rem;
}

const MASKS = [
  (x, y) => (x + y) % 2 === 0,
  (x, y) => y % 2 === 0,
  (x) => x % 3 === 0,
  (x, y) => (x + y) % 3 === 0,
  (x, y) => (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0,
  (x, y) => ((x * y) % 2) + ((x * y) % 3) === 0,
  (x, y) => (((x * y) % 2) + ((x * y) % 3)) % 2 === 0,
  (x, y) => (((x + y) % 2) + ((x * y) % 3)) % 2 === 0,
];

class Matrix {
  constructor(version) {
    this.size = version * 4 + 17;
    this.dark = Array.from({ length: this.size }, () => new Array(this.size).fill(false));
    this.fixed = Array.from({ length: this.size }, () => new Array(this.size).fill(false));
    this.drawFunctionPatterns(version);
  }

  set(x, y, dark) {
    this.dark[y][x] = dark;
    this.fixed[y][x] = true;
  }

  drawFunctionPatterns(version) {
    const { size } = this;
    for (let i = 0; i < size; i++) {
      this.set(6, i, i % 2 === 0);
      this.set(i, 6, i % 2 === 0);
    }
    for (const [x, y] of [
      [3, 3],
      [size - 4, 3],
      [3, size - 4],
    ]) {
      for (let dy = -4; dy <= 4; dy++) {
        for (let dx = -4; dx <= 4; dx++) {
          const d = Math.max(Math.abs(dx), Math.abs(dy));
          if (x + dx >= 0 && x + dx < size && y + dy >= 0 && y + dy < size) this.set(x + dx, y + dy, d !== 2 && d !== 4);
        }
      }
    }
    const positions = alignmentPositions(version);
    const last = positions.length - 1;
    positions.forEach((px, i) => {
      positions.forEach((py, j) => {
        if ((i === 0 && j === 0) || (i === 0 && j === last) || (i === last && j === 0)) return;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) this.set(px + dx, py + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
        }
      });
    });
    this.drawFormat(0, "L"); // reserves the format areas; drawn for real once the mask is chosen
    if (version >= 7) {
      const bits = versionBits(version);
      for (let i = 0; i < 18; i++) {
        const dark = ((bits >>> i) & 1) === 1;
        const a = size - 11 + (i % 3);
        const b = Math.floor(i / 3);
        this.set(a, b, dark);
        this.set(b, a, dark);
      }
    }
  }

  drawFormat(mask, level) {
    const bits = formatBits(level, mask);
    const bit = (i) => ((bits >>> i) & 1) === 1;
    const { size } = this;
    for (let i = 0; i <= 5; i++) this.set(8, i, bit(i));
    this.set(8, 7, bit(6));
    this.set(8, 8, bit(7));
    this.set(7, 8, bit(8));
    for (let i = 9; i < 15; i++) this.set(14 - i, 8, bit(i));
    for (let i = 0; i < 8; i++) this.set(size - 1 - i, 8, bit(i));
    for (let i = 8; i < 15; i++) this.set(8, size - 15 + i, bit(i));
    this.set(8, size - 8, true);
  }

  /** Place the codewords in the zigzag the standard reads them in, two columns at a time from the right. */
  drawCodewords(codewords) {
    const { size } = this;
    let i = 0;
    for (let right = size - 1; right >= 1; right -= 2) {
      if (right === 6) right = 5;
      for (let vert = 0; vert < size; vert++) {
        for (let j = 0; j < 2; j++) {
          const x = right - j;
          const upward = ((right + 1) & 2) === 0;
          const y = upward ? size - 1 - vert : vert;
          if (!this.fixed[y][x] && i < codewords.length * 8) {
            this.dark[y][x] = ((codewords[i >>> 3] >>> (7 - (i & 7))) & 1) === 1;
            i++;
          }
        }
      }
    }
  }

  applyMask(mask) {
    const test = MASKS[mask];
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) if (!this.fixed[y][x] && test(x, y)) this.dark[y][x] = !this.dark[y][x];
    }
  }

  /** The standard's penalty: runs of one colour, 2x2 blocks, finder-like patterns, and imbalance of dark and light. */
  penalty() {
    const { size, dark } = this;
    let score = 0;
    const line = (get) => {
      let total = 0;
      let run = 1;
      for (let i = 1; i <= size; i++) {
        if (i < size && get(i) === get(i - 1)) run++;
        else {
          if (run >= 5) total += 3 + run - 5;
          run = 1;
        }
      }
      const pattern = [true, false, true, true, true, false, true];
      for (let i = 0; i + 7 <= size; i++) {
        if (!pattern.every((p, k) => get(i + k) === p)) continue;
        const lightBefore = i >= 4 && [1, 2, 3, 4].every((k) => !get(i - k));
        const lightAfter = i + 11 <= size && [7, 8, 9, 10].every((k) => !get(i + k));
        if (lightBefore) total += 40;
        if (lightAfter) total += 40;
      }
      return total;
    };
    for (let y = 0; y < size; y++) score += line((x) => dark[y][x]);
    for (let x = 0; x < size; x++) score += line((y) => dark[y][x]);
    for (let y = 0; y + 1 < size; y++) {
      for (let x = 0; x + 1 < size; x++) {
        const c = dark[y][x];
        if (c === dark[y][x + 1] && c === dark[y + 1][x] && c === dark[y + 1][x + 1]) score += 3;
      }
    }
    const darkCount = dark.reduce((n, row) => n + row.filter(Boolean).length, 0);
    const percent = (darkCount * 100) / (size * size);
    score += 10 * Math.floor(Math.abs(percent - 50) / 5);
    return score;
  }
}

/**
 * Encode text as a QR code, in byte mode, in the smallest version that holds it at ``level``.
 * ``version`` and ``mask`` may be given, as the tests do; otherwise the mask with the lowest
 * penalty is chosen. Returns {version, size, dark}, dark a list of rows of booleans, or throws
 * when the text is too long for any version.
 */
export function qrCode(text, { level = "M", version = null, mask = null } = {}) {
  const bytes = new TextEncoder().encode(text);
  let v = version;
  if (v === null) {
    v = 1;
    while (v <= 40 && capacity(v, level) < bytes.length) v++;
  }
  if (v > 40 || capacity(v, level) < bytes.length) throw new Error(`${bytes.length} bytes are too many for a QR code.`);
  const codewords = withCorrection(dataBits(bytes, v, level), v, level);
  const masks = mask === null ? [0, 1, 2, 3, 4, 5, 6, 7] : [mask];
  let best = null;
  for (const m of masks) {
    const matrix = new Matrix(v);
    matrix.drawCodewords(codewords);
    matrix.applyMask(m);
    matrix.drawFormat(m, level);
    const score = masks.length > 1 ? matrix.penalty() : 0;
    if (best === null || score < best.score) best = { score, matrix };
  }
  return { version: v, size: best.matrix.size, dark: best.matrix.dark };
}

/** Draw a QR code as SVG, one path, with the four-module quiet zone the standard asks for. */
export function qrSvg(code, { quiet = 4 } = {}) {
  const total = code.size + 2 * quiet;
  const path = [];
  code.dark.forEach((row, y) => {
    row.forEach((dark, x) => {
      if (dark) path.push(`M${x + quiet} ${y + quiet}h1v1h-1z`);
    });
  });
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${total} ${total}" shape-rendering="crispEdges">` +
    `<rect width="${total}" height="${total}" fill="#fff"/><path d="${path.join("")}" fill="#000"/></svg>`
  );
}
