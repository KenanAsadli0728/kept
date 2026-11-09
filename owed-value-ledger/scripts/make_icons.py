"""Generate the app icons (pure Python, no Pillow needed).  python scripts/make_icons.py
Design: teal tile, white ring (a ledger 'coin') with a dot - what the brand owes, returned."""
import math
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "ovl" / "static" / "icons"
TEAL = (15, 107, 92)
WHITE = (255, 255, 255)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def render(size, maskable):
    """RGBA bytes. maskable=True: full-bleed square with content inside the 80% safe zone."""
    s = size
    cx = cy = s / 2
    scale = 0.80 if maskable else 1.0
    ring_r, ring_w, dot_r = 0.30 * s * scale, 0.075 * s * scale, 0.11 * s * scale
    corner = 0.0 if maskable else 0.22 * s
    rows = []
    for y in range(s):
        row = bytearray([0])
        for x in range(s):
            px, py = x + 0.5, y + 0.5
            # rounded-square coverage (anti-aliased)
            if corner:
                qx, qy = abs(px - cx) - (cx - corner), abs(py - cy) - (cy - corner)
                d = math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - corner
                a_tile = clamp(0.5 - d)
            else:
                a_tile = 1.0
            r = math.hypot(px - cx, py - cy)
            ring = clamp(0.5 - (abs(r - ring_r) - ring_w / 2))
            dot = clamp(0.5 - (r - dot_r))
            w = max(ring, dot)
            col = tuple(round(TEAL[i] * (1 - w) + WHITE[i] * w) for i in range(3))
            row += bytes((*col, round(255 * a_tile)))
        rows.append(bytes(row))
    return b"".join(rows)


def png(size, maskable=False):
    raw = render(size, maskable)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" rx="113" fill="#0f6b5c"/>
<circle cx="256" cy="256" r="154" fill="none" stroke="#fff" stroke-width="38.4"/><circle cx="256" cy="256" r="56" fill="#fff"/></svg>
"""

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, size, mk in (("icon-192.png", 192, False), ("icon-512.png", 512, False), ("icon-maskable-512.png", 512, True), ("apple-touch-icon.png", 180, False)):
        (OUT / name).write_bytes(png(size, mk))
        print("wrote", name)
    (OUT / "icon.svg").write_text(SVG, encoding="utf8")
    print("wrote icon.svg")
