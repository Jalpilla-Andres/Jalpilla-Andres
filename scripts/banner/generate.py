#!/usr/bin/env python3
"""Generate Andres Jalpilla's animated GitHub profile banner.

The portrait is converted to a 1-bit dither and morphed through Java, code,
and cybersecurity/network silhouettes using SVG SMIL animations.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/andres-cutout.png"
ASSETS = ROOT / "assets"
DATA = ROOT / "scripts/banner/data"

W, H = 1180, 610
LOOP_SECONDS = 15.0
INTRO_SECONDS = 2.8
TRAVELLER_COUNT = 1050
SEED = 20260921

THEMES = {
    "dark": {
        "bg": "#070B14",
        "panel": "#0D1322",
        "panel2": "#101B30",
        "line": "#25344C",
        "muted": "#7D8CA3",
        "text": "#E6EDF3",
        "portrait": "#67D5FF",
        "chrome": "#38BDF8",
        "accent": "#8B5CF6",
        "green": "#22C55E",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#EDF3F7",
        "line": "#CBD7E1",
        "muted": "#687386",
        "text": "#172033",
        "portrait": "#31547A",
        "chrome": "#0284C7",
        "accent": "#7C3AED",
        "green": "#16A34A",
        "shadow": "#AAB7C4",
    },
}


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        ltr = y % 2 == 0
        xs = range(width) if ltr else range(width - 1, -1, -1)
        step = 1 if ltr else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + step
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - step < width:
                    work[y + 1, x - step] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    image = Image.open(SOURCE).convert("RGBA")
    # Tight portrait crop: keeps hair, face, tie and shoulders while dropping most backdrop.
    crop = image.crop((320, 80, 1210, 1170)).resize((300, 340), Image.Resampling.LANCZOS)
    rgba = np.asarray(crop)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    lum = np.asarray(ImageOps.grayscale(crop.convert("RGB")), dtype=np.float32)
    # Preserve the face silhouette and sharpen the suit/tie edges before dithering.
    subject_mask = Image.fromarray(np.uint8(alpha > 0.08) * 255, "L")
    prepared = ImageOps.autocontrast(Image.fromarray(np.uint8(lum)), cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.7)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=175, threshold=2))
    bits = floyd_steinberg(np.asarray(prepared))

    # Dark theme = luminous pixels on dark panel; light theme = ink pixels on light panel.
    active = bits if theme == "dark" else ~bits
    active &= alpha > 0.15

    ys, xs = np.where(active)
    points = np.column_stack((75 + xs, 155 + ys)).astype(np.float32)
    if len(points) > 10000:
        # Keep deterministic density while preserving facial detail.
        points = points[rng.choice(len(points), 10000, replace=False)]
    return points


def bitmap_points(rows: list[str], x0: float, y0: float, scale: float = 1.0) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell not in ("0", " "):
                pts.append((x0 + x * scale, y0 + y * scale))
    return np.asarray(pts, dtype=np.float32)


def logo_points(kind: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "java":
        # 5x7 bitmap wordmark, scaled into the visual frame.
        glyphs = {
            "J": ["11111", "00100", "00100", "00100", "10100", "10100", "01100"],
            "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
            "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
        }
        rows = [""] * 7
        for gi, ch in enumerate("JAVA"):
            g = glyphs[ch]
            rows = [r + g[i] + "0" for i, r in enumerate(rows)]
        target = bitmap_points(rows, 92, 218, 7.0)
        # Add a simple coffee-cup curve underneath for personality.
        extra = []
        for i in range(38):
            a = math.pi * (0.15 + 0.7 * i / 37)
            extra.append((208 + math.cos(a) * 92, 370 + math.sin(a) * 18))
        target = np.vstack([target, np.asarray(extra, dtype=np.float32)])
    elif kind == "code":
        target = bitmap_points([
            "1000001", "0100010", "0010100", "0001000",
            "0010100", "0100010", "1000001",
        ], 168, 235, 22.0)
    else:
        # Shield + network nodes for cybersecurity/AI.
        theta = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        outer = np.column_stack((239 + 125 * np.cos(theta), 300 + 145 * np.sin(theta)))
        inner = np.column_stack((239 + 95 * np.cos(theta), 300 + 112 * np.sin(theta)))
        ring = outer[np.arange(0, 180, 2)]
        center = np.array([[239.0, 300.0]], dtype=np.float32)
        nodes = np.array([[145, 225], [333, 225], [145, 375], [333, 375], [239, 300]], dtype=np.float32)
        target = np.vstack([ring, inner[np.arange(0, 180, 3)], center, nodes])
        # Connectors sampled as small point clouds.
        segs = []
        for p in nodes[:4]:
            for t in np.linspace(0, 1, 35):
                segs.append(p * (1 - t) + nodes[-1] * t)
        target = np.vstack([target, np.asarray(segs, dtype=np.float32)])

    if len(target) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    chosen = rng.choice(len(target), count, replace=len(target) < count)
    return target[chosen]


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(v: float) -> str:
    return f"{float(v):.1f}".rstrip("0").rstrip(".")


def animate_values(frames: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index,0])} {num(p[index,1])}" for p in frames)


def point_path(points: np.ndarray) -> str:
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    if not unique:
        return ""
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1-x0+1}")
    return "".join(chunks)


def render(theme_name: str, portrait: np.ndarray, targets: dict[str, np.ndarray], rng: np.random.Generator) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]
    java = transport(source, targets["java"][:n])
    code = transport(java, targets["code"][:n])
    cyber = transport(code, targets["cyber"][:n])

    times = [0, 2.8, 4.1, 6.1, 7.4, 9.4, 10.7, 12.7, 15.0]
    key_times = ";".join(num(v / LOOP_SECONDS) for v in times)
    frames = [source, source, java, java, code, code, cyber, cyber, source]
    opacity_values = "0;0;1;1;1;1;1;1;0"

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" '
        'viewBox="0 0 1180 610" role="img" aria-labelledby="title desc">',
        '<title id="title">Andres Jalpilla — animated developer profile</title>',
        '<desc id="desc">Animated dithered portrait morphing through Java, code and cybersecurity silhouettes.</desc>',
        '<defs>',
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/></filter>',
        f'<filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" flood-opacity=".35"/><feComposite in2="b" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        '</defs>',
        f'<rect width="1180" height="610" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/><circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" letter-spacing=".4">andres@jalpilla:~$ ./profile.sh --live</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / 1-BIT / LIVE</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        '<g opacity="1">',
    ]

    # Ambient portrait drift layer.
    center = java.mean(axis=0)
    band_ids = rng.integers(0, 120, size=len(portrait))
    noise = rng.normal(0, 3.2, size=(120, 2))
    for band in range(120):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        delta = (center - pts.mean(axis=0)) * 0.15 + noise[band]
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" stroke-width="1" opacity=".72">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" values="0 0;0 0;{num(delta[0])} {num(delta[1])};{num(delta[0])} {num(delta[1])};0 0;0 0;0 0;0 0;0 0"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" keyTimes="{key_times}" values=".72;.72;0;0;0;0;0;0;.72"/></path>'
        )

    for i in range(n):
        parts.append(
            f'<path d="M-.7-.7h1.4v1.4h-1.4z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" values="{animate_values(frames, i)}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" values="{opacity_values}"/>'
            '</path>'
        )
    parts.append('</g>')

    # Animated load-in, similar to a terminal boot sequence.
    intro_ids = rng.integers(0, 56, size=len(portrait))
    order = rng.permutation(56)
    starts = np.empty(56)
    starts[order] = np.linspace(0.05, 1.25, 56)
    for group in range(56):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="2.66s" dur=".14s" values="1;0" fill="freeze"/></path>'
        )

    parts += [
        '</g>',
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">PTS {len(portrait):05d} · DITHER / MORPH</text>',
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
        f'<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A"><animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/></circle></g>',
        f'<text x="927" y="111" fill="#FF4D5A" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="700">LIVE</text>',
        f'<rect x="970" y="94" width="158" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" stroke="{t["chrome"]}"/>',
        f'<text x="1049" y="111" text-anchor="middle" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14" font-weight="700">@Jalpilla-Andres</text>',
    ]

    rows = [
        ("Subject", "Andres Jalpilla"),
        ("Role", "Telematics Engineer · Jr. Developer"),
        ("Origin", "Mexico / MX"),
        ("Core.Lang", "Java · Kotlin · Python · TypeScript"),
        ("Focus", "Software · Networking · Systems"),
        ("Next", "Cybersecurity · AI"),
        ("Build", "TlalocBox · Distributed Systems"),
        ("Tools", "Git · Linux · APIs · Virtualization"),
        ("GitHub", "Jalpilla-Andres"),
        ("Status", "Learning + Building + Shipping"),
        ("Mode", "ENGINEERING / ON"),
        ("Stack", "Java-first · systems-minded"),
        ("Mindset", '"keep learning, keep building"'),
        ("Node", "LATAM · UTC-6"),
    ]
    value_right = 1127.0
    row_y = 153.0

    def mono_width(text: str, size: float) -> float:
        return len(text) * size * 0.605

    for label, value in rows:
        label_len = mono_width(label, 14)
        value_len = mono_width(value, 14)
        start = 491 + label_len + 12
        end = value_right - value_len - 12
        leader = ''.join(f'M{x} {num(row_y - 4)}h1' for x in np.arange(start, max(start, end), 5.0))
        parts += [
            f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14">{esc(label)}</text>',
            f'<path d="{leader}" fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14" textLength="{num(value_len)}" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>',
        ]
        row_y += 22.4

    parts += [
        f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
        f'<text x="491" y="548" fill="{t["green"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">● ALL SYSTEMS NOMINAL</text>',
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">SVG-SMIL · AUTO LOOP · OPEN SOURCE</text>',
        '</svg>',
    ]
    return ''.join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source portrait: {SOURCE}")
    DATA.mkdir(parents=True, exist_ok=True)
    portraits: dict[str, np.ndarray] = {}
    for idx, theme in enumerate(THEMES):
        portraits[theme] = portrait_points(theme, np.random.default_rng(SEED + idx))
        np.save(DATA / f"portrait-{theme}.npy", portraits[theme])

    for idx, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + idx)
        n = min(TRAVELLER_COUNT, len(portraits[theme]))
        targets = {k: logo_points(k, n, rng) for k in ("java", "code", "cyber")}
        for name, points in targets.items():
            np.save(DATA / f"{name}-{theme}.npy", points)
        svg = render(theme, portraits[theme], targets, rng)
        out = ASSETS / f"banner-{theme}.svg"
        out.write_text(svg, encoding="utf-8")
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size/1024:.1f} KiB · {len(portraits[theme])} dots · {n} travellers")


if __name__ == '__main__':
    main()
