#!/usr/bin/env python3
"""Generate Andres Jalpilla's animated GitHub profile banner.

The portrait is converted into a dithered particle field and morphed through
professional developer/engineering motifs. The sequence and transition style
are intentionally easy to customize near the top of this file.

Quick customization:
    SCENE_SEQUENCE   = ('portrait', 'code', 'neural', 'cyber')
    TRANSITION_STYLE = 'cinematic'

Available scenes:
    portrait, code, neural, cyber, globe, terminal, java

Available transition styles:
    cinematic, smooth, snappy, linear
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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

# ---------------------------------------------------------------------------
# CUSTOMIZE THESE TWO SETTINGS
# ---------------------------------------------------------------------------
SCENE_SEQUENCE = ("portrait", "code", "neural", "cyber")
TRANSITION_STYLE = "cinematic"

# The visual frame is x=49..439, y=124..538 -> center = (244, 331).
VISUAL_CENTER = np.array([244.0, 331.0], dtype=np.float32)

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

TRANSITION_CURVES = {
    # Luxurious ease-in/out: starts gently, accelerates, then settles.
    "cinematic": "0.55 0 0.15 1",
    # Standard UI motion.
    "smooth": "0.42 0 0.58 1",
    # Quicker snap into the destination.
    "snappy": "0.76 0 0.24 1",
    # Original-style constant motion.
    "linear": "0 0 1 1",
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
    # Tighter crop while retaining hair, face, tie and shoulders.
    crop = image.crop((320, 80, 1210, 1170)).resize((300, 340), Image.Resampling.LANCZOS)
    rgba = np.asarray(crop)
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    lum = np.asarray(ImageOps.grayscale(crop.convert("RGB")), dtype=np.float32)
    prepared = ImageOps.autocontrast(Image.fromarray(np.uint8(lum)), cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.7)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=175, threshold=2))
    bits = floyd_steinberg(np.asarray(prepared))

    active = bits if theme == "dark" else ~bits
    active &= alpha > 0.15

    ys, xs = np.where(active)
    points = np.column_stack((75 + xs, 155 + ys)).astype(np.float32)

    # Center the visible particle silhouette inside the actual visual viewport.
    # The previous version was left/up-biased by ~20px/~7px; using the bbox center
    # keeps the face and shoulders visually centered even if the crop changes later.
    if len(points):
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        bbox_center = (lo + hi) / 2.0
        points += VISUAL_CENTER - bbox_center

    if len(points) > 10000:
        points = points[rng.choice(len(points), 10000, replace=False)]
    return points


def sample_segment(p0: tuple[float, float], p1: tuple[float, float], count: int) -> np.ndarray:
    t = np.linspace(0, 1, max(2, count))[:, None]
    a = np.array(p0, dtype=np.float32)[None, :]
    b = np.array(p1, dtype=np.float32)[None, :]
    return a * (1 - t) + b * t


def sample_polyline(points: list[tuple[float, float]], per_seg: int = 28) -> np.ndarray:
    chunks = [sample_segment(a, b, per_seg) for a, b in zip(points[:-1], points[1:])]
    return np.vstack(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)


def circle_points(cx: float, cy: float, r: float, count: int = 140) -> np.ndarray:
    theta = np.linspace(0, 2 * math.pi, count, endpoint=False)
    return np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta))).astype(np.float32)


def bitmap_points(rows: list[str], x0: float, y0: float, scale: float = 1.0) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell not in ("0", " "):
                pts.append((x0 + x * scale, y0 + y * scale))
    return np.asarray(pts, dtype=np.float32)


def scene_points(kind: str) -> np.ndarray:
    """Generate an intentionally clean, recognisable particle silhouette."""
    cx, cy = VISUAL_CENTER

    if kind == "code":
        left = sample_polyline([(164, 245), (122, 331), (164, 417)], 34)
        slash = sample_segment((220, 245), (268, 417), 120)
        right = sample_polyline([(324, 245), (366, 331), (324, 417)], 34)
        cursor = sample_segment((188, 448), (320, 448), 76)
        block = sample_polyline([(184, 448), (205, 448), (205, 462), (184, 462)], 15)
        return np.vstack([left, slash, right, cursor, block])

    if kind == "neural":
        layers = [
            np.column_stack((np.full(5, 145.0), np.linspace(246, 416, 5))),
            np.column_stack((np.full(7, 244.0), np.linspace(226, 436, 7))),
            np.column_stack((np.full(5, 343.0), np.linspace(246, 416, 5))),
        ]
        chunks: list[np.ndarray] = []
        for left, right in zip(layers[:-1], layers[1:]):
            for p0 in left:
                for p1 in right:
                    chunks.append(sample_segment(tuple(p0), tuple(p1), 11))
        nodes = np.vstack(layers)
        for p in nodes:
            chunks.append(circle_points(float(p[0]), float(p[1]), 5.5, 24))
        # Central orbit ring makes the neural scene feel more like an AI system.
        chunks.append(circle_points(float(cx), float(cy), 92, 130))
        chunks.append(circle_points(float(cx), float(cy), 104, 70))
        return np.vstack(chunks)

    if kind == "cyber":
        # Shield outline + circuit traces + central lock/network node.
        theta = np.linspace(0, 2 * math.pi, 240, endpoint=False)
        x = cx + 122 * np.cos(theta)
        y = cy + 142 * np.sin(theta)
        # Cut the top into a shield-like point by pulling its upper arc toward center.
        y = np.where(y < cy - 70, cy - 70 + (y - (cy - 142)) * 0.65, y)
        outer = np.column_stack((x, y)).astype(np.float32)
        inner = np.column_stack((cx + 92 * np.cos(theta), cy + 108 * np.sin(theta)))
        nodes = np.array([[150, 246], [338, 246], [150, 416], [338, 416], [244, 331]], dtype=np.float32)
        chunks = [outer[::2], inner[::3]]
        for p in nodes[:4]:
            chunks.append(sample_segment(tuple(p), (cx, cy), 32))
            chunks.append(circle_points(float(p[0]), float(p[1]), 5.5, 24))
        chunks.append(circle_points(float(cx), float(cy), 16, 64))
        chunks.append(circle_points(float(cx), float(cy), 6, 36))
        return np.vstack(chunks)

    if kind == "globe":
        chunks = [circle_points(float(cx), float(cy), 126, 220), circle_points(float(cx), float(cy), 92, 160)]
        # Longitude / latitude arcs.
        for rx in (36, 72, 104):
            theta = np.linspace(0, 2 * math.pi, 160, endpoint=False)
            chunks.append(np.column_stack((cx + rx * np.cos(theta), cy + 126 * np.sin(theta))))
        for ry in (34, 64, 92):
            theta = np.linspace(0, 2 * math.pi, 160, endpoint=False)
            chunks.append(np.column_stack((cx + 126 * np.cos(theta), cy + ry * np.sin(theta))))
        return np.vstack(chunks)

    if kind == "terminal":
        prompt = sample_polyline([(124, 290), (178, 331), (124, 372)], 38)
        underscore = sample_segment((205, 381), (350, 381), 95)
        cursor = np.vstack([
            sample_segment((332, 348), (365, 348), 24),
            sample_segment((365, 348), (365, 381), 24),
        ])
        return np.vstack([prompt, underscore, cursor])

    if kind == "java":
        glyphs = {
            "J": ["11111", "00100", "00100", "00100", "10100", "10100", "01100"],
            "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
            "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
        }
        rows = [""] * 7
        for ch in "JAVA":
            g = glyphs[ch]
            rows = [r + g[i] + "0" for i, r in enumerate(rows)]
        target = bitmap_points(rows, 92, 218, 7.0)
        cup = sample_polyline([(130, 383), (340, 383)], 80)
        bowl = sample_polyline([(155, 383), (168, 405), (302, 405), (318, 383)], 25)
        return np.vstack([target, cup, bowl])

    raise ValueError(f"Unknown scene: {kind}")


def resample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    chosen = rng.choice(len(points), count, replace=len(points) < count)
    return points[chosen]


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
    available = {"portrait", *targets.keys()}
    unknown = [x for x in SCENE_SEQUENCE if x not in available]
    if unknown:
        raise ValueError(f"Unknown scenes in SCENE_SEQUENCE: {unknown}")
    if SCENE_SEQUENCE[0] != "portrait":
        raise ValueError("SCENE_SEQUENCE must start with 'portrait'.")

    n = min(TRAVELLER_COUNT, len(portrait))
    source = resample_points(portrait, n, rng)

    scene_arrays: dict[str, np.ndarray] = {"portrait": source}
    current = source
    for scene in SCENE_SEQUENCE[1:]:
        current = transport(current, resample_points(targets[scene], n, rng))
        scene_arrays[scene] = current

    # Evenly spaced transition/hold rhythm. Each scene gets a short hold, then
    # a cinematic morph to the next scene. The last frame closes the loop.
    transitions = len(SCENE_SEQUENCE)
    hold = 1.15
    duration = 2.05
    tail = LOOP_SECONDS - hold - transitions * duration
    if tail < 1.0:
        duration = (LOOP_SECONDS - hold - 1.0) / transitions
    times: list[float] = [0.0, hold]
    frames: list[np.ndarray] = [scene_arrays[SCENE_SEQUENCE[0]], scene_arrays[SCENE_SEQUENCE[0]]]
    current_time = hold
    for idx, scene in enumerate(SCENE_SEQUENCE[1:], start=1):
        current_time += duration
        times.append(current_time)
        frames.append(scene_arrays[scene])
        current_time += 0.72
        times.append(current_time)
        frames.append(scene_arrays[scene])
    times[-1] = LOOP_SECONDS
    frames[-1] = scene_arrays[SCENE_SEQUENCE[0]]

    # Build one cubic-bezier spline per keyframe interval.
    curve = TRANSITION_CURVES.get(TRANSITION_STYLE, TRANSITION_CURVES["cinematic"])
    key_times = ";".join(num(v / LOOP_SECONDS) for v in times)
    key_splines = ";".join(curve for _ in range(len(times) - 1))
    opacity_values = ";".join("0" if i == 0 else "1" for i in range(len(frames)))

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" '
        'viewBox="0 0 1180 610" role="img" aria-labelledby="title desc">',
        '<title id="title">Andres Jalpilla — animated engineering profile</title>',
        '<desc id="desc">Centered dithered portrait morphing through code, neural network and cybersecurity motifs.</desc>',
        '<defs>',
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/></filter>',
        f'<filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" flood-opacity=".35"/><feComposite in2="b" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        '</defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/><circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" letter-spacing=".4">andres@jalpilla:~$ ./profile.sh --live</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / PARTICLES / LIVE</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        # Premium motion accents: subtle orbital ring + scan sweep.
        f'<circle cx="244" cy="331" r="171" fill="none" stroke="{t["chrome"]}" stroke-width="1" stroke-dasharray="2 11" opacity=".13">',
        f'<animateTransform attributeName="transform" type="rotate" from="0 244 331" to="360 244 331" dur="18s" repeatCount="indefinite"/></circle>',
        f'<rect x="49" y="124" width="3" height="414" fill="{t["chrome"]}" opacity="0">',
        f'<animate attributeName="x" values="49;438" dur="2.4s" begin="{INTRO_SECONDS}s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;.34;0" dur="2.4s" begin="{INTRO_SECONDS}s" repeatCount="indefinite"/></rect>',
        '<g opacity="1">',
    ]

    # A soft ambient dust layer that responds to the portrait-to-shape motion.
    ambient = resample_points(portrait, min(900, len(portrait)), rng)
    ambient_groups = rng.integers(0, 84, size=len(ambient))
    for band in range(84):
        pts = ambient[ambient_groups == band]
        if not len(pts):
            continue
        delta = rng.normal(0, 4.2, size=2)
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" stroke-width="1" opacity=".18">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" keySplines="{key_splines}" values="0 0;{num(delta[0])} {num(delta[1])};0 0;{num(-delta[0])} {num(-delta[1])};0 0;0 0;0 0;0 0;0 0;0 0;0 0"/>'
            '</path>'
        )

    for i in range(n):
        parts.append(
            f'<path d="M-.7-.7h1.4v1.4h-1.4z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" keySplines="{key_splines}" values="{animate_values(frames, i)}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" keySplines="{key_splines}" values="{opacity_values}"/>'
            '</path>'
        )
    parts.append('</g>')

    # Animated load-in: terminal boot sequence, kept intentionally short.
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

    # Scene label in the lower-left visual panel.
    sequence_text = " → ".join(SCENE_SEQUENCE[1:]).upper()
    parts += [
        '</g>',
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">PTS {len(portrait):05d} · {esc(sequence_text)} · {esc(TRANSITION_STYLE.upper())}</text>',
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
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">SVG-SMIL · {esc(TRANSITION_STYLE.upper())} · AUTO LOOP</text>',
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

    scenes = sorted({x for x in SCENE_SEQUENCE if x != "portrait"})
    for idx, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + idx)
        n = min(TRAVELLER_COUNT, len(portraits[theme]))
        targets = {}
        for scene in scenes:
            target = scene_points(scene)
            np.save(DATA / f"{scene}-{theme}.npy", target)
            targets[scene] = target
        svg = render(theme, portraits[theme], targets, rng)
        out = ASSETS / f"banner-{theme}.svg"
        out.write_text(svg, encoding="utf-8")
        print(
            f"{out.relative_to(ROOT)}: {out.stat().st_size/1024:.1f} KiB · "
            f"{len(portraits[theme])} portrait dots · {n} travellers · "
            f"sequence={'→'.join(SCENE_SEQUENCE)} · transition={TRANSITION_STYLE}"
        )


if __name__ == '__main__':
    main()
