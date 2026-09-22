"""Particle-portrait engine for the profile banner.

Turns a real photo into a static, pre-baked SVG/SMIL animation: the photo is
dithered into a point cloud once at build time, two more scenes (an AI motif
and a networking motif) are generated as point clouds too, and the browser
just interpolates between those frozen arrays forever. Because every state
in the loop is a *stored, pre-computed* array (never something recomputed
live), cycle 1, cycle 50 and cycle 5000 are bit-for-bit identical -- there is
no state that can drift, blur or accumulate error between repeats.

No JavaScript, no canvas, no external runtime: everything heavy (dithering,
sampling, optimal-transport pairing, path packing) happens once in Python
during the GitHub Actions build, and the output is a handful of numbers
baked into the SVG.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _serpentine_dither(gray: np.ndarray) -> np.ndarray:
    """1-bit Floyd-Steinberg dithering, boustrophedon (zig-zag) scan order.

    Returns a boolean grid where True = a "lit" (background/white) pixel and
    False = "ink" (a particle). Serpentine scanning (alternating left-right,
    right-left per row) avoids the directional streaking a naive left-right
    scan produces on portrait edges (hairline, glasses rim, lapel).
    """
    work = gray.astype(np.float32) / 255.0
    h, w = work.shape
    lit = np.zeros((h, w), dtype=bool)
    for y in range(h):
        going_right = y % 2 == 0
        xs = range(w) if going_right else range(w - 1, -1, -1)
        step = 1 if going_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            lit[y, x] = bool(new)
            err = old - new
            nx = x + step
            if 0 <= nx < w:
                work[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    work[y + 1, x - step] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    work[y + 1, nx] += err * 1 / 16
    return lit


def _subject_bbox(rgb: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    """Bounding box of everything that isn't near-white studio background."""
    dist_from_white = np.sqrt(((255.0 - rgb.astype(np.float32)) ** 2).sum(axis=2))
    ys, xs = np.where(dist_from_white > 22)
    if len(xs) == 0:
        h, w = rgb.shape[:2]
        return 0, 0, w, h
    h, w = rgb.shape[:2]
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, w)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, h)
    return x0, y0, x1, y1


def photo_to_points(
    path,
    grid_w: int,
    grid_h: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dither a photo into a capped point cloud + a per-point darkness value.

    Returns (points[N,2] in grid coordinates, weight[N] in 0..1 where 1 is
    the darkest/most defining ink, e.g. hair or glasses).
    """
    source = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = _subject_bbox(np.asarray(source), pad=max(source.size) // 18)
    crop = source.crop((x0, y0, x1, y1))

    # Fit the crop into the grid box without distorting proportions, then
    # center-pad so the framing stays a clean head-and-shoulders portrait.
    target_ratio = grid_w / grid_h
    crop_ratio = crop.width / crop.height
    if crop_ratio > target_ratio:
        new_h = crop.height
        new_w = int(new_h * target_ratio)
        left = (crop.width - new_w) // 2
        crop = crop.crop((left, 0, left + new_w, new_h))
    else:
        new_w = crop.width
        new_h = int(new_w / target_ratio)
        top = 0  # keep the top (face), trim from the bottom (shoulders/chest)
        crop = crop.crop((0, top, new_w, top + new_h))

    crop = crop.resize((grid_w, grid_h), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=1))

    lum = np.asarray(gray, dtype=np.float32)
    lit = _serpentine_dither(np.asarray(gray))
    ink = ~lit  # dark pixels become particles; white studio background stays empty

    ys, xs = np.where(ink)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    weight = 1.0 - (lum[ys, xs] / 255.0)  # darker pixel -> higher weight

    if len(xs) > max_points:
        xs, ys, weight = _stratified_thin(xs, ys, weight, grid_w, grid_h, max_points, rng)

    points = np.column_stack((xs, ys)).astype(np.float32)
    return points, weight.astype(np.float32)


def _stratified_thin(
    xs: np.ndarray,
    ys: np.ndarray,
    weight: np.ndarray,
    grid_w: int,
    grid_h: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap point count without losing small, low-mass features.

    A pure global weighted sample is dominated by whichever region has the
    most ink (hair, a suit) and quietly erases small high-value details like
    eyes or glasses rims. Instead, tile the frame into cells sized so there
    are roughly as many cells as the target point budget, keep exactly one
    (the highest-weight) point per occupied cell, then -- only if budget is
    left over -- add extra points in the densest cells for texture.
    """
    cell = max(1.0, np.sqrt((grid_w * grid_h) / (max_points * 1.15)))
    col = (xs / cell).astype(np.int32)
    row = (ys / cell).astype(np.int32)
    cell_id = row.astype(np.int64) * (grid_w // int(cell) + 2) + col

    order = np.lexsort((-weight, cell_id))  # highest weight first within each cell
    cell_sorted = cell_id[order]
    first_in_cell = np.ones(len(order), dtype=bool)
    first_in_cell[1:] = cell_sorted[1:] != cell_sorted[:-1]
    reps = order[first_in_cell]

    if len(reps) >= max_points:
        # Still too many representatives: keep the globally highest-weight ones.
        top = np.argsort(-weight[reps])[:max_points]
        reps = reps[top]
        return xs[reps], ys[reps], weight[reps]

    remaining_budget = max_points - len(reps)
    rest = order[~first_in_cell]
    if remaining_budget > 0 and len(rest):
        probs = weight[rest] + 0.1
        probs = probs / probs.sum()
        extra = rng.choice(rest, size=min(remaining_budget, len(rest)), replace=False, p=probs)
        reps = np.concatenate([reps, extra])

    return xs[reps], ys[reps], weight[reps]


def exploded_state(
    points: np.ndarray,
    frame_w: float,
    frame_h: float,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    """One fixed 'dispersed' position per particle: a radial burst from the
    portrait's own centroid, plus mild per-particle jitter so the burst
    reads as organic rather than a mechanical starburst.

    Every particle keeps a 1:1 identity with its portrait point (no
    reassignment needed), so the return trip is guaranteed to land each
    particle back on its exact original pixel -- the source of the
    no-degradation guarantee.
    """
    if len(points) == 0:
        return points.copy()
    centroid = points.mean(axis=0)
    direction = points - centroid
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = direction / norm

    # Headroom to each wall along this particle's own outward direction, so
    # the burst distance is scaled per-particle instead of being clipped --
    # clipping piles particles up along the frame edges, which reads as a
    # glitch rather than a dispersal. Each particle gets a comfortable
    # fraction of its own available room, never the wall itself.
    margin = 10.0
    room_x = np.where(unit[:, 0] >= 0, frame_w - margin - points[:, 0], points[:, 0] - margin)
    room_y = np.where(unit[:, 1] >= 0, frame_h - margin - points[:, 1], points[:, 1] - margin)
    room_x = np.clip(room_x, 4.0, None)
    room_y = np.clip(room_y, 4.0, None)
    safe_ux = np.abs(unit[:, 0]); safe_ux[safe_ux < 1e-3] = 1e-3
    safe_uy = np.abs(unit[:, 1]); safe_uy[safe_uy < 1e-3] = 1e-3
    max_travel = np.minimum(room_x / safe_ux, room_y / safe_uy)

    fraction = rng.uniform(0.30, 0.62, size=len(points)) * strength
    travel = (max_travel * fraction)[:, None]
    jitter = rng.normal(0, 3.0, size=points.shape)
    burst = points + unit * travel + jitter

    burst[:, 0] = np.clip(burst[:, 0], margin * 0.4, frame_w - margin * 0.4)
    burst[:, 1] = np.clip(burst[:, 1], margin * 0.4, frame_h - margin * 0.4)
    return burst.astype(np.float32)


def num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")
