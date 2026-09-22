#!/usr/bin/env python3
"""Generate Andres Jalpilla's animated GitHub profile banner.

The banner uses a dithered portrait as a particle field and morphs it through
engineering scenes with a visible burst/vortex transition between scenes.
Everything is SVG/SMIL so no JavaScript is required in the README.

Easy customization:
    SCENE_SEQUENCE = ("portrait", "ai", "network", "telemetry", "cyber")
    TRANSITION_STYLE = "cinematic"

Available scenes:
    portrait, ai, network, telemetry, cyber, terminal, code, java

Available transition styles:
    cinematic, energetic, smooth, minimal
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
LOOP_SECONDS = 19.0
TRAVELLER_COUNT = 2300
SEED = 20260921

# ---------------------------------------------------------------------------
# CUSTOMIZE THESE SETTINGS
# ---------------------------------------------------------------------------
SCENE_SEQUENCE = ("portrait", "ai", "network", "telemetry", "cyber")
TRANSITION_STYLE = "cinematic"

# Visual viewport of the portrait panel: x=49..439, y=124..538.
# The portrait is intentionally shifted a little upward so the FACE, rather
# than the broad shoulders, sits on the panel's optical center.
VISUAL_CENTER = np.array([244.0, 331.0], dtype=np.float32)
PORTRAIT_Y_BIAS = -34.0

THEMES = {
    "dark": {
        "bg": "#050912", "panel": "#0B1220", "panel2": "#0E1728",
        "line": "#263754", "muted": "#7890AA", "text": "#EAF4FF",
        "particle": "#5EDBFF", "particle2": "#A78BFA", "chrome": "#38BDF8",
        "accent": "#8B5CF6", "green": "#22C55E", "red": "#FB7185",
        "amber": "#FBBF24", "shadow": "#01030A"
    },
    "light": {
        "bg": "#F5F8FB", "panel": "#FFFFFF", "panel2": "#F0F5F9",
        "line": "#CBD8E5", "muted": "#637388", "text": "#142033",
        "particle": "#1E6FA8", "particle2": "#6D43C6", "chrome": "#0284C7",
        "accent": "#7C3AED", "green": "#16A34A", "red": "#E11D48",
        "amber": "#D97706", "shadow": "#AAB8C5"
    },
}

STYLE = {
    "cinematic": dict(ease="0.65 0 0.15 1", burst=2.85, spin=1.0, snap=0.55),
    "energetic": dict(ease="0.76 0 0.24 1", burst=3.15, spin=1.65, snap=0.35),
    "smooth": dict(ease="0.42 0 0.58 1", burst=1.55, spin=0.70, snap=0.80),
    "minimal": dict(ease="0 0 1 1", burst=1.0, spin=0.20, snap=1.00),
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
    # A slightly tighter square crop keeps the head prominent while retaining
    # the tie/lapel details that make the silhouette recognisable.
    crop = image.crop((310, 42, 1218, 1218)).resize((322, 384), Image.Resampling.LANCZOS)
    rgba = np.asarray(crop)
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    lum = np.asarray(ImageOps.grayscale(crop.convert("RGB")), dtype=np.float32)
    prepared = ImageOps.autocontrast(Image.fromarray(np.uint8(lum)), cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.95)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=190, threshold=2))
    bits = floyd_steinberg(np.asarray(prepared))

    active = bits if theme == "dark" else ~bits
    active &= alpha > 0.12
    ys, xs = np.where(active)
    if not len(xs):
        return np.zeros((0, 2), dtype=np.float32)

    points = np.column_stack((84 + xs, 148 + ys)).astype(np.float32)

    # First center the full silhouette, then apply a small optical lift. This
    # makes the FACE sit around the visual center rather than the shoulders.
    lo, hi = points.min(axis=0), points.max(axis=0)
    bbox_center = (lo + hi) / 2.0
    points += VISUAL_CENTER - bbox_center
    points[:, 1] += PORTRAIT_Y_BIAS

    # Gentle edge thinning keeps the silhouette airy instead of a solid blob.
    if len(points) > 16000:
        points = points[rng.choice(len(points), 16000, replace=False)]
    return points


def sample_segment(p0, p1, count):
    t = np.linspace(0, 1, max(2, count))[:, None]
    return np.asarray(p0, dtype=np.float32)[None, :] * (1 - t) + np.asarray(p1, dtype=np.float32)[None, :] * t


def sample_polyline(points, per_seg=24):
    return np.vstack([sample_segment(a, b, per_seg) for a, b in zip(points[:-1], points[1:])])


def circle_points(cx, cy, r, count=140):
    th = np.linspace(0, 2 * math.pi, count, endpoint=False)
    return np.column_stack((cx + r * np.cos(th), cy + r * np.sin(th))).astype(np.float32)


def ellipse_points(cx, cy, rx, ry, count=180):
    th = np.linspace(0, 2 * math.pi, count, endpoint=False)
    return np.column_stack((cx + rx * np.cos(th), cy + ry * np.sin(th))).astype(np.float32)


def bitmap_points(rows, x0, y0, scale=1.0):
    pts = []
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c not in ("0", " "):
                pts.append((x0 + x * scale, y0 + y * scale))
    return np.asarray(pts, dtype=np.float32)


def scene_points(kind: str) -> np.ndarray:
    cx, cy = VISUAL_CENTER

    if kind == "ai":
        # Hex-core + branching neural graph + twin orbital rings.
        outer = np.column_stack((cx + 118*np.cos(np.arange(6)*math.pi/3), cy + 118*np.sin(np.arange(6)*math.pi/3)))
        mid = np.column_stack((cx + 72*np.cos(np.arange(6)*math.pi/3), cy + 72*np.sin(np.arange(6)*math.pi/3)))
        center = np.array([[cx, cy]], dtype=np.float32)
        chunks = [sample_polyline([tuple(x) for x in np.vstack([outer, outer[0]])], 32),
                  sample_polyline([tuple(x) for x in np.vstack([mid, mid[0]])], 24),
                  circle_points(cx, cy, 34, 110), circle_points(cx, cy, 45, 100)]
        for a, b in zip(outer, mid):
            chunks.append(sample_segment(tuple(a), tuple(b), 22))
        for p in mid:
            chunks.append(sample_segment(tuple(p), (cx, cy), 28))
            chunks.append(circle_points(float(p[0]), float(p[1]), 6.2, 26))
        # tiny side nodes
        side = np.array([[132, 250], [356, 250], [132, 412], [356, 412]], dtype=np.float32)
        for p in side:
            chunks.append(sample_segment(tuple(p), (cx, cy), 22))
            chunks.append(circle_points(float(p[0]), float(p[1]), 5.3, 24))
        return np.vstack(chunks)

    if kind == "network":
        # Globe/network: sphere + longitude/latitude arcs + connected nodes.
        chunks = [circle_points(cx, cy, 128, 250), ellipse_points(cx, cy, 54, 128, 190), ellipse_points(cx, cy, 94, 128, 200)]
        for ry in (34, 66, 96):
            th = np.linspace(0, 2*math.pi, 200, endpoint=False)
            chunks.append(np.column_stack((cx + 128*np.cos(th), cy + ry*np.sin(th))))
        nodes = np.array([[150,270],[199,408],[330,263],[359,386],[244,206],[244,456]], dtype=np.float32)
        for p in nodes:
            chunks.append(circle_points(float(p[0]), float(p[1]), 5.5, 26))
            chunks.append(sample_segment((cx,cy), tuple(p), 30))
        return np.vstack(chunks)

    if kind == "telemetry":
        # Radar/telemetry: target rings, sweep ray and signal pulse traces.
        chunks = [circle_points(cx, cy, 124, 240), circle_points(cx, cy, 82, 170), circle_points(cx, cy, 42, 110)]
        for r in (124, 82, 42):
            for ang in (0, math.pi/2, math.pi, 3*math.pi/2):
                chunks.append(sample_segment((cx,cy),(cx+r*math.cos(ang),cy+r*math.sin(ang)),18))
        sweep_end = (cx + 112, cy - 70)
        chunks.append(sample_segment((cx,cy), sweep_end, 120))
        # signal waveform on the lower side
        pts=[(122,432),(152,432),(168,386),(186,452),(212,410),(233,432),(264,432),(284,398),(304,432),(340,432),(360,412)]
        chunks.append(sample_polyline(pts, 18))
        for p in [(164,389),(286,399),(360,412)]:
            chunks.append(circle_points(float(p[0]), float(p[1]), 5.5, 24))
        return np.vstack(chunks)

    if kind == "cyber":
        # Shield with a circuit lock core.
        th = np.linspace(-math.pi, math.pi, 260)
        x = cx + 122*np.cos(th)
        y = cy + 142*np.sin(th)
        y = np.where(y < cy-65, cy-72 + (y-(cy-142))*0.62, y)
        outer = np.column_stack((x,y)).astype(np.float32)
        inner = np.column_stack((cx + 96*np.cos(th), cy + 112*np.sin(th))).astype(np.float32)
        chunks=[outer[::2], inner[::3]]
        lock_top = (cx-45, cy+4)
        lock_right = (cx+45, cy+4)
        chunks.append(sample_polyline([(cx-45,cy+5),(cx-45,cy+58),(cx+45,cy+58),(cx+45,cy+5)], 20))
        arch = ellipse_points(cx, cy+2, 34, 32, 100)
        chunks.append(arch[(arch[:,1] < cy+6)])
        for p in [(151,260),(337,260),(151,406),(337,406)]:
            chunks.append(sample_segment(p,(cx,cy),28))
            chunks.append(circle_points(*p,5.5,24))
        chunks.append(circle_points(cx,cy+31,7,30))
        return np.vstack(chunks)

    if kind == "terminal":
        prompt = sample_polyline([(132,272),(182,331),(132,390)],34)
        line1 = sample_segment((208,282),(356,282),92)
        line2 = sample_segment((208,330),(325,330),76)
        line3 = sample_segment((208,378),(294,378),58)
        cursor = sample_segment((313,416),(355,416),30)
        return np.vstack([prompt,line1,line2,line3,cursor])

    if kind == "code":
        left = sample_polyline([(162,244),(122,331),(162,418)],34)
        slash = sample_segment((220,244),(268,418),120)
        right = sample_polyline([(326,244),(366,331),(326,418)],34)
        cursor = sample_segment((190,456),(324,456),80)
        return np.vstack([left,slash,right,cursor])

    if kind == "java":
        glyphs={
            "J":["11111","00100","00100","00100","10100","10100","01100"],
            "A":["01110","10001","10001","11111","10001","10001","10001"],
            "V":["10001","10001","10001","10001","01010","01010","00100"],
        }
        rows=[""]*7
        for ch in "JAVA":
            rows=[r+glyphs[ch][i]+"0" for i,r in enumerate(rows)]
        return bitmap_points(rows,70,218,6.7)

    raise ValueError(f"Unknown scene: {kind}")


def resample_points(points, count, rng):
    if len(points) == 0:
        return np.zeros((count,2),dtype=np.float32)
    idx=rng.choice(len(points),count,replace=len(points)<count)
    return points[idx]


def transport(source, target):
    rows, cols=linear_sum_assignment(cdist(source,target,metric="sqeuclidean"))
    ordered=np.empty_like(target)
    ordered[rows]=target[cols]
    return ordered


def num(v):
    return f"{float(v):.1f}".rstrip("0").rstrip(".")


def particle_values(frames, i):
    return ";".join(f"{num(p[i,0])} {num(p[i,1])}" for p in frames)


def path_from_points(points):
    integer=np.rint(points).astype(int)
    unique=sorted({(int(x),int(y)) for x,y in integer}, key=lambda p:(p[1],p[0]))
    if not unique: return ""
    chunks=[]; i=0
    while i<len(unique):
        x0,y=unique[i]; x1=x0; i+=1
        while i<len(unique) and unique[i][1]==y and unique[i][0] <= x1+1:
            x1=unique[i][0]; i+=1
        chunks.append(f"M{x0} {y}h{x1-x0+1}")
    return "".join(chunks)


def add_particle(parts, idx, n, frames, times, style, color, rng):
    # Different particle sizes make the scene feel less like a uniformly tiled bitmap.
    r = (0.55, 0.9, 1.15)[idx % 3]
    op = (0.34, 0.58, 0.82)[idx % 3]
    vals = particle_values(frames, idx)
    key_times = ";".join(num(x/LOOP_SECONDS) for x in times)
    key_splines = ";".join(style["ease"] for _ in range(len(times)-1))
    # Visibility pulses at burst/snap/hold keyframes. Keep the number of
    # opacity values exactly aligned with the generated keyframe count.
    pulse=[]
    for k in range(len(times)):
        if k == 0:
            pulse.append(str(op))
        elif k >= 2 and k % 3 == 2:
            pulse.append("0.10")       # burst: particles spread / thin out
        elif k >= 3 and k % 3 == 0:
            pulse.append("0.86")       # snap: particles arrive strongly
        else:
            pulse.append(str(op))
    opacity_values=";".join(pulse)
    parts.append(
        f'<circle r="{r}" fill="{color}" opacity="{op}">'
        f'<animateTransform attributeName="transform" type="translate" begin="0s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" keySplines="{key_splines}" values="{vals}"/>'
        f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" keySplines="{key_splines}" values="{opacity_values}"/>'
        '</circle>'
    )


def render(theme_name, portrait, targets, rng):
    t=THEMES[theme_name]; style=STYLE[TRANSITION_STYLE]
    n=min(TRAVELLER_COUNT,len(portrait))
    source=resample_points(portrait,n,rng)

    scenes={"portrait":source}
    current=source
    for scene in SCENE_SEQUENCE[1:]:
        target=resample_points(targets[scene],n,rng)
        current=transport(current,target)
        scenes[scene]=current

    # Explicit phases make the transitions visually obvious:
    # hold -> burst -> settle -> hold -> burst -> settle...
    transition_count=len(SCENE_SEQUENCE)
    hold=1.20
    settle=0.95
    burst=1.05
    phase=hold+burst+settle
    if phase*transition_count > LOOP_SECONDS-0.7:
        phase=(LOOP_SECONDS-0.7)/transition_count
        hold=phase*0.40; burst=phase*0.38; settle=phase*0.22

    times=[0.0, hold]
    frame_points=[scenes["portrait"], scenes["portrait"]]
    labels=[SCENE_SEQUENCE[0], SCENE_SEQUENCE[0]]
    current_time=hold

    for idx in range(1,len(SCENE_SEQUENCE)):
        prev_name=SCENE_SEQUENCE[idx-1]; next_name=SCENE_SEQUENCE[idx]
        a=scenes[prev_name]; b=scenes[next_name]
        center=np.asarray(VISUAL_CENTER,dtype=np.float32)
        vec=a-center
        norm=np.linalg.norm(vec,axis=1,keepdims=True)+1e-5
        radial=vec/norm
        tang=np.column_stack([-radial[:,1], radial[:,0]])
        noise=rng.normal(0,1.0,size=a.shape).astype(np.float32)
        burst_pos=center + radial*(style["burst"]*(35 + rng.random((n,1))*80)) + tang*(rng.normal(0,28,size=(n,1))) + noise*7
        # A flash-like overshoot near the target gives the morph a snap.
        snap_pos=b + (b-center)*(0.08 + 0.04*rng.random((n,1))) + rng.normal(0,2.0,size=b.shape)

        current_time += burst
        times.append(current_time); frame_points.append(burst_pos); labels.append(next_name)
        current_time += settle
        times.append(current_time); frame_points.append(snap_pos); labels.append(next_name)
        current_time += 0.88
        times.append(current_time); frame_points.append(b); labels.append(next_name)

    # close with a final burst from the last scene back into the portrait
    a=scenes[SCENE_SEQUENCE[-1]]; b=scenes["portrait"]
    center=np.asarray(VISUAL_CENTER,dtype=np.float32)
    vec=a-center; norm=np.linalg.norm(vec,axis=1,keepdims=True)+1e-5; radial=vec/norm
    tang=np.column_stack([-radial[:,1],radial[:,0]])
    burst_pos=center + radial*(style["burst"]*(42 + rng.random((n,1))*86)) + tang*(rng.normal(0,30,size=(n,1)))
    current_time += burst; times.append(current_time); frame_points.append(burst_pos); labels.append("portrait")
    times.append(LOOP_SECONDS); frame_points.append(b); labels.append("portrait")

    # normalize in case rounding pushed the end slightly beyond LOOP_SECONDS
    scale=LOOP_SECONDS/max(times[-1],LOOP_SECONDS)
    times=[x*scale for x in times]

    key_times=";".join(num(x/LOOP_SECONDS) for x in times)
    key_splines=";".join(style["ease"] for _ in range(len(times)-1))

    parts=[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Andres Jalpilla — animated engineering profile</title>',
        '<desc id="desc">A centered dithered portrait transforms into AI, network, telemetry and cybersecurity particle scenes.</desc>',
        '<defs>',
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".25"/></filter>',
        f'<filter id="glow" x="-120%" y="-120%" width="340%" height="340%"><feGaussianBlur stdDeviation="2.8" result="b"/><feFlood flood-color="{t["chrome"]}" flood-opacity=".48"/><feComposite in2="b" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="clip"><rect x="49" y="125" width="390" height="414" rx="3"/></clipPath>',
        f'<radialGradient id="vignette"><stop offset="0" stop-color="{t["chrome"]}" stop-opacity=".07"/><stop offset="1" stop-color="{t["chrome"]}" stop-opacity="0"/></radialGradient>',
        '</defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/><circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" letter-spacing=".5">andres@jalpilla:~$ ./profile.sh --visualize</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">PARTICLE.FIELD</text>',
        f'<text x="439" y="111" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">OPTICAL CENTER / {n:,} NODES</text>',
        f'<rect x="49" y="125" width="390" height="414" fill="url(#vignette)" opacity=".8"/>',
        # animated frame corners
        f'<path d="M49 141h14M49 141v14M439 141h-14M439 141v14M49 539h14M49 539v-14M439 539h-14M439 539v-14" fill="none" stroke="{t["chrome"]}" opacity=".45"/>',
        '<g clip-path="url(#clip)">',
        # permanent micro-grid
        f'<g opacity=".18" stroke="{t["line"]}"><path d="M49 207H439M49 289H439M49 372H439M49 455H439"/><path d="M118 125V539M196 125V539M274 125V539M352 125V539"/></g>',
        # orbit + scan sweep + target reticle
        f'<circle cx="244" cy="331" r="169" fill="none" stroke="{t["chrome"]}" stroke-width="1" stroke-dasharray="2 12" opacity=".11"><animateTransform attributeName="transform" type="rotate" from="0 244 331" to="360 244 331" dur="26s" repeatCount="indefinite"/></circle>',
        f'<circle cx="244" cy="331" r="126" fill="none" stroke="{t["chrome"]}" stroke-width="1" opacity=".10"><animate attributeName="opacity" values=".05;.18;.05" dur="4.6s" repeatCount="indefinite"/></circle>',
        f'<path d="M244 145V170M244 492V517M58 331H83M405 331H430" stroke="{t["chrome"]}" opacity=".20"/>',
    ]

    # Scene labels with a real animated state indicator.
    label_map={
        "portrait":"PORTRAIT / IDENTITY",
        "ai":"AI CORE / NEURAL",
        "network":"NETWORK / TELEMATICS",
        "telemetry":"TELEMETRY / SIGNAL",
        "cyber":"CYBERSECURITY / SHIELD",
        "terminal":"TERMINAL / EXEC",
        "code":"CODE / BUILD",
        "java":"JAVA / JVM",
    }
    for si, scene in enumerate(SCENE_SEQUENCE):
        # Each scene is mostly opaque during its hold/settle phases.
        indices=[i for i,lbl in enumerate(labels) if lbl==scene]
        opacity=["0"]*len(times)
        for i in indices: opacity[i]="1"
        values=";".join(opacity)
        parts.append(f'<text x="61" y="522" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10" opacity="0">{esc(label_map.get(scene,scene.upper()))}'
                     f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{key_times}" values="{values}"/></text>')

    # Particle field: a second, larger ambient layer gives the burst depth.
    ambient=resample_points(source,min(620,n),rng)
    for j,p in enumerate(ambient):
        parts.append(
            f'<circle cx="{num(p[0])}" cy="{num(p[1])}" r="{0.5 if j%2 else 0.8}" fill="{t["particle2"]}" opacity="0.10">'
            f'<animate attributeName="opacity" values=".02;.13;.02" dur="{2.4+(j%7)*.12:.2f}s" begin="{(j%9)*.11:.2f}s" repeatCount="indefinite"/></circle>'
        )

    # Some travellers use the secondary accent for depth.
    accent_indices=set(rng.choice(n,max(1,n//7),replace=False).tolist())
    for i in range(n):
        add_particle(parts,i,n,frame_points,times,style,t["particle2"] if i in accent_indices else t["particle"],rng)

    # Burst ring flashes synchronized with the major transitions.
    for k,tm in enumerate(times[2::3]):
        if tm >= LOOP_SECONDS: continue
        parts.append(
            f'<circle cx="244" cy="331" r="28" fill="none" stroke="{t["chrome"]}" stroke-width="1.2" opacity="0" filter="url(#glow)">'
            f'<animate attributeName="r" begin="{num(tm)}s" dur=".85s" values="24;174" fill="freeze"/>'
            f'<animate attributeName="opacity" begin="{num(tm)}s" dur=".85s" values=".42;0" fill="freeze"/>'
            '</circle>'
        )

    parts += [
        '</g>',
        f'<text x="61" y="548" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">CENTER LOCKED · BURST MORPH · {esc(TRANSITION_STYLE.upper())}</text>',
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
        f'<g filter="url(#glow)"><circle cx="904" cy="106" r="4" fill="{t["red"]}"><animate attributeName="opacity" values="1;.25;1" dur="1.6s" repeatCount="indefinite"/></circle></g>',
        f'<text x="916" y="111" fill="{t["red"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="700">LIVE MORPH</text>',
        f'<rect x="1003" y="94" width="125" height="24" rx="12" fill="{t["chrome"]}" opacity=".15" stroke="{t["chrome"]}"/>',
        f'<text x="1065" y="111" text-anchor="middle" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="700">@Jalpilla-Andres</text>',
    ]

    rows=[
        ("Subject","Andres Jalpilla"),("Role","Telematics Engineer · Jr. Developer"),
        ("Origin","Mexico / MX"),("Core.Lang","Java · Kotlin · Python · TypeScript"),
        ("Focus","Software · Networking · Systems"),("Next","Cybersecurity · AI"),
        ("Build","TlalocBox · Distributed Systems"),("Tools","Git · Linux · APIs · Virtualization"),
        ("GitHub","Jalpilla-Andres"),("Status","Learning + Building + Shipping"),
        ("Mode","ENGINEERING / ON"),("Stack","Java-first · systems-minded"),
        ("Mindset",'"keep learning, keep building"'),("Node","LATAM · UTC-6"),
    ]
    value_right=1127.0; row_y=153.0
    def mono_width(text,size): return len(text)*size*0.605
    for label,value in rows:
        start=491+mono_width(label,14)+12; end=value_right-mono_width(value,14)-12
        leader=''.join(f'M{x} {num(row_y-4)}h1' for x in np.arange(start,max(start,end),5.0))
        parts += [
            f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14">{esc(label)}</text>',
            f'<path d="{leader}" fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14" textLength="{num(mono_width(value,14))}" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>'
        ]
        row_y+=22.4
    parts += [
        f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
        f'<text x="491" y="548" fill="{t["green"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">● ALL SYSTEMS NOMINAL</text>',
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">SVG-SMIL · BURST · AUTO LOOP</text>',
        '</svg>'
    ]
    return ''.join(parts)


def main():
    if not SOURCE.exists(): raise SystemExit(f"Missing source portrait: {SOURCE}")
    DATA.mkdir(parents=True,exist_ok=True)
    portraits={}
    for idx,theme in enumerate(THEMES):
        portraits[theme]=portrait_points(theme,np.random.default_rng(SEED+idx))
        np.save(DATA/f"portrait-{theme}.npy",portraits[theme])
    scenes=sorted({x for x in SCENE_SEQUENCE if x!="portrait"})
    for idx,theme in enumerate(THEMES):
        rng=np.random.default_rng(SEED+100+idx)
        targets={}
        for scene in scenes:
            targets[scene]=scene_points(scene)
            np.save(DATA/f"{scene}-{theme}.npy",targets[scene])
        svg=render(theme,portraits[theme],targets,rng)
        out=ASSETS/f"banner-{theme}.svg"; out.write_text(svg,encoding="utf-8")
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size/1024:.1f} KiB · {len(portraits[theme])} portrait pts · {min(TRAVELLER_COUNT,len(portraits[theme]))} travellers · sequence={'→'.join(SCENE_SEQUENCE)} · style={TRANSITION_STYLE}")

if __name__=='__main__': main()
