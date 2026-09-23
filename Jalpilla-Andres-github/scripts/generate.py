#!/usr/bin/env python3
"""Generate profile SVG assets for Jalpilla-Andres.

The banner's particle portrait needs numpy + Pillow (see requirements.txt);
every other card only uses the standard library. When GH_TOKEN is available,
repo/user/language statistics are refreshed from GitHub's public REST API.
"""
from __future__ import annotations
import json, math, os, html, sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from particles import (
    photo_to_points,
    exploded_state,
    neural_layers_state,
    network_graph_state,
    num as pnum,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SOURCE_PHOTO = ASSETS / "source" / "andres.png"
CFG_PATH = ASSETS / "profile.json"
USER = "Jalpilla-Andres"
API = "https://api.github.com"

# Particle portrait tuning. Point count/seed are fixed so every regeneration
# (local or in CI) produces the exact same dot cloud -- reproducible builds,
# not a fresh random portrait on every commit.
#
# PORTRAIT_GRID's aspect ratio is kept equal to the on-banner frame's
# aspect ratio (see frame_w/frame_h in banner()) -- previously the two
# didn't match (220x258 vs a 400x420 frame), which silently stretched the
# face horizontally by ~12% every render and was a real part of why the
# portrait was hard to recognize. Grid resolution and point budget are also
# raised so more fine detail (eyes, brows, hairline) survives dithering.
PORTRAIT_GRID = (250, 295)
PORTRAIT_MAX_POINTS = 2000
PORTRAIT_SEED = 1102
LOOP_SECONDS = 17.0


def load_config():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def github_get(path: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Jalpilla-Andres-profile-generator",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(API + path, headers=headers)
    with urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_stats():
    fallback = {
        "followers": "—",
        "public_repos": 6,
        "stars": 0,
        "forks": 0,
        "languages": {},
        "last_commit": "",
    }
    try:
        user = github_get(f"/users/{USER}")
        repos = []
        page = 1
        while page <= 10:
            batch = github_get(f"/users/{USER}/repos?per_page=100&page={page}&type=owner&sort=updated")
            repos.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        langs = {}
        stars = forks = 0
        for repo in repos:
            if repo.get("fork"):
                continue
            stars += repo.get("stargazers_count", 0)
            forks += repo.get("forks_count", 0)
            try:
                lb = github_get(f"/repos/{USER}/{repo['name']}/languages")
                for lang, val in lb.items():
                    langs[lang] = langs.get(lang, 0) + int(val)
            except Exception:
                pass
        fallback.update({
            "followers": int(user.get("followers", 0)),
            "public_repos": int(user.get("public_repos", len(repos))),
            "stars": stars,
            "forks": forks,
            "languages": langs,
        })
    except Exception as exc:
        print(f"GitHub refresh skipped: {exc}")
    return fallback


def esc(s):
    return html.escape(str(s), quote=True)


def svg_doc(body, width=1000, height=520, bg="transparent"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<style>
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace}}
.sans{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
</style>
<rect width="100%" height="100%" rx="28" fill="{bg}"/>
{body}
</svg>'''


_PARTICLE_CACHE: dict[str, tuple] = {}


# The loop cycles through four scenes: portrait -> explode -> AI neural net
# -> networking mesh -> back to portrait. `times` marks, in seconds, the
# start/end of each hold; the gaps between are the morph transitions.
_SCENE_TIMES = [0, 2.8, 3.9, 5.1, 6.3, 8.6, 9.8, 12.1, 13.3, LOOP_SECONDS]
# Which of the four states (0=portrait,1=exploded,2=neural,3=network) is
# "active" at each of the keyframe times above -- used to build both the
# particle position values and every opacity track that needs to sync to a
# specific scene (connection lines, captions).
_SCENE_AT_KEYFRAME = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]


def _opacity_track(active_state: int) -> str:
    return ";".join("1" if s == active_state else "0" for s in _SCENE_AT_KEYFRAME)


def _frac(v: float) -> str:
    """Format a 0..1 SMIL keyTime fraction. Needs more precision than pnum's
    1-decimal rounding (fine for pixel coordinates, but at LOOP_SECONDS=17
    with ~1.2s transitions, 0.1-resolution rounding can make two distinct
    keyTimes collide onto the same rounded value and collapse a transition
    to zero duration -- i.e. particles would snap instead of morphing)."""
    return f"{v:.4f}".rstrip("0").rstrip(".")


_SCENE_KEY_TIMES = ";".join(_frac(t / LOOP_SECONDS) for t in _SCENE_TIMES)


def _particle_layer(theme: str, frame_x: float, frame_y: float, frame_w: float, frame_h: float, cyan: str, violet: str) -> tuple[str, str, str, int]:
    """Build the 4-scene particle-portrait <g> and return
    (svg, ai_caption_opacity_values, net_caption_opacity_values, point_count).

    The two returned opacity-value strings let the caller sync its own text
    captions to exactly the AI-scene and networking-scene windows.

    Anti-drift guarantee: `portrait`, `exploded`, `neural` and `network` are
    each computed exactly once (cached across the dark/light calls so both
    themes share identical geometry) and stored as plain arrays. The SMIL
    keyframe list below starts and ends on the *same* `portrait` array
    object, so every loop repeat replays identical numbers -- nothing is
    recomputed, so nothing can drift, soften or accumulate error between
    cycle 1, cycle 2, cycle N.
    """
    if "grid" not in _PARTICLE_CACHE:
        rng = np.random.default_rng(PORTRAIT_SEED)
        grid_w, grid_h = PORTRAIT_GRID
        points, weight = photo_to_points(SOURCE_PHOTO, grid_w, grid_h, PORTRAIT_MAX_POINTS, rng)
        exploded = exploded_state(points, grid_w, grid_h, rng, strength=1.0)
        neural, neural_edges = neural_layers_state(len(points), grid_w, grid_h, rng)
        network, network_edges = network_graph_state(len(points), grid_w, grid_h, rng)
        _PARTICLE_CACHE["grid"] = (
            points, weight, exploded, neural, neural_edges, network, network_edges, grid_w, grid_h,
        )
    points, weight, exploded, neural, neural_edges, network, network_edges, grid_w, grid_h = _PARTICLE_CACHE["grid"]

    sx, sy = frame_w / grid_w, frame_h / grid_h

    def to_px(arr):
        return arr[:, 0] * sx + frame_x, arr[:, 1] * sy + frame_y

    portrait_px, portrait_py = to_px(points)
    exploded_px, exploded_py = to_px(exploded)
    neural_px, neural_py = to_px(neural)
    network_px, network_py = to_px(network)

    def edges_px(edges):
        return [(x1 * sx + frame_x, y1 * sy + frame_y, x2 * sx + frame_x, y2 * sy + frame_y) for x1, y1, x2, y2 in edges]

    key_times = _SCENE_KEY_TIMES

    parts = [f'<clipPath id="portraitClip-{theme}"><rect x="{frame_x:.1f}" y="{frame_y:.1f}" width="{frame_w:.1f}" height="{frame_h:.1f}" rx="4"/></clipPath>']
    parts.append(f'<g clip-path="url(#portraitClip-{theme})">')

    # Connection lines for the AI (neural net) and networking (mesh graph)
    # scenes, drawn behind the dots and faded in/out only during their own
    # window so the rest of the loop reads exactly as before.
    ai_op = _opacity_track(2)
    net_op = _opacity_track(3)
    line_parts = [f'<g stroke="{cyan}" stroke-width=".7" opacity="0" fill="none">']
    for x1, y1, x2, y2 in edges_px(neural_edges):
        line_parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    line_parts.append(
        f'<animate attributeName="opacity" begin="0s" dur="{pnum(LOOP_SECONDS)}s" repeatCount="indefinite" '
        f'calcMode="linear" keyTimes="{key_times}" values="{ai_op}"/></g>'
    )
    line_parts.append(f'<g stroke="{violet}" stroke-width=".7" opacity="0" fill="none">')
    for x1, y1, x2, y2 in edges_px(network_edges):
        line_parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    line_parts.append(
        f'<animate attributeName="opacity" begin="0s" dur="{pnum(LOOP_SECONDS)}s" repeatCount="indefinite" '
        f'calcMode="linear" keyTimes="{key_times}" values="{net_op}"/></g>'
    )
    parts.append("".join(line_parts))

    parts.append(f'<defs><circle id="pdot-{theme}" r="1.05"/></defs>')

    # Bucket particles by a coarse (opacity, color) key so siblings that would
    # otherwise repeat identical static attributes can share one <g> instead
    # of restating fill/opacity on every single <use> -- pure byte trimming,
    # no visual change versus one-attribute-per-element.
    n = len(points)
    buckets: dict[tuple[str, str], list[str]] = {}
    for i in range(n):
        w = float(weight[i])
        color = cyan if (i % 5) else violet  # mostly cyan with a violet accent thread, matches brand palette
        base_op = pnum(0.45 + 0.55 * w)
        p0x, p0y = pnum(portrait_px[i]), pnum(portrait_py[i])
        p1x, p1y = pnum(exploded_px[i]), pnum(exploded_py[i])
        p2x, p2y = pnum(neural_px[i]), pnum(neural_py[i])
        p3x, p3y = pnum(network_px[i]), pnum(network_py[i])
        pos_values = (
            f"{p0x} {p0y};{p0x} {p0y};{p1x} {p1y};{p1x} {p1y};"
            f"{p2x} {p2y};{p2x} {p2y};{p3x} {p3y};{p3x} {p3y};"
            f"{p0x} {p0y};{p0x} {p0y}"
        )
        use = (
            f'<use href="#pdot-{theme}" transform="translate({p0x} {p0y})">'
            f'<animateTransform attributeName="transform" type="translate" begin="0s" '
            f'dur="{pnum(LOOP_SECONDS)}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{pos_values}"/></use>'
        )
        buckets.setdefault((color, base_op), []).append(use)

    for (color, base_op), uses in buckets.items():
        parts.append(f'<g fill="{color}" opacity="{base_op}">{"".join(uses)}</g>')
    parts.append("</g>")
    return "".join(parts), ai_op, net_op, n


def banner(theme: str, cfg: dict | None = None, stats: dict | None = None):
    dark = theme == "dark"
    cfg = cfg or load_config()
    stats = stats or {}
    bg = "#070B14" if dark else "#F7FAFC"
    panel = "#0D1322" if dark else "#FFFFFF"
    panel2 = "#0A0F1C" if dark else "#F1F5F9"
    fg = "#E6EDF3" if dark else "#172033"
    muted = "#7D8CA3" if dark else "#687386"
    grid = "#1B2638" if dark else "#E5EAF0"
    cyan = cfg.get("accent", "#38BDF8")
    violet = cfg.get("accent2", "#8B5CF6")

    W, H = 1180, 628
    frame_x, frame_y, frame_w, frame_h = 44, 100, 400, 472

    particle_svg, ai_scene_op, net_scene_op, point_count = _particle_layer(
        theme, frame_x, frame_y, frame_w, frame_h, cyan, violet
    )

    name = cfg.get("display_name", USER)
    title = cfg.get("title", "")
    location = cfg.get("location", "")
    skills = cfg.get("skills", {})
    languages = cfg.get("languages", {})
    top_langs = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)[:3]
    focus_words = list(skills.keys())[:5]
    projects = cfg.get("projects", [])[:2]

    body = [f'''
<defs>
 <linearGradient id="g-{theme}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{cyan}"/><stop offset="1" stop-color="{violet}"/></linearGradient>
 <filter id="glow-{theme}" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect x="10" y="10" width="{W-20}" height="{H-20}" rx="22" fill="{panel}" stroke="{grid}"/>
<rect x="34" y="34" width="{W-68}" height="34" rx="9" fill="{bg}" stroke="{grid}"/>
<circle cx="56" cy="51" r="6" fill="#FF5F57"/><circle cx="77" cy="51" r="6" fill="#FEBC2E"/><circle cx="98" cy="51" r="6" fill="#28C840"/>
<text x="120" y="56" class="mono" font-size="14" fill="{muted}">{esc(USER.lower())}@profile:~$ ./whoami --live</text>
<circle cx="{W-70}" cy="51" r="4" fill="#FF4D5A"><animate attributeName="opacity" values="1;.35;1" dur="1.6s" repeatCount="indefinite"/></circle>
<text x="{W-58}" y="55" class="mono" font-size="12" font-weight="700" fill="#FF4D5A">LIVE</text>
''']

    # Left: particle portrait frame.
    body.append(
        f'<rect x="{frame_x-6}" y="{frame_y-32}" width="{frame_w+12}" height="{frame_h+44}" rx="8" fill="{panel2}" stroke="{grid}"/>'
        f'<path d="M{frame_x-6} {frame_y-2}H{frame_x+frame_w+6}" stroke="{grid}"/>'
        f'<text x="{frame_x+8}" y="{frame_y-13}" class="mono" font-size="13" font-weight="700" '
        f'fill="{cyan}" letter-spacing="1">ID.PARTICLES</text>'
        f'<text x="{frame_x+frame_w-8}" y="{frame_y-13}" text-anchor="end" class="mono" font-size="11" '
        f'fill="{muted}">{point_count} PTS · DITHER</text>'
    )
    body.append(f'<rect x="{frame_x}" y="{frame_y}" width="{frame_w}" height="{frame_h}" rx="4" fill="{bg}"/>')
    body.append(particle_svg)

    # Caption cycles through all four scenes, each <text> sharing the exact
    # same spot and cross-fading in only during its own scene window (the
    # AI/network windows reuse ai_scene_op/net_scene_op so the words appear
    # in perfect sync with the dots reforming into each shape).
    cap_key_times = _SCENE_KEY_TIMES
    captions = [
        ("reassembles from source · no cross-cycle drift", _opacity_track(0)),
        (f"scatters into {point_count} raw points · zero data loss", _opacity_track(1)),
        ("reforms as a neural net · pattern recognition mode", ai_scene_op),
        ("reforms as a mesh network · always connected", net_scene_op),
    ]
    cap_y = frame_y + frame_h + 24
    for text, op_values in captions:
        body.append(
            f'<text x="{frame_x+8}" y="{cap_y}" class="mono" font-size="11" fill="{muted}" opacity="0">{esc(text)}'
            f'<animate attributeName="opacity" begin="0s" dur="{pnum(LOOP_SECONDS)}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{cap_key_times}" values="{op_values}"/></text>'
        )

    # Right: identity + info panel.
    rx = frame_x + frame_w + 34
    rw = W - 20 - rx
    body.append(f'<rect x="{rx}" y="{frame_y-32}" width="{rw}" height="{frame_h+44}" rx="8" fill="{panel2}" stroke="{grid}"/>')
    body.append(f'<path d="M{rx} {frame_y-2}H{rx+rw}" stroke="{grid}"/>')
    body.append(
        f'<text x="{rx+16}" y="{frame_y-13}" class="mono" font-size="13" font-weight="700" fill="{cyan}" '
        f'letter-spacing="1">SYSTEM.INFO</text>'
    )
    body.append(
        f'<rect x="{rx+rw-158}" y="{frame_y-27}" width="142" height="24" rx="12" fill="{cyan}" opacity=".14" stroke="{cyan}"/>'
        f'<text x="{rx+rw-87}" y="{frame_y-11}" text-anchor="middle" class="mono" font-size="13" font-weight="700" '
        f'fill="{cyan}">@{esc(USER)}</text>'
    )

    ty = frame_y + 34
    body.append(f'<text x="{rx+16}" y="{ty}" class="sans" font-size="38" font-weight="800" fill="{fg}">{esc(name)}</text>')
    ty += 32
    body.append(f'<text x="{rx+16}" y="{ty}" class="mono" font-size="18" fill="{muted}">{esc(title)}  ·  {esc(location)}</text>')
    ty += 36
    body.append(f'<path d="M{rx+16} {ty-16}H{rx+rw-16}" stroke="{grid}"/>')

    ty += 22
    body.append(f'<text x="{rx+16}" y="{ty}" class="mono" font-size="14" fill="{muted}">$ focus --areas</text>')
    ty += 28
    fx = rx + 16
    for word in focus_words:
        wlen = len(word) * 9.1 + 26
        if fx + wlen > rx + rw - 16:
            fx = rx + 16
            ty += 34
        body.append(
            f'<rect x="{fx:.0f}" y="{ty-17}" width="{wlen:.0f}" height="28" rx="14" fill="{bg}" stroke="{grid}"/>'
            f'<text x="{fx+wlen/2:.0f}" y="{ty+2}" text-anchor="middle" class="mono" font-size="13" fill="{fg}">{esc(word)}</text>'
        )
        fx += wlen + 8
    ty += 46
    body.append(f'<path d="M{rx+16} {ty-16}H{rx+rw-16}" stroke="{grid}"/>')

    ty += 20
    body.append(f'<text x="{rx+16}" y="{ty}" class="mono" font-size="14" fill="{muted}">$ top --languages</text>')
    ty += 26
    bar_w = rw - 32 - 130
    for lang, val in top_langs:
        body.append(f'<text x="{rx+16}" y="{ty}" class="mono" font-size="14" fill="{fg}">{esc(lang)}</text>')
        track_x = rx + 118
        body.append(f'<rect x="{track_x}" y="{ty-11}" width="{bar_w}" height="9" rx="4.5" fill="{bg}" stroke="{grid}"/>')
        fill_w = max(bar_w * val / 100, 6)
        body.append(f'<rect x="{track_x}" y="{ty-11}" width="{fill_w:.1f}" height="9" rx="4.5" fill="url(#g-{theme})"/>')
        body.append(f'<text x="{rx+rw-16}" y="{ty}" text-anchor="end" class="mono" font-size="13" fill="{muted}">{val}%</text>')
        ty += 26
    ty += 14
    body.append(f'<path d="M{rx+16} {ty-16}H{rx+rw-16}" stroke="{grid}"/>')

    ty += 20
    body.append(f'<text x="{rx+16}" y="{ty}" class="mono" font-size="14" fill="{muted}">$ projects --featured</text>')
    ty += 26
    for proj in projects:
        pname = proj.get("name", "")
        pdesc = proj.get("description", "")
        body.append(f'<text x="{rx+16}" y="{ty}" class="mono" font-size="14" fill="{cyan}">▸ {esc(pname)}</text>')
        ty += 20
        trimmed = pdesc if len(pdesc) < 62 else pdesc[:59] + "..."
        body.append(f'<text x="{rx+30}" y="{ty}" class="mono" font-size="12.5" fill="{muted}">{esc(trimmed)}</text>')
        ty += 24

    footer_y = frame_y + frame_h + 12
    body.append(f'<path d="M{rx+16} {footer_y-16}H{rx+rw-16}" stroke="{grid}"/>')
    body.append(
        f'<circle cx="{rx+22}" cy="{footer_y}" r="5" fill="#28C840" filter="url(#glow-{theme})"/>'
        f'<text x="{rx+35}" y="{footer_y+4}" class="mono" font-size="13" fill="{fg}">building · learning · shipping</text>'
    )
    body.append(
        f'<text x="{rx+rw-16}" y="{footer_y+4}" text-anchor="end" class="mono" font-size="12" fill="{muted}">'
        "UTC-6 · MX NODE</text>"
    )

    return svg_doc("".join(body), W, H, bg)


def radar_svg(values, title, theme):
    dark = theme == "dark"
    bg = "#0B1220" if dark else "#FFFFFF"
    fg = "#E6EDF3" if dark else "#1A2533"
    muted = "#7D8CA3" if dark else "#687386"
    grid = "#2A3A52" if dark else "#D8E0EA"
    accent = "#38BDF8" if title.lower().startswith("skill") else "#8B5CF6"
    w, h, cx, cy, R = 560, 430, 280, 235, 145
    keys = list(values)
    n = len(keys)
    def pt(r, i):
        a = -math.pi/2 + 2*math.pi*i/n
        return cx + r*math.cos(a), cy + r*math.sin(a)
    body = [f'<rect x="0" y="0" width="560" height="430" rx="24" fill="{bg}"/>', f'<text x="28" y="36" class="sans" font-size="20" font-weight="800" fill="{fg}">{esc(title)}</text>']
    for level in (0.25,0.5,0.75,1.0):
        pts = " ".join(f'{x:.1f},{y:.1f}' for x,y in (pt(R*level,i) for i in range(n)))
        body.append(f'<polygon points="{pts}" fill="none" stroke="{grid}" stroke-width="1"/>')
    for i, k in enumerate(keys):
        x, y = pt(R, i); x0,y0=pt(0,i); body.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="{grid}"/>')
        lx,ly=pt(R+28,i)
        anchor='middle'
        if lx<cx-10: anchor='end'
        elif lx>cx+10: anchor='start'
        body.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="mono" font-size="13" fill="{muted}">{esc(k)}</text>')
        v=values[k]/100
        # fake ring percentage text near axis outer end
        vx,vy=pt(R*v,i)
        body.append(f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="2.8" fill="{accent}"/>')
    data_pts = " ".join(f'{x:.1f},{y:.1f}' for x,y in (pt(R*values[k]/100,i) for i,k in enumerate(keys)))
    body.append(f'<polygon points="{data_pts}" fill="{accent}" fill-opacity=".18" stroke="{accent}" stroke-width="3"/>')
    body.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{accent}"/>')
    body.append(f'<text x="28" y="425" class="mono" font-size="12" fill="{muted}">Self-configured focus map • edit assets/profile.json</text>')
    return "\n".join(body), w, h


def stats_card(stats, theme):
    dark = theme == 'dark'
    bg = '#0B1220' if dark else '#FFFFFF'
    panel = '#101827' if dark else '#F6F8FB'
    fg = '#E6EDF3' if dark else '#1A2533'
    muted = '#7D8CA3' if dark else '#687386'
    cyan='#38BDF8'; violet='#8B5CF6'; grid='#223249' if dark else '#DCE4ED'
    items=[('REPOS',stats['public_repos']),('STARS',stats['stars']),('FORKS',stats['forks']),('FOLLOWERS',stats['followers'])]
    body=[f'<rect width="700" height="290" rx="24" fill="{bg}"/><text x="32" y="42" class="sans" font-size="21" font-weight="800" fill="{fg}">GitHub • numbers at a glance</text>', f'<text x="32" y="67" class="mono" font-size="12" fill="{muted}">@{USER} • generated locally from GitHub API</text>']
    for i,(lab,val) in enumerate(items):
        x=32+(i%2)*324; y=92+(i//2)*88
        body += [f'<rect x="{x}" y="{y}" width="300" height="70" rx="16" fill="{panel}" stroke="{grid}"/>',f'<text x="{x+18}" y="{y+24}" class="mono" font-size="11" fill="{muted}">{lab}</text>',f'<text x="{x+18}" y="{y+55}" class="sans" font-size="25" font-weight="800" fill="{cyan if i%2==0 else violet}">{val}</text>']
    body += [f'<line x1="32" y1="272" x2="668" y2="272" stroke="{grid}"/>',f'<text x="32" y="286" class="mono" font-size="10" fill="{muted}">Auto-refresh via .github/workflows/update-profile.yml</text>']
    return svg_doc('\n'.join(body), 700, 300, bg)


def language_card(stats, theme):
    dark = theme=='dark'; bg='#0B1220' if dark else '#FFFFFF'; fg='#1A2533' if not dark else '#E6EDF3'; muted='#687386' if not dark else '#7D8CA3'; grid='#DCE4ED' if not dark else '#223249'; colors=['#38BDF8','#8B5CF6','#22C55E','#F59E0B','#F43F5E','#14B8A6']
    langs=stats.get('languages') or {}
    if not langs:
        langs={k: v for k,v in load_config().get('languages',{}).items()}
    top=sorted(langs.items(), key=lambda kv:kv[1], reverse=True)[:6]
    total=sum(v for _,v in top) or 1
    body=[f'<rect width="760" height="190" rx="24" fill="{bg}"/>',f'<text x="30" y="36" class="sans" font-size="20" font-weight="800" fill="{fg}">Most used languages</text>']
    x=30; y=62; width=700; h=18
    start=0
    for i,(lang,v) in enumerate(top):
        seg=width*v/total
        body.append(f'<rect x="{x+start:.1f}" y="{y}" width="{max(seg,2):.1f}" height="{h}" fill="{colors[i%len(colors)]}"/>'); start+=seg
    yy=112
    for i,(lang,v) in enumerate(top):
        pct=v/total*100
        col=colors[i%len(colors)]
        colx=30+(i%3)*240; row=yy+(i//3)*30
        body += [f'<circle cx="{colx+5}" cy="{row-4}" r="5" fill="{col}"/>', f'<text x="{colx+18}" y="{row}" class="mono" font-size="12" fill="{fg}">{esc(lang)}</text>', f'<text x="{colx+165}" y="{row}" class="mono" font-size="11" fill="{muted}">{pct:.1f}%</text>']
    return svg_doc('\n'.join(body),760,190,bg)


def main():
    cfg=load_config(); stats=fetch_stats();
    (ASSETS/'banner-dark.svg').write_text(banner('dark',cfg,stats),encoding='utf-8')
    (ASSETS/'banner-light.svg').write_text(banner('light',cfg,stats),encoding='utf-8')
    body,w,h=radar_svg(cfg['skills'],'Skill signals','dark'); (ASSETS/'radar-dark.svg').write_text(svg_doc(body,w,h,'#0B1220'),encoding='utf-8')
    body,w,h=radar_svg(cfg['skills'],'Skill signals','light'); (ASSETS/'radar-light.svg').write_text(svg_doc(body,w,h,'#FFFFFF'),encoding='utf-8')
    body,w,h=radar_svg(cfg['languages'],'Language signals','dark'); (ASSETS/'radar-langs-dark.svg').write_text(svg_doc(body,w,h,'#0B1220'),encoding='utf-8')
    body,w,h=radar_svg(cfg['languages'],'Language signals','light'); (ASSETS/'radar-langs-light.svg').write_text(svg_doc(body,w,h,'#FFFFFF'),encoding='utf-8')
    (ASSETS/'card-stats-dark.svg').write_text(stats_card(stats,'dark'),encoding='utf-8')
    (ASSETS/'card-stats-light.svg').write_text(stats_card(stats,'light'),encoding='utf-8')
    (ASSETS/'metrics.languages.svg').write_text(language_card(stats,'dark'),encoding='utf-8')
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
