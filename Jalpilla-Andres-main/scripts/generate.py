#!/usr/bin/env python3
"""Generate profile SVG assets for Jalpilla-Andres.

No third-party packages required. When GH_TOKEN is available, repo/user/language
statistics are refreshed from GitHub's public REST API.
"""
from __future__ import annotations
import json, math, os, html
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CFG_PATH = ASSETS / "profile.json"
USER = "Jalpilla-Andres"
API = "https://api.github.com"


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


def banner(theme: str):
    dark = theme == "dark"
    bg = "#070B14" if dark else "#F7FAFC"
    panel = "#0D1322" if dark else "#FFFFFF"
    fg = "#E6EDF3" if dark else "#172033"
    muted = "#7D8CA3" if dark else "#687386"
    grid = "#1B2638" if dark else "#E5EAF0"
    cyan = "#38BDF8"
    violet = "#8B5CF6"
    body = f'''
<defs>
 <linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{cyan}"/><stop offset="1" stop-color="{violet}"/></linearGradient>
 <filter id="glow"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect x="10" y="10" width="980" height="500" rx="24" fill="{panel}" stroke="{grid}"/>
<g opacity=".35">'''
    for x in range(50, 1000, 50):
        body += f'<path d="M{x} 60V475" stroke="{grid}"/>'
    for y in range(75, 500, 40):
        body += f'<path d="M30 {y}H970" stroke="{grid}"/>'
    body += f'''</g>
<rect x="35" y="35" width="930" height="36" rx="10" fill="{bg}" stroke="{grid}"/>
<circle cx="58" cy="53" r="6" fill="#FF5F57"/><circle cx="80" cy="53" r="6" fill="#FEBC2E"/><circle cx="102" cy="53" r="6" fill="#28C840"/>
<text x="125" y="59" class="mono" font-size="14" fill="{muted}">andres@jalpilla:~$ ./profile.sh --live</text>
<text x="55" y="135" class="mono" font-size="22" fill="{cyan}">$ whoami</text>
<text x="55" y="180" class="sans" font-size="42" font-weight="800" fill="{fg}">ANDRES JALPILLA</text>
<text x="55" y="217" class="mono" font-size="20" fill="{muted}">Telematics Engineer  •  Jr. Developer  •  Java</text>
<text x="55" y="254" class="mono" font-size="15" fill="{fg}">$ focus --on</text>
<text x="55" y="285" class="mono" font-size="18" fill="{cyan}">software  networking  systems  cybersecurity  AI</text>
<text x="55" y="345" class="mono" font-size="15" fill="{muted}">$ status</text>
<circle cx="142" cy="341" r="7" fill="#28C840" filter="url(#glow)"/>
<text x="160" y="347" class="mono" font-size="17" fill="{fg}">building projects • learning • shipping</text>
<g transform="translate(650 142)">
 <rect width="255" height="245" rx="18" fill="{bg}" stroke="{grid}"/>
 <text x="20" y="35" class="mono" font-size="14" fill="{muted}">quick_profile.json</text>
 <text x="20" y="75" class="mono" font-size="15" fill="{violet}">"location"</text><text x="112" y="75" class="mono" font-size="15" fill="{fg}">: "Mexico 🇲🇽"</text>
 <text x="20" y="108" class="mono" font-size="15" fill="{violet}">"role"</text><text x="85" y="108" class="mono" font-size="15" fill="{fg}">: "Engineer"</text>
 <text x="20" y="141" class="mono" font-size="15" fill="{violet}">"main"</text><text x="85" y="141" class="mono" font-size="15" fill="{fg}">: "Java"</text>
 <text x="20" y="174" class="mono" font-size="15" fill="{violet}">"mindset"</text><text x="112" y="174" class="mono" font-size="15" fill="{fg}">: "build"</text>
 <text x="20" y="207" class="mono" font-size="15" fill="{violet}">"next"</text><text x="82" y="207" class="mono" font-size="15" fill="{fg}">: "cybersecurity"</text>
</g>
<rect x="55" y="398" width="540" height="3" rx="2" fill="url(#g)"/>
<text x="55" y="438" class="mono" font-size="14" fill="{muted}">~$ git commit -m "keep learning, keep building"</text>
<text x="55" y="465" class="mono" font-size="14" fill="{cyan}">✓ profile loaded successfully</text>
'''
    return svg_doc(body, 1000, 520, bg)


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
    (ASSETS/'banner-dark.svg').write_text(banner('dark'),encoding='utf-8')
    (ASSETS/'banner-light.svg').write_text(banner('light'),encoding='utf-8')
    body,w,h=radar_svg(cfg['skills'],'Skill signals','dark'); (ASSETS/'radar-dark.svg').write_text(svg_doc(body,w,h,'#0B1220'),encoding='utf-8')
    body,w,h=radar_svg(cfg['skills'],'Skill signals','light'); (ASSETS/'radar-light.svg').write_text(svg_doc(body,w,h,'#FFFFFF'),encoding='utf-8')
    body,w,h=radar_svg(cfg['languages'],'Language signals','dark'); (ASSETS/'radar-langs-dark.svg').write_text(svg_doc(body,w,h,'#0B1220'),encoding='utf-8')
    body,w,h=radar_svg(cfg['languages'],'Language signals','light'); (ASSETS/'radar-langs-light.svg').write_text(svg_doc(body,w,h,'#FFFFFF'),encoding='utf-8')
    (ASSETS/'card-stats-dark.svg').write_text(stats_card(stats,'dark'),encoding='utf-8')
    (ASSETS/'card-stats-light.svg').write_text(stats_card(stats,'light'),encoding='utf-8')
    (ASSETS/'metrics.languages.svg').write_text(language_card(stats,'dark'),encoding='utf-8')
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
