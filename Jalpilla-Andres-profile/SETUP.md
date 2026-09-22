# Setup — Jalpilla-Andres GitHub Profile

1. Create a **public** repository named exactly `Jalpilla-Andres` under the account `Jalpilla-Andres`.
2. Upload everything from this folder, keeping the directory structure.
3. Commit to the `main` branch.
4. Open `https://github.com/Jalpilla-Andres` and the README should render as your profile README.
5. The workflow in `.github/workflows/update-profile.yml` refreshes the stats and language SVGs weekly and whenever `assets/profile.json` or the generator changes.

## Personalize the radar charts

Edit `assets/profile.json` and change the values in `skills` and `languages` from 0–100. The next workflow run regenerates the charts.

The `projects` section in `assets/profile.json` is useful as a single source of truth for future automation, but the displayed project table currently lives in `README.md` so you can control its order and wording.

## Local preview

Run:

```bash
python3 scripts/generate.py
python3 -m http.server 8000
```

Then open `http://localhost:8000/preview.html`.

## Notes

- The banner and radar graphics are local SVG files, so they are not dependent on a third-party image host.
- GitHub-hosted Actions refresh the numeric cards using your repository data.
- The README still uses `readme-typing-svg`, `skillicons.dev`, and `komarev.com` for the lightweight animated typing, tech icons and profile-view badge.
