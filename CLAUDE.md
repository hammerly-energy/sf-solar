# CLAUDE.md

Project context for Claude Code. Read this at the start of every session.

## What this project is

A near-real-time estimate of aggregate rooftop solar generation across San Francisco, published as an interactive website.

The pipeline: identify SF's installed PV fleet → model each system's geometry → compute clear-sky output from sun position → derate for real cloud cover → sum to citywide MW → display on a map.

Eventually this extends to a 0–6h forecast. Not yet.

## Who I am

I'm Christian. **I don't have a software engineering background.** This is my first real coding project.

How to work with me:

- **Explain as you go.** When you write code, tell me what it does and why in plain language. I want to understand the project, not just possess it.
- **Prefer boring, obvious solutions** over clever or idiomatic ones. Readability beats elegance every time here.
- **Don't add abstractions I didn't ask for.** No class hierarchies, no plugin systems, no config frameworks. Flat scripts are fine.
- **Tell me when I'm asking for something wrong.** If I request something that's a bad idea, push back and explain why rather than just building it.
- **Small steps.** One working change at a time, committed, before moving on.
- If a step needs a tool or account I don't have yet, stop and tell me — don't work around it silently.

## The most important rule: silent wrongness

**Physics bugs do not crash.** A sign error, a unit mixup, or a timezone bug produces a smooth, plausible, completely wrong curve. I cannot catch these by reading the code — I don't have the background.

So:

- **After any change to the physics, plot the result and check it against a sanity range.** Don't just confirm the code runs.
- **State your assumptions out loud** when you make them, especially about units, conventions, and time.
- **When something looks suspiciously clean, say so.** I'd rather hear "this matches too well, we may be validating against our own model" than get a confident thumbs up.
- Never fabricate or hardcode a data value to make something run. If data is missing, say it's missing.

### Sanity checks to apply

| Quantity | Expected range |
|---|---|
| Clear-sky GHI, SF, solar noon, summer | ~950–1050 W/m² |
| Peak AC output vs. nameplate DC | ~70–85% |
| Daily energy, 5 kW system, clear summer day | ~25–35 kWh |
| Annual capacity factor, fixed-tilt SF | ~15–18% |
| Output at night | Exactly zero |

If a number is off by 10×, it's units. If it's off by a few hours, it's timezone. If it peaks at the wrong time of day, it's azimuth convention or UTC.

### Conventions that cause bugs — be explicit about these

- **Time:** store and compute everything in **UTC**. Convert to `America/Los_Angeles` only for display. Never do arithmetic on local time. Watch DST.
- **Azimuth:** pvlib's `surface_azimuth` is **0 = North, 90 = East, 180 = South, 270 = West**. The CA DG Stats dataset may use a different convention — **verify this explicitly against the data key before trusting any azimuth**, don't assume. Getting this wrong tilts the whole daily curve east or west.
- **Units:** be explicit in every variable name. `power_kw` vs `energy_kwh` vs `irradiance_wm2`. Capacity is kW (power); generation over an interval is kWh (energy). Mixing these is the most common bug in this domain.
- **DC vs AC:** systems have both a DC nameplate and an AC inverter limit. Clipping happens at the AC limit and matters at peak hours.

## Key decisions already made

Don't relitigate these without telling me why:

- **GOES-18** is the satellite (operational GOES-West). **Not GOES-16** — that's been off the air since April 2025. Many tutorials are stale on this.
- **pvlib-python** for all solar physics. Don't hand-roll solar position or irradiance math.
- **No backend server.** GitHub Actions cron → writes `data/latest.json` → commits → GitHub Pages serves a static site that fetches it.
- **Plain HTML + JavaScript** for the frontend. Chart.js for time series, MapLibre for maps. **No React**, no build step, no bundler.
- **`uv`** for Python environment and dependency management.
- Fleet data comes from the CA DG Stats **Interconnected Project Sites** dataset (not the Applications dataset).

## Scope — deferred until later

Not in scope right now. If they'd help, mention it, but don't build them:

- LiDAR-derived horizon/shading profiles
- Archetype clustering (probably unnecessary — SF's fleet may be small enough to simulate directly)
- Satellite parallax correction
- Advection / forecasting
- Marine-layer burn-off model

## Where I am now

**Level 1** — one hypothetical SF rooftop, clear-sky output for one day, plotted.

Full staged plan is in `docs/sf-solar-roadmap.md`. Technical design is in `docs/sf-solar-nowcast-architecture.md`. Read both if you need context beyond this file.

**Update this section as levels complete.**

## Workflow

- Commit after every working change, with a message explaining what changed and why.
- Keep scripts runnable standalone: `uv run python scripts/whatever.py`.
- Data files go in `data/`. Raw downloads in `data/raw/` and **gitignored** — they're large and re-downloadable.
- If a script takes more than a few seconds, cache its output to disk so I'm not re-running it constantly.

## Repo layout

```
sf-solar/
├── CLAUDE.md
├── README.md
├── docs/            # architecture + roadmap
├── scripts/         # standalone Python steps
├── data/
│   ├── raw/         # gitignored
│   └── latest.json  # published output
├── site/            # static website (GitHub Pages)
└── .github/workflows/   # scheduled update job
```
