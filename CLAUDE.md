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
- **GitHub Pages deploys via a GitHub Actions workflow** (`.github/workflows/pages.yml`), not the classic "deploy from branch" setting. That setting only serves the repo root or a `/docs` folder, and the site lives in `site/`. The workflow packages `site/` as the Pages artifact on every push to `main`.

## Scope — deferred until later

Not in scope right now. If they'd help, mention it, but don't build them:

- LiDAR-derived horizon/shading profiles
- Archetype clustering (probably unnecessary — SF's fleet may be small enough to simulate directly)
- Satellite parallax correction
- Advection / forecasting
- Marine-layer burn-off model

## Where I am now

**Level 3, in progress.** Level 2's single-rooftop curve is live at https://hammerly-energy.github.io/sf-solar/. For Level 3, the real SF PV fleet (PG&E interconnection data, `scripts/level3_fleet.py`) is loaded, cleaned, gap-filled, and visualized (81.1 MW DC, 13,485 systems, ZIP-level map). Still to do: per-system pvlib clear-sky simulation summed to a citywide MW curve, then update the site.

Currently on a side quest exploring building-level (not ZIP-level) detail — see "Building-level detection detour" below. This is *not* blocking Level 3; the ZIP-level model is the real deliverable and can ship independent of how the detour resolves.

Full staged plan is in `docs/sf-solar-roadmap.md`. Technical design is in `docs/sf-solar-nowcast-architecture.md`. Read both if you need context beyond this file.

**Update this section as levels complete.**

## Building-level detection detour

Explored going beyond ZIP-code fleet data to real per-building locations + LiDAR-derived roof geometry (tilt/azimuth), instead of the statistical ZIP-level model. Recording the dead ends so they aren't re-tried blind:

**Dead ends found (all for San Francisco specifically):**

- **CA DG Stats has no address field** — only `Service City`/`Zip`/`County`. Can't geocode individual systems from our main fleet dataset at all.
- **SF DBI building permits** (`data.sfgov.org`, dataset `i98e-djp9`) do have real addresses + lat/lon, but only cover **~8–12%** of the known fleet (1,516 complete/issued solar permits vs. 13,485 interconnected systems citywide; same ratio held in the Richmond 94118/94121 test). Even within that subset, capacity isn't reliably machine-parseable from the free-text `description` field, and keyword search produces false positives (reroof shingle products literally named "Solaris", prep-only permits, battery-only retrofits with no new PV).
- **DeepSolar's public datasets** (original 2018 release and DeepSolar-3M) only publish **tract/blockgroup-level aggregates**, never point-level detections — no public dataset solves the "where are the panels" problem directly.
- **Running DeepSolar-3M's own model ourselves** works mechanically — weights are downloadable via `gdown` from their Google Drive (linked from their GitHub README), and the pipeline runs end-to-end on a Mac (MPS backend). Requires `transformers==4.46.3` pinned exactly (the segmentation checkpoint is a pickled full object, version-coupled to their exact class definitions) and `torch>=2.3` (their documented `torch==2.2.2` breaks under NumPy 2.x — `torch.from_numpy` silently fails). But **San Francisco has no viable free imagery source at the resolution the model needs** (~0.24 m/px, "Google Maps zoom ≥19"): NAIP tops out at 0.6 m/px, SF's own GIS imagery is commercially licensed (Pictometry/EagleView), and scraping Google/Bing/Mapbox for this would violate their terms of service. Commercial imagery (Vexcel, 7.5cm) has no public pricing — sales-quote-only, enterprise-budget territory.
- **Tested anyway on NAIP** (upsampled 0.6m → ~0.24m to match the model's expected scale): even a positive-control case with zero ambiguity — SFPUC's Sunset Reservoir solar farm, ~24,000 panels, clearly visible in the imagery to the naked eye — scored **p(PV) < 0.11 on every tile**, far under the model's own 0.5 decision threshold. Inconclusive whether that's resolution loss or a domain mismatch (the model is trained on rooftop-mounted arrays; Sunset Reservoir is ground-mounted on a reservoir cover).

**Pivoted to Zurich, Switzerland to validate the method** (San Francisco itself still has no imagery fix — this is proving out the pipeline elsewhere, not solving SF's blocker):

- **Data:** SWISSIMAGE orthophotos at **10 cm/px** (finer than the model needs — we *downsample* to match, the opposite of the NAIP upsampling) + swissSURFACE3D DSM raster at **0.5 m/px**, both via a public STAC API (`data.geo.admin.ch`). Genuinely open terms of use (commercial use, redistribution, and processing all explicitly permitted; attribution required: "©swisstopo").
- **Result: it works.** Test area = Wiedikon, a dense residential district (~600m box, 24 tiles, no pre-picked positive control — just a broad scan). Classifier scored several tiles **above its own 0.5 decision threshold** (top two: 0.758, 0.739), vs. a max of 0.10 anywhere in the San Francisco/NAIP test. Segmentation on the plausible tiles produced masks that visibly track real panel rows on the two largest, most obvious arrays (1,318 m² and 583 m² — both confirmed by eye against the source imagery, not just the model's say-so).
- **Full pipeline now runs end-to-end**, three stages: classify (ViT-MAE-large + LoRA) → segment (SegFormer) → fit a roof plane (least-squares on swissSURFACE3D DSM pixels under each segmented footprint) → tilt/azimuth. Confirmed working for 4 of 8 detected arrays, including both of the large confirmed ones:
  - 1,318 m² array: tilt 4.2°, azimuth 155° (~SSE)
  - 583 m² array: tilt 4.6°, azimuth 335° (~NNW)
  - Both nearly flat — plausible for large commercial/industrial roofs, but note azimuth is inherently noisy at low tilt (same reason flat SF systems record azimuth=0 as a meaningless placeholder — see `level3_fleet.py`'s gap-fill logic). Don't over-read the exact degree values on these two.
  - Two smaller arrays also fit: tilt 20.8°/azimuth 7° (~N — a poor solar orientation, though real small mis-oriented installs do exist) and tilt 13.4°/azimuth 58° (~ENE).
- **Known bug, not yet fixed:** the other 4 arrays failed with "footprint too small to fit a plane" — this is **not** a real size problem, it's `fetch_dsm_chip()` in `scripts/level3_cv_detection.py` only fetching `hrefs[0]`, the first of 8 DSM tiles covering the bbox, instead of mosaicking all of them. Any array footprint outside that one tile silently gets an out-of-bounds (truncated) window. **Next step if resuming this thread:** fix the DSM fetch to mosaic all covering tiles (or pick the right tile per array) for full coverage.

Backup candidates if Zurich hits a wall: Netherlands (PDOK/AHN, 25cm ortho + 6–10 pts/m² LiDAR, fully open) and Massachusetts, USA (MassGIS, 15cm ortho + QL1/QL2 LiDAR with DSM, public domain).

Prototype code: `scripts/level3_cv_detection.py` (explicitly marked experimental, not part of the main pipeline). Model weights live in `data/raw/deepsolar_weights/` (gitignored, ~1.2GB — re-download via `gdown` from the Google Drive link in the DeepSolar-3M GitHub README if missing). Run with:
```
uv run --isolated --with "torch>=2.3" --with transformers==4.46.3 --with rasterio \
  --with pystac-client --with planetary-computer --with matplotlib --with requests \
  python scripts/level3_cv_detection.py zurich
```
(`--isolated` matters — this repo's own `.venv` has a newer NumPy that conflicts with these pins if you don't isolate. Modes: `zurich`, `smoke` (SF Richmond, known-broken), `control` (SF Sunset Reservoir, known-broken).)

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
