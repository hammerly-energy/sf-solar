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
- **Full pipeline now runs end-to-end**, three stages: classify (ViT-MAE-large + LoRA) → segment (SegFormer) → fit roof plane(s) (least-squares on swissSURFACE3D DSM pixels under each segmented footprint) → tilt/azimuth. Of the 8 tiles scoring above the 0.2 segmentation threshold, 7 got a geometry fit and 1 had no PV pixels segmented (a real "classifier probably wrong" case, not a data problem).
- **Two data/geometry bugs found and fixed, in order — each caught by actually plotting the result, not by reading the numbers:**
  1. `fetch_dsm_chip()` originally only fetched `hrefs[0]`, the first of 8 DSM tiles covering the bbox, instead of mosaicking all of them — footprints outside that one tile got a truncated window and sometimes wrongly reported "too small to fit a plane." Fixed with `rasterio.merge.merge()` over all covering tiles.
  2. Even after that fix, `fit_roof_plane()` was fit over the DSM pixels in the **rectangular bounding box** of a segmented footprint, not the footprint's actual (often irregular, often disconnected) shape — visible only once a 3D plot of one building (`save_building_detail()`) showed the fit swallowing a ~20m elevation drop nowhere near the roof. Fixed by rasterizing the real segmentation mask onto the DSM grid (`footprint_mask` in `add_roof_geometry()`) and fitting only those pixels.
  3. That still wasn't enough: a footprint's DSM pixels can be **spatially disconnected** — real gaps between panel rows, or a roof with genuinely different pitches on different sections, or (worst case) a spurious false-positive blob elsewhere in the tile lumped into the same detected "array." Fitting one plane across disconnected clusters produced physically impossible results (one array fit at 73.9° tilt — essentially a wall, not a roof). Fixed by connected-component clustering (`scipy.ndimage.label`) on the rasterized mask: each spatially-connected cluster of DSM pixels now gets its own plane fit (`a["roof_planes"]`, sorted largest-first; the largest is still exposed as `a["tilt_deg"]`/`a["azimuth_deg"]` for existing single-plane consumers).
- **Within-cluster outlier rejection, with a guardrail:** even a single connected cluster can contain a few stray LiDAR returns (antenna/vent/wire/bird strikes) that pull the fit. `fit_roof_plane()` now fits, drops points >`OUTLIER_RESIDUAL_M` (30cm) from the plane, and refits, up to 3 times. Tested against every detected array, unconditional trimming turned out to be its own bug: on a genuinely rougher or multi-building roof, "a few artifacts" and "this isn't really one flat plane" look identical point-by-point, and naive trimming quietly discarded up to 35% of a cluster to force a falsely-tidy fit. `MAX_OUTLIER_FRACTION` (20%) caps this: if cleaning up a cluster would mean discarding more than that share, the fit keeps its pre-trim result instead of fabricating a clean answer from a cherry-picked subset. Verified by visualizing both the case that should get cleaned (a 112 m² array: 2 stray points correctly excluded) and the case that shouldn't (a 583 m² array spanning real multi-section roof structure: no longer forced into one artificially clean plane).
- **Net effect of all these fixes:** most Wiedikon roofs turned out to be much flatter than the pre-fix numbers suggested — several "confidently pitched" arrays (e.g. 17.7°) were actually near-flat once the fit stopped including off-roof terrain. This is exactly the kind of clean-looking-but-wrong number the project's sanity-check discipline exists to catch; the fix was only found by visualizing one building's raw LiDAR points next to its segmentation mask, not by inspecting the printed tilt/azimuth values, which looked perfectly plausible on their own.
- **Touching-but-differently-tilted facets, split properly:** connected-component clustering only splits by spatial adjacency, so several individually-tilted rack-mounted rows sitting close together (touching in the mask) still came out as one meaningless averaged plane. `fit_group_planes()` fixes this with sequential-plane-peeling: fit one plane, and if enough of the leftover (rejected) points still form a spatially-connected group, recurse on that as a new facet. On the 1,322 m² roof this took it from "1 plane, 610 unexplained outliers" to "11 separate facets, 45 genuinely unexplained points" — most of what looked like noise was actually real, distinct roof/rack sections. **Known gap:** this only splits facets that differ by more than `OUTLIER_RESIDUAL_M` (30cm) from a single compromise plane — a small roof with several short, close-together tilted rows can still average out to within 30cm of one plane and never trigger a split (confirmed on a 97 m² test case with 4 visually-distinct tilted rows that still fit as one plane). Catching that needs local surface-normal segmentation, not distance-from-fit; documented as a gap, not built.
- **Registration offset check — no evidence of misalignment.** After noticing outlier clusters concentrating near roof edges, checked whether that's real surface structure or the SWISSIMAGE/DSM datasets being spatially offset from each other. `check_registration_offset()` computes independent edge-strength maps (DSM height-gradient vs. the mask's own boundary gradient) and finds the pixel shift maximizing their normalized cross-correlation, per array — if most buildings agree on the same shift, that's a real offset; if scattered, it isn't. Result across both test areas: **no consistent offset**. Altstetten's larger, more textured roofs gave enough edge signal to trust 11/13 shift estimates, and even there the best shifts were completely scattered (no shift repeated more than once). Combined with the single-building visual check below, the outlier clusters are better explained by real surface structure (raised panel racking) than by data misregistration. (First attempt at this check was flawed — raw mean gradient at the shifted boundary pinned against the search range's edge in 8/14 cases instead of converging inside it; rebuilt with normalized cross-correlation, which is pattern-based and bounded, not magnitude-based.)
- **Four diagnostic plots + one console diagnostic**, regenerated on every `zurich`/`altstetten` run:
  - `data/level3_cv_{mode}_roof_geometry.png` — wind-rose of all detected arrays (azimuth = compass angle, radius = tilt, size = panel area, color = classifier confidence). Near-flat (<2°) arrays get a dashed ring since azimuth is noise-dominated there.
  - `data/level3_cv_{mode}_building_detail.png` — one building (currently: highest-confidence detection) in two views: the SWISSIMAGE tile with its segmentation mask, and a 3D plot of the raw DSM heights with each fitted plane's cluster of points highlighted in its own color.
  - `data/level3_cv_{mode}_registration_check.png` — sanity check that SWISSIMAGE and the DSM are actually co-registered (not just both claiming EPSG:2056): draws the same segmentation footprint outline over both datasets in shared real-world coordinates. Checked on Wiedikon: no evidence of misalignment — a small rooftop feature (vent/drain) visible in the photo lines up with a matching height bump in the DSM at the same coordinates.
  - `data/level3_cv_{mode}_outlier_check.png` — shows exactly where rejected outlier points sit, on both the orthophoto and the DSM, for whichever array has the most rejected outliers. Lets you eyeball whether a rejection corresponds to something real and visible or looks like plain noise.
  - `check_registration_offset()` — console-only (no plot): the quantitative registration check described above, printed per array plus an aggregate verdict.
  - All three are gitignored (`data/level3_cv_*.png`) — regenerate by rerunning the script, not committed as static artifacts.
- **Second test area added: Altstetten (`altstetten` mode).** Wiedikon turned out almost entirely flat commercial/industrial roofs, not much of a stress test. Picking a second area properly (not guessing) took three tries:
  1. First pass: raw density of OpenStreetMap `generator:source=solar` tags across the city. The densest hotspot turned out to be ~90 individually-mapped panel nodes on **one** flat service building — an OSM mapping artifact (one enthusiastic mapper), not neighborhood-wide adoption. Landed the bbox mostly on a sports field and a highway.
  2. Second pass: cross-referenced actual building-footprint density against nearby solar-point density, to distinguish "many small buildings" from "one heavily-mapped roof." Found a much better cell (95 distinct buildings + 91 solar points within 300m) — but it turned out to center on **Bändlistrasse, next to the ARA Werdhölzli wastewater treatment plant**, not real houses. The high-confidence detections (13 arrays, 0.8–0.99 confidence, genuine tilt variety: 4.3°, 9.3°, 9.6°) are almost all on treatment-plant buildings; the actual residential blocks visible in the same frame scored ~0.001–0.03, essentially nothing.
  3. Decision (confirmed with Christian): kept this area anyway. It genuinely delivers what the stress test needed — real pitch variety and plenty of confirmed solar — even though the source is a utility complex rather than single-family homes. `ZURICH_ALTSTETTEN = (8.4747, 47.4012)` in `scripts/level3_cv_detection.py` has the full story in a code comment.
  - **Two more bugs found while chasing this down, both fixed:**
    - `fetch_swissimage_chip()` had the *exact same* single-tile bug `fetch_dsm_chip()` had before its earlier fix — only reading `items[0]` instead of mosaicking. Altstetten's bbox happened to straddle a SWISSIMAGE tile boundary (Wiedikon's didn't), silently truncating the fetched image to a fraction of the requested width and crashing the tile grid layout. Fixed the same way: `rasterio.merge.merge()` over all covering tiles, newest-first so overlaps still prefer the most recent acquisition.
    - The wind-rose's label-decluttering (added for Wiedikon) had a latent bug that only showed up with more detected arrays: it unwrapped azimuths past 360° by comparing each point's raw angle to the *previous label's already-pushed* position instead of to the *previous point's raw* position, so a raw angle could look like a false "wraparound" mid-walk and get shifted by multiples of 360° — landing some labels back on top of earlier, unrelated ones. Fixed by unwrapping the raw azimuth sequence once, cleanly, before doing any pushing.

### Open findings — investigated, NOT yet fixed (as of 2026-08-17)

Verified against the live STAC API; recorded here because the code still has the bug.

- **We are silently using 2018 elevation data against 2025 imagery.** Every DSM tile over both test areas exists in *both* a 2018 and a 2024 version. `fetch_dsm_chip()` collects every covering `.tif` with **no date sorting**, and `rasterio.merge` keeps the *first* dataset's pixels where tiles overlap — and the STAC API returns 2018 first. So every run so far has fit planes to the 2018 surface. (`fetch_swissimage_chip()` already sorts newest-first; `fetch_dsm_chip()` never got the same treatment.) **Fix is small:** sort items by `properties.datetime` descending before building the href list, same as the imagery fetch. That closes the imagery/elevation gap from 7 years to 1.
  - **Why this matters beyond tidiness:** any array installed 2019–2025 appears in the 2025 photo but sits on a 2018 DSM that predates it — so we'd be fitting a plane to the roof *as it was before the panels existed*. This is a strong candidate explanation for several things chased in this session: outlier clusters under panel footprints, "panels raised above the roof deck," and especially the per-building registration shifts that scattered randomly instead of agreeing on one direction. The NCC check's "no consistent global offset" conclusion may be right about there being no *global* offset while still missing a real *per-building, temporal* mismatch. **Re-run both test areas after fixing the date selection before trusting any of the current tilt/azimuth numbers.**
- **Higher-resolution LiDAR is available and we're not using it.** swissSURFACE3D publishes two products, and we've only ever used the derived raster:

  | Collection | What it is | Density | Format |
  |---|---|---|---|
  | `ch.swisstopo.swisssurface3d-raster` | what the code uses now | 4.0 samples/m² | 0.5 m GeoTIFF |
  | `ch.swisstopo.swisssurface3d` | the native point cloud | **28.7 points/m²** | 2024: COPC (`.copc.laz`), 2018: `.las.zip` |

  ~7× denser, ~0.19 m average point spacing vs 0.5 m. Density read directly off a 2024 tile's LAS 1.4 header (28,667,671 points over a 1000×1000 m tile), not from documentation. The 2024 files are **COPC** (Cloud Optimized Point Cloud) and the server honours HTTP range requests (verified: returns `206`), so a single building's points can be read without downloading the ~1 km tile — the same trick as Cloud-Optimized GeoTIFF. Would also give multiple returns per pulse and classification codes, neither of which survive into the raster.
  - Most relevant to the **known gap** above (small roofs whose several tilted rows all fit within 30 cm of one compromise plane, so `fit_group_planes()` never splits them): at 0.5 m the 97 m² test case only gets ~20 samples across its ~10 m span; at 0.19 m spacing it would get ~50, with enough points *per row* to fit each row separately.

Backup candidates if Zurich hits a wall: Netherlands (PDOK/AHN, 25cm ortho + 6–10 pts/m² LiDAR, fully open) and Massachusetts, USA (MassGIS, 15cm ortho + QL1/QL2 LiDAR with DSM, public domain). Note both are *lower* density than the swissSURFACE3D point cloud (28.7 pts/m²) — there's no resolution argument for switching regions; the densest data we've found is already where we're working.

Prototype code: `scripts/level3_cv_detection.py` (explicitly marked experimental, not part of the main pipeline). Model weights live in `data/raw/deepsolar_weights/` (gitignored, ~1.2GB — re-download via `gdown` from the Google Drive link in the DeepSolar-3M GitHub README if missing). Run with:
```
uv run --isolated --with "torch>=2.3" --with transformers==4.46.3 --with rasterio \
  --with pystac-client --with planetary-computer --with matplotlib --with requests --with scipy \
  python scripts/level3_cv_detection.py zurich      # Wiedikon -- flat roofs
```
(swap the trailing `zurich` for `altstetten` to run the Werdhölzli-adjacent, varied-tilt area instead.
`--isolated` matters — this repo's own `.venv` has a newer NumPy that conflicts with these pins if you
don't isolate. Modes: `zurich`, `altstetten`, `smoke` (SF Richmond, known-broken), `control` (SF Sunset
Reservoir, known-broken).)

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
