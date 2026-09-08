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
- **Do one thing per turn.** Don't investigate adjacent problems you notice — mention them in one line and stop. Wait for me to say go.
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
  3. Decision (confirmed with Christian): kept this area anyway. **NOTE (2026-09-05): the "real pitch
     variety" justification has since been refuted** — see "Cross-examination findings" below. After the
     DSM date fix, all 13 Altstetten detections read 0.0–1.4°, and the two apparent exceptions (5.7°,
     10.2°) turned out to be tile fragments whose contiguous other halves read 0.21° and 1.06°. The area
     still has plenty of confirmed real solar; it does not have the tilt variety it was chosen for. `ZURICH_ALTSTETTEN = (8.4747, 47.4012)` in `scripts/level3_cv_detection.py` has the full story in a code comment.
  - **Two more bugs found while chasing this down, both fixed:**
    - `fetch_swissimage_chip()` had the *exact same* single-tile bug `fetch_dsm_chip()` had before its earlier fix — only reading `items[0]` instead of mosaicking. Altstetten's bbox happened to straddle a SWISSIMAGE tile boundary (Wiedikon's didn't), silently truncating the fetched image to a fraction of the requested width and crashing the tile grid layout. Fixed the same way: `rasterio.merge.merge()` over all covering tiles, newest-first so overlaps still prefer the most recent acquisition.
    - The wind-rose's label-decluttering (added for Wiedikon) had a latent bug that only showed up with more detected arrays: it unwrapped azimuths past 360° by comparing each point's raw angle to the *previous label's already-pushed* position instead of to the *previous point's raw* position, so a raw angle could look like a false "wraparound" mid-walk and get shifted by multiples of 360° — landing some labels back on top of earlier, unrelated ones. Fixed by unwrapping the raw azimuth sequence once, cleanly, before doing any pushing.

### Open findings — investigated, NOT yet fixed (as of 2026-08-17)

Verified against the live STAC API; recorded here because the code still has the bug.

- **~~We are silently using 2018 elevation data against 2025 imagery.~~ FIXED 2026-09-05** (commit
  `1307286`). `fetch_dsm_chip()` now sorts STAC items by `properties.datetime` descending before
  building the href list, same as the imagery fetch. Verified live: both test areas return 4x 2018 +
  4x 2024 tiles, and the API returns 2018 first, so `rasterio.merge` was keeping 2018 pixels.
  **The effect was large.** Over one 20m box at the Wiedikon test building (8624 px), 2024 vs 2018
  differ by >0.30m (the whole outlier threshold) across **49.3%** of pixels and by >1m across 18.4%.
  Mean difference +0.007m, so no vertical bias -- the change is entirely local. One detection's
  reported tilt moved 8.0deg -> 12.6deg on the fix alone. **Any tilt/azimuth number recorded in this
  file before 2026-09-05 was fitted to 2018 elevation and should not be trusted.**

- **Higher-resolution LiDAR is available and we're not using it.** swissSURFACE3D publishes two products, and we've only ever used the derived raster:

  | Collection | What it is | Density | Format |
  |---|---|---|---|
  | `ch.swisstopo.swisssurface3d-raster` | what the code uses now | 4.0 samples/m² | 0.5 m GeoTIFF |
  | `ch.swisstopo.swisssurface3d` | the native point cloud | **28.7 points/m²** | 2024: COPC (`.copc.laz`), 2018: `.las.zip` |

  ~7× denser, ~0.19 m average point spacing vs 0.5 m. Density read directly off a 2024 tile's LAS 1.4 header (28,667,671 points over a 1000×1000 m tile), not from documentation. The 2024 files are **COPC** (Cloud Optimized Point Cloud) and the server honours HTTP range requests (verified: returns `206`), so a single building's points can be read without downloading the ~1 km tile — the same trick as Cloud-Optimized GeoTIFF. Would also give multiple returns per pulse and classification codes, neither of which survive into the raster.
  - Most relevant to the **known gap** above (small roofs whose several tilted rows all fit within 30 cm of one compromise plane, so `fit_group_planes()` never splits them): at 0.5 m the 97 m² test case only gets ~20 samples across its ~10 m span; at 0.19 m spacing it would get ~50, with enough points *per row* to fit each row separately.

Backup candidates if Zurich hits a wall: Netherlands (PDOK/AHN, 25cm ortho + 6–10 pts/m² LiDAR, fully open) and Massachusetts, USA (MassGIS, 15cm ortho + QL1/QL2 LiDAR with DSM, public domain). Note both are *lower* density than the swissSURFACE3D point cloud (28.7 pts/m²) — there's no resolution argument for switching regions; the densest data we've found is already where we're working.

**`cv` in filenames means computer vision** — the imagery-based detection branch. It was never spelled
out anywhere until 2026-09-08; don't let a new abbreviation into this repo without expanding it once.

Prototype code, two files (split 2026-09-08 by dependency, not by topic — the point is that the half
where the silent-wrongness bugs live can be tested without the model stack):

- **`scripts/roof_geometry.py`** — orientation, footprint regularization, plane fitting, the view vote,
  plausibility gates, the registration NCC. **numpy + scipy + rasterio only, no torch.** Tests are
  `scripts/test_roof_geometry.py`, which runs in seconds:
  `uv run --isolated --with numpy --with scipy --with rasterio python scripts/test_roof_geometry.py`
  (24 checks, all passing; coverage is PARTIAL — `fit_roof_plane()` produces the tilt and azimuth this
  project depends on and still has no unit test.)
- **`scripts/level3_cv_detection.py`** — the pipeline: bboxes, fetching, model loading, tiling,
  `find_pv_arrays()`, `add_roof_geometry()`, all the plotting, `main()`. Imports from `roof_geometry`.

Both explicitly experimental, not part of the main pipeline. Model weights live in `data/raw/deepsolar_weights/` (gitignored, ~1.2GB — re-download via `gdown` from the Google Drive link in the DeepSolar-3M GitHub README if missing). Run with:
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
- **Rebuild the pipeline artifact whenever the CV pipeline's numbers change**, and republish it to
  the same URL. This is part of committing that change, not a separate task:

  ```
  uv run --isolated --with numpy --with pillow python scripts/build_pipeline_artifact.py
  ```

  Then ask Claude to publish `data/pipeline_artifact.html` to
  **https://claude.ai/code/artifact/7c58f59e-4634-4af5-b481-0eeb1baf31fc** (pass that as `url` so the
  link stays the same). The HTML is gitignored — it is a build output; `scripts/build_pipeline_artifact.py`
  is the source of truth.

  **Why this is a rule and not a nicety.** The page states measured quantities — detection counts,
  panel areas, the tilt-vs-area correlation, how much headroom the ground gate has left. A page of
  hand-typed figures goes stale silently and keeps looking authoritative. That already happened:
  five figures in *this file* were stale by the time the page was first built, and one of them
  (ground-gate headroom, recorded as 8.9 m, actually 2.3 m) had stopped supporting the conclusion
  drawn from it. So every figure on the page is read from `data/level3_cv_*_detections.npz` at build
  time and nothing is transcribed — if a number can't be derived from the caches, it doesn't belong
  on the page.

  **The rebuild half is automated.** `.githooks/pre-commit` rebuilds the HTML whenever a detection
  cache is newer than it — i.e. whenever the pipeline has been re-run since the page was last built.
  Enable it once per clone:

  ```
  git config core.hooksPath .githooks
  ```

  It keys off cache mtime rather than staged paths on purpose: editing `level3_cv_detection.py`
  changes nothing on the page until the pipeline actually runs, so a staged-path trigger would fire
  constantly and rebuild an identical file. **The hook never blocks a commit** — it exits 0 on every
  path, including when the rebuild fails. Failing a commit over a documentation page would be worse
  than a stale page.

  **Publishing stays manual, and cannot be otherwise.** Pushing to the artifact URL needs the Artifact
  tool and therefore a Claude session. The hook prints the reminder and the link; say "republish the
  pipeline artifact" and it goes out on the same URL.

  **Versions are cut by measurement, not by commit.** `data/pipeline_versions.json` (committed — it is
  the record, not a build output) accumulates one entry per build in which a *tracked figure moved*.
  Editing a comment in the pipeline is not a new version; re-running it and getting 30 arrays instead
  of 29 is. The 22 tracked figures are declared in `TRACKED` in the build script — detection counts,
  panel areas, ρ(area, tilt), median tilts either side of 120 m², steepest tilt, and lowest kept
  detection, per test area. **To track a new claim the page makes, add it to `TRACKED`**; anything not
  in that list can change silently.

  Each entry records the commits landed since the previous entry rather than a single commit. That is
  deliberate: the build runs from a pre-commit hook, so `HEAD` is the last *completed* commit and the
  one being written does not exist yet — and in practice several commits sit between two pipeline
  runs anyway, so "which commits moved this number" is the honest question. The hook stages the
  history file so the entry ships in the commit whose work produced the numbers.
- Keep scripts runnable standalone: `uv run python scripts/whatever.py`.
- Data files go in `data/`. Raw downloads in `data/raw/` and **gitignored** — they're large and re-downloadable.
- If a script takes more than a few seconds, cache its output to disk so I'm not re-running it constantly.

## Repo layout

```
sf-solar/
├── CLAUDE.md
├── README.md
├── docs/            # architecture + roadmap
├── scripts/         # standalone Python steps (+ roof_geometry.py, imported by the CV pipeline)
├── data/
│   ├── raw/         # gitignored
│   └── latest.json  # published output
├── site/            # static website (GitHub Pages)
└── .github/workflows/   # scheduled update job
```

## Cross-examination findings (2026-09-05)

An adversarial review of the whole CV branch. Every number below was **reproduced independently**
against the cached detections (`data/level3_cv_*_detections.npz`) before being written down.

**Confirmed, and serious:**

- **Fitted tilt is a function of footprint size, not roof physics.** Spearman(area, tilt) over Wiedikon
  = **−0.705, n=14, p=0.0048**; survives dropping the 66.3° outlier (−0.676, p=0.011). Median tilt is
  **12.8° for footprints <120 m²** and **0.5° for ≥120 m²**. Altstetten shows −0.170 (p=0.58) only
  because its smallest footprint is 212 m² — range restriction, not different physics. Mechanism is
  leverage: for a small footprint, one contaminating edge DSM pixel at Δz≈10m swings the fit by
  degrees. `MAX_OUTLIER_FRACTION = 0.2` makes this worse by switching the estimator between trimmed
  and untrimmed as a function of size, so trimmed and untrimmed tilts are pooled in every plot and
  claim despite being different estimators. **Plot tilt vs area on every run.**
  **Still present after the 2026-09-06 stitching fix**, which was never going to touch it: Wiedikon
  ρ = **−0.436, n=24, p=0.033**, median tilt **12.6° under 120 m²** vs **1.0° at or above**, and two
  impossible roofs survive at **66.3°** (43 m²) and **60.9°** (17 m²). Altstetten's ρ = −0.165
  (p=0.57) is still range-restricted, and its whole fleet now reads 0.0–3.6°.
- **~~A "detection" is a tile, not an array.~~ FIXED 2026-09-06** (commit `7f2e475`).
  `find_pv_arrays()` now stitches every flagged tile's mask into one full-image canvas and runs a
  single connected-component pass over the whole thing, returning one record per contiguous blob.
  Records keep the same `(mask, y0, x0, pixels)` contract, reinterpreted as the blob's bounding box,
  so `add_roof_geometry()` / `save_detection_cache()` / `audit_detections.py` needed no changes.
  **Effect, Altstetten:** 13 detections (12 of them seam fragments) → **14 arrays assembled from 24
  tile-level pieces, 7 spanning more than one tile**, and the tilt range collapsed from "0.0–1.4°
  plus apparent exceptions at 5.7° and 10.2°" to **0.0–3.6°**. Those two exceptions were seam
  artifacts, now confirmed by merging — the pipeline was producing them, not the roofs.
  **Effect, Wiedikon:** 14 records → 24 arrays from 26 pieces; the 1462 m² industrial array is one
  detection instead of three. The count went *up* because the old code also lumped several separate
  buildings in one tile into a single record — see the next bullet.
  Each record now also carries `n_tiles` and `possibly_truncated` (blob edge runs into ground no
  tile position segmented — the case stitching cannot fix; **fixed separately by overlapping
  inference, below**).
- **Gap bridging — one roof is one array. FIXED 2026-09-07** (commit `655e49e`). Stitching removed
  the *tile* as an artificial divider but left pixel adjacency as the definition of "one array",
  which is too strict: the segmenter drops out across walkways, shadow and its own uncertainty, and
  on the Wiedikon test roof that split one array into three detections whose nearest mask pixels were
  1.5, 4.3 and 7.2 m apart (Christian spotted it by eye — three panels of the same building presented
  as three buildings). Labelling now runs over a morphologically closed copy of the canvas
  (`bridge_mask_gaps()`, dilate then erode by `GAP_BRIDGE_PX`).
  **The closed canvas decides grouping ONLY** — every returned mask is cut back to the pixels the
  segmenter actually called panel. If the bridged gaps counted as panels, every area and every
  capacity estimate built on one would be inflated by the walkways between rows. **The invariant is
  that total panel area is identical at every radius including zero**, and it holds: Wiedikon
  3021.8 m² before and after, to the decimal. (Altstetten reads 6779.8 vs 6778.9 — that is one
  16 px fragment which used to fall below the 30 px size floor and now gets bridged into a
  neighbour, so a recovery, not a leak. Verified: 1 blob dropped before, 0 after.)
  **The radius came off a measured plateau, not a preference.** Sweeping 0–30 px against the cached
  masks of both areas, the count is stable from **9 px through 16 px in both** (Wiedikon 16 arrays,
  Altstetten 8) and only merges further at 21 px in Altstetten and 30 px in Wiedikon. 12 px (2.8 m)
  sits in the middle of the range where the two independently agree. Same shape of reasoning as the
  2 m ground gate: look for the flat region before arguing about the value.
  **Effect:** Wiedikon 24 → 16 detections (6 assembled across gaps), Altstetten 14 → 7 (4 across
  gaps). The three pieces on the Wiedikon roof are now one 1968 m² array.
- **Overlapping inference — the tile stops being a partition. DONE 2026-09-08** (commit `dc2f798`).
  Stitching and gap bridging both operated on masks that *existed*. The remaining seam damage was
  masks that were never produced: the classifier gates segmentation per tile, so a roof split evenly
  across a seam can score just under 0.2 in **both** tiles and vanish entirely — no fragment, no
  warning. That is a false negative, and CLAUDE.md's own cross-examination notes missed capacity is
  the one-sided bias an aggregate MW total cannot afford. `tile_chip()` now slides the window
  `TILE_STRIDE_PX` (default `TILE_PX // 2`) in **both** axes, so every interior pixel falls inside 4
  positions and sits ≥64 px from the edge of its best one (was 0 px).
  **Cost is ~1.6× wall clock, not the ~3.6× the position count suggests** (Wiedikon 3:09 → 4:54) —
  segmentation runs only on flagged positions, and fetch + model load + DSM are fixed costs.
  **Combining the views.** The boolean OR canvas became three accumulators (`votes` / `views` /
  `conf_sum`) and `combine_views()` takes a **majority vote among the views that actually segmented
  each pixel**. An unflagged tile does not vote — it is the classifier's judgement about a whole 60 m
  square, and counting it as "not panel" for each of its 65k pixels would over-read a tile-level
  score and make the result depend on the 0.2 threshold twice. Ties go to panel. Union was rejected
  (can only ever *grow* area, and this classifier's false positives are real); averaged probability
  would need a new threshold with no ground truth anywhere in Zurich to set it against.
  **The degenerate case is the regression test.** At stride == `TILE_PX` every segmented pixel has
  `views == 1`, so `combine_views()` reduces exactly to `votes >= 1` — the OR canvas it replaces.
  Unit-tested, and confirmed end to end: `TILE_STRIDE_PX=256` reproduces the previous run to the
  decimal on both areas. **The old behaviour is an env var away, not a git revert.**
  **Effect — the seam fix is decisive.** Area flagged possibly-cut-short: Altstetten **3614 → 61 m²**,
  Wiedikon 169 → 63 m². One Wiedikon roof grew **+85%** (85 → 157 m²). Arrays that already existed
  grew only **+2.7% / +2.4%** and held their tilts, so nothing established moved. Altstetten's
  2198 m² blob correctly **split** into 1184 + 1026 — the pieces are 8.5 m apart, 3× the gap-bridge
  radius, joined at stride 256 only by a thin connector the overlapping views outvoted.
  **Repeatability, measured for the first time.** `report_view_agreement()` runs every pass. Four
  views per pixel turn every mask from a single unrepeatable measurement into one with an error bar:
  **84.3% of contested Wiedikon pixels and 93.6% of Altstetten's are unanimous**, 2.4–6.2% get
  outvoted. Union vs majority is worth only **170–208 m² (2–6%)**, so the estimator choice mattered
  far less than expected. **Read this as repeatability, NOT correctness** — five views agreeing on a
  skylight is five views consistently wrong, and the false-positive rate remains unmeasured.
  **The 20% area rise is the honest caveat.** Wiedikon total went 3022 → 3627 m², and **86% of that
  (522 m²) is entirely new detections**, not recovered ones — 13 of them, median 28 m², median tilt
  29.8°, in an area whose confirmed arrays are 0–13°. Their tilts prove nothing either way (the
  documented leverage bias makes *any* small footprint's tilt unreliable), and the size-tilt
  correlation strengthening to ρ = −0.453 (p=0.014) is mechanically expected from adding small
  detections, not independent evidence. 5 of 13 were seen by only one flagged view, so voting could
  not contest them; the ground/tilt gates caught 4. **This is best read as denser sampling of a
  detector whose false-positive rate was always unmeasured, not as a new defect introduced here** —
  under-sampling to keep it hidden would be the wrong fix. Measuring that rate against
  `ch.bfe.elektrizitaetsproduktionsanlagen` is the outstanding job.
  Also fixed here: `tile_chip()`'s exclusive `range()` stop, which dropped the last full row and
  column at any dimension that is an exact multiple of `TILE_PX` (a 4×4 grid became 3×3). No effect
  at the Zurich chip size, which is what makes the stride-256 regression test valid.
- **~~`pv_area_m2` and `tilt_deg` describe different objects.~~ FIXED by the same change.** Area
  used to sum the whole tile mask across every building in it while tilt came from `planes[0]`, the
  largest connected facet of that pile. Each record is now one contiguous blob, so both describe the
  same object.
- **Segmentation badly under-covers large true positives.** On the 1322 m² and 584 m² arrays the masks
  trace narrow strips across roofs that are panel-covered edge to edge.

**Confirmed code defects (all still present):**

- `fit_roof_plane()` returns coefficients inconsistent with its own mask when the loop exits by
  exhausting `MAX_OUTLIER_ITERATIONS`: the last pass assigns `valid = tightened` and the loop ends with
  no refit, so the returned `(a,b,c)` was fit to the previous, larger point set. `fit_group_planes()`
  then peels leftovers against that stale pairing.
- ~~`tile_chip()` uses `range(0, h - TILE_PX, TILE_PX)` — exclusive, so the last full tile row and
  column are never generated.~~ **FIXED 2026-09-08** (commit `dc2f798`), alongside the stride change.
- ~~`fetch_swissimage_chip()` uses `Resampling.bilinear` for a **2.36× decimation** with no
  anti-alias prefilter~~ — **TESTED 2026-09-06, NOT THE CAUSE. Do not re-try this.** The mechanism
  was plausible (panel rows, skylight glazing grids and painted parking-bay lines are all periodic
  structure that folds under undersampling), but across the same 100-tile Wiedikon grid
  `Resampling.average` moved classifier scores by a mean of **+0.0055** (max 0.085) and `lanczos` by
  **−0.0055** (max 0.087). No tile crossed the 0.2 segmentation threshold under `average`; one did
  under `lanczos`. On nine objects labelled independently of the model — six arrays visible
  edge-to-edge in the imagery, the dormer skylights Christian confirmed at native 10 cm, and two
  detections swissALTI3D puts at 0.01 m and −0.03 m above bare ground — the separation between real
  and not-PV was **+0.006 (bilinear), −0.027 (average), −0.011 (lanczos)**: all three
  indistinguishable from zero, and under `average` the false positives rose on all three. Detection
  count, total area (3022/3027/3001 m²) and the two impossible tilts were unchanged. Default stays
  `bilinear`; `SWISSIMAGE_RESAMPLING` env var kept so the test is one command to repeat elsewhere.
  **The real finding is that the classifier does not separate PV from skylights or parking decks at
  all, under any filter** — this is a model/discrimination problem, not a preprocessing one.
- ~~No plausibility gate anywhere: a 66.3° "roof" was computed, cached, plotted and reported without
  a warning~~ — **FIXED 2026-09-06.** `apply_plausibility_gates()` adds two: **height above bare
  terrain < 2 m** (swissSURFACE3D DSM minus swissALTI3D DTM, `fetch_dtm_chip()`) and **tilt > 60°**.
  Wiedikon: 3 of 24 flagged, 3.2% of area, **no real array touched** — the two at-grade detections
  measured +0.02 m and −0.03 m against 10.9–37.8 m for every confirmed array, and the 43 m² object
  trips both gates independently. Altstetten: 0 of 14 flagged.
  **Nothing is deleted, deliberately** — ground-mounted PV is genuinely at grade (this project's own
  positive control, SF's Sunset Reservoir farm, is ground-mounted) and facade PV is genuinely
  near-vertical; a silent drop would delete real capacity, the one direction an aggregate MW total
  cannot afford to be wrong in. The wind-rose plots the plausible set only (one 66° point sets the
  radial scale for every other marker); the console and cache keep everything.
  **Caveat:** the gate's headroom is a property of the neighbourhood. Altstetten's lowest real
  detection is 2.7 m (industrial sheds), a 0.7 m margin vs Wiedikon's 8.9 m. In a low-rise area
  (carports, single-storey garages) 2 m would start clipping real arrays — check the height
  distribution before trusting this in a new city.
  **Does not touch the skylights**: at 14.6 m up and 12.6° tilt they are geometrically unremarkable.
  Both gates are physical-impossibility checks, and the skylights are not impossible, just not PV.
  Note swissALTI3D publishes BOTH a 0.5 m and a 2 m GeoTIFF per item — matching on `.tif` alone
  mosaics two resolutions together, hence the explicit `_0.5_` filter in `fetch_dtm_chip()`.

**The false-positive audit, withdrawn as a rate.** The native-10cm audit found the highest-confidence
Wiedikon detection (p=0.952) is four dormer skylights while the two largest unambiguous real arrays
score 0.348 and 0.260 — that specific observation stands and was independently confirmed by Christian.
But the *rate* does not: labels were assigned with the model's own contour drawn over the imagery, by
the person who built the pipeline (anchoring), and with 4 clear positives vs 4 clear negatives the
minimum attainable one-sided p is 1/70, so the design cannot reject chance ranking even in principle.
Depending on denominator the FP rate spans [0.14, 0.91]. **Treat it as unmeasured.** Nobody has looked
at false *negatives* at all, and for an aggregate MW total missed capacity is the one-sided bias that
actually matters.

**Useful data source found:** `ch.bfe.elektrizitaetsproduktionsanlagen` (Swiss federal electricity
production plants) is queryable via `api3.geo.admin.ch/rest/services/all/MapServer/identify` and returns
PV plants with capacity and commissioning date — 17 within a 700m box over Wiedikon, 6 over Altstetten.
Use it as a **denominator for recall**, not as a per-case confirmation oracle: coverage is incomplete
and points are geocoded to street addresses, not array centroids. Also `ch.bfe.solarenergie-eignung-daecher`
(sonnendach) publishes per-roof-facet tilt/azimuth nationally — a way to validate `fit_group_planes()`
at n≈500 with no CV involved, though it measures the *roof* not the *panels*, so racks should disagree.

### The strategic question nobody had asked

`scripts/level3_fleet.py` already reads per-system `Tilt` and `Azimuth` straight from CA DG Stats, with
the azimuth convention verified against the official data key. Running it:

```
Systems needing tilt/azimuth gap-fill ('multiple'): 376 (4.1% of capacity)
```

**95.9% of San Francisco's installed capacity already has measured tilt and azimuth.** This branch
exists to derive, from imagery SF does not have, a quantity SF already knows. It has earned its keep —
it surfaced the DSM date bug and a lot about the data — but it is **not on the critical path** to a
citywide MW curve.

**The sensitivity table has now been run** — `scripts/level3_geometry_sensitivity.py`, which sweeps
tilt and azimuth error over the real fleet through pvlib and sums to a citywide MW curve. All four
sanity checks pass (peak clear-sky GHI 967 W/m², peak AC 77% of nameplate DC, exactly zero at night,
and the vectorised transposition matches a per-geometry loop to 2.3e-13 MW).

**Answer: the branch cannot pay for itself.** Even in an absurd worst case where the entire 4.1%
gap-filled slice is simultaneously off by 20° in tilt *and* flipped 180° in azimuth, the citywide
error is **−0.73% at summer noon and −3.35% on a winter morning**. That is the ceiling on what
perfect geometry measurement could buy, and the realistic figure is far smaller.

Other findings worth keeping:

- **Systematic bias matters enormously more than random scatter.** A +20° tilt *bias* costs +48.6% at
  winter 09:00; +20° of independent per-system *scatter* costs +13.2%. Azimuth: ±60° bias reaches ±24%
  at winter 09:00, while σ=60° scatter reaches only −6.4%. Errors that cancel across 13,485 systems
  are cheap; errors that point the same way are not. **The CV pipeline's measured error is the
  expensive kind** — a size-dependent tilt bias (ρ = −0.705), not noise.
- **Annual energy is a bad proxy for a nowcast.** ±20° of tilt bias moves annual clear-sky energy by
  only −3.2%/+8.3%, but moves instantaneous winter-morning MW by −14.8%/+48.6%. Summer *noon* is
  almost completely insensitive to tilt (0.2% at ±20°) — the hours that matter are the shoulders.
- **Azimuth error flips sign between morning and evening** (a 60° eastward bias is +12.0% at 09:00 and
  −9.6% at 17:00 in summer), so it partly cancels in a daily total while badly distorting the shape.
- **42.7% of SF capacity (34.6 MW) is recorded flat, and 30.0 MW of that stores azimuth=0 as a
  placeholder, not a bearing.** Any error model that tilts those systems must not let them inherit
  due north, or it will overstate the sensitivity badly. `apply_error()` assigns the tilted fleet's
  capacity-weighted median azimuth (180°, due south) instead; `FLAT_STAYS_FLAT` toggles the
  alternative.

Full three-way pipeline comparison (current / +point cloud / +normal segmentation) with all of today's
measurements: https://claude.ai/code/artifact/c259959a-0425-451c-a7be-3dde0e962d5c
