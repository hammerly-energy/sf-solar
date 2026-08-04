# San Francisco Solar Nowcasting — Architecture

**Goal:** estimate, in near-real-time, the aggregate PV generation of all solar installations in the City and County of San Francisco. Stage 1 delivers a *current-state estimate*; Stage 2 extends it to a *0–6h forecast*.

**Context:** personal learning project, intended to be portfolio-visible. That argues for: a working end-to-end pipeline early, honest validation, a nice map, and depth in one place rather than uniform polish. The place to go deep is the marine layer — see §6.

---

## 0. Design principles

1. **End-to-end before deep.** Get a crude number for the whole city before improving any one stage. A working ugly pipeline beats three excellent disconnected modules.
2. **Baseline at every stage.** Every model must beat a dumb alternative (persistence, clear-sky climatology, or a commercial API). If it doesn't, you've learned that cheaply.
3. **Static vs. dynamic separation.** Fleet geometry and terrain shading are computed once and cached forever. Only irradiance and temperature are on the hot path. This is what makes real-time feasible on a laptop.
4. **Archetypes, not systems.** Never simulate 8,000 systems individually. See §3.3.

---

## 1. Pipeline overview

```
STATIC (rebuild quarterly)
  Stage 0  Fleet inventory ........ interconnection data + geocoding
  Stage 1  Site geometry .......... LiDAR → tilt/azimuth gap-fill, horizon profiles
  Stage 2  Archetype clustering ... ~8,000 systems → ~300 archetypes

DYNAMIC (every 5 min)
  Stage 3  Satellite ingest ....... GOES-18 ABI Band 2 → cloud index field
  Stage 4  Irradiance ............. clear-sky × cloud index → GHI → DNI/DHI
  Stage 5  PV model ............... POA transposition → DC → inverter → AC
  Stage 6  Aggregation ............ archetype output × capacity weights → MW

DYNAMIC v2 (every 5 min, extends Stage 3)
  Stage 3b Advection .............. optical flow → semi-Lagrangian forward projection
  Stage 3c NWP blend .............. crossover to HRRR at longer horizons

CONTINUOUS
  Validation ...................... PVOutput scrape → skill scores vs. baselines
```

---

## 2. Stage 0 — Fleet inventory

**Primary source:** California DG Stats, Rule 21 / NEM interconnection datasets.
`https://www.californiadgstats.ca.gov/downloads/` — verified live, current as of Apr 30, 2026.

Two datasets, and the distinction matters:

| Dataset | Grain | Includes | Use |
|---|---|---|---|
| **Interconnected Project Sites** | one row per address/site | excludes pending *and* decommissioned | **This is your fleet.** |
| **Interconnected Applications** (149 MB) | one row per application | excludes pending, *includes* decommissioned | Use for install history / capacity additions |

Applications get merged into project rows via `Superseding ID` / `Preceding ID` when a system is expanded. Use Project Sites unless you specifically need the history — it has already done the dedup for you, and it has already dropped decommissioned systems, which kills one of the risks in §11.

The page has a **filter widget** (Projects vs. Applications × PGE/SCE/SDGE/All × time window). Grab PG&E only; that's a much smaller download than the statewide file.

Critically, these contain **Tilt** and **Azimuth** columns where the installer reported them — so steps 1 and 2 of the original plan are largely a download, not a research project.

Known data quirks, per the CPUC read-me:

- Multi-array systems with differing geometry list `"Multiple"` instead of a number. These need gap-filling.
- `Tracking Type` flags single/dual-axis and `"Mixed"`. Rooftop trackers are rare in SF; treat any as suspect data.
- Address granularity varies. Some records give street address, some only ZIP + city.
- **2024 PG&E generator/inverter columns were erroneously duplicated**; fixed as of the Dec 2024 publication. If you use an archived file, check for this.
- Publication lag is ~6 weeks after month end. Your fleet is always slightly stale — fine for a fleet that grows a few percent a year.

**Secondary sources:**

| Source | Use |
|---|---|
| LBNL *Tracking the Sun* | Sanity-check tilt/azimuth distributions; module and inverter model priors |
| DeepSolar (Stanford) | CV panel detection — catches unpermitted / pre-NEM systems |
| DataSF building footprints + address points | Geocoding, parcel matching |
| CEC / PG&E county capacity totals | Aggregate sanity check on your fleet total |

**Output schema** (`fleet.parquet`, one row per system):

```
system_id, lat, lon, elevation_m, dc_kw, ac_kw, tilt_deg, azimuth_deg,
install_year, module_tech, tracking, geocode_precision, source,
tilt_source, azimuth_source, horizon_id, archetype_id
```

Carry `*_source` and `geocode_precision` fields explicitly. Your spatial resolution is only as good as your worst geocode, and you will want to stratify error by it later.

**Gap-filling missing tilt/azimuth:**

1. Match system to a building footprint → fit roof planes from LiDAR (§3.1) → take the largest south-facing-ish plane.
2. If no LiDAR match, sample from a prior conditioned on building type. SF's rowhouse stock skews flat-roof and east–west; the "everything faces south at 30°" assumption is wrong here and will bias your peak.

**Privacy note:** don't publish site-level addresses in a public portfolio piece. Aggregate to census tract or neighborhood for display.

---

## 3. Stage 1–2 — Site geometry and archetypes

### 3.1 Roof planes from LiDAR

Sources: USGS 3DEP (statewide QL1/QL2 coverage), plus higher-density city LiDAR if available for SF.

Method: rasterize to a ~0.5 m DSM → segment roof surfaces within building footprints (RANSAC plane fitting or region-growing on the normal field) → derive tilt and azimuth per plane, keep planes above a minimum area.

### 3.2 Horizon profiles — do not skip this

**In San Francisco, shading is first-order, not a refinement.** Dense rowhouse blocks, Sutro Tower, Twin Peaks, Bernal Heights, and 280 m of relief across the city mean many arrays lose real beam hours.

Method: from the DSM (plus a DTM for far terrain), ray-cast 360° around each site in 5° azimuth steps and record the maximum horizon elevation angle. Store as a 72-element array. At runtime, if solar elevation < horizon[azimuth_bin], the beam component is zero (soften the edge slightly to avoid a hard step artifact).

Also compute a **sky view factor** per site from the same profile — used to derate the isotropic diffuse component. Under fog, when everything is diffuse, SVF is the dominant geometric term and array azimuth barely matters at all. That's a useful property: your orientation modeling matters least exactly when your cloud modeling matters most.

Cost: expensive (hours), but computed exactly once.

### 3.3 Archetype clustering

Bin the fleet on `(tilt, azimuth, horizon signature, 1 km spatial cell)`. Target ~200–500 archetypes. Simulate one 1 kW reference system per archetype per timestep, then scale by summed capacity within the bin.

This is a ~20–40× compute reduction with negligible accuracy loss, because the within-bin variance is far smaller than the cloud-field variance you can't resolve anyway. It's also what makes a 5-minute update cadence trivially achievable.

---

## 4. Stage 3 — Satellite ingest

### 4.1 Use GOES-18

**GOES-18 is the operational GOES-West satellite**, on station at 137.2°W since January 10, 2023. It is the only correct choice for San Francisco.

Note the current constellation state, since a lot of tutorials are stale:

- **GOES-18 → West (137.2°W).** Yours.
- **GOES-19 → East (75.2°W)**, operational since April 4, 2025.
- **GOES-16 → decommissioned from the East slot** when GOES-19 took over; data distribution turned off, drifted to storage at 104.7°W. A reprocessed GOES-16 L1b archive (2018–2024) exists with improved calibration, but it's East-view and irrelevant here.
- **GOES-17 → offline**, on-orbit storage at ~105°W.

The original plan said GOES-16 — that satellite is off the air, and even when live it viewed SF at a punishing slant angle. This matters enormously for parallax (§4.4).

**Products:**

| Product | Res | Cadence | Use |
|---|---|---|---|
| ABI L1b Radiances, **Band 2** (0.64 µm) | 0.5 km | 5 min (CONUS) | Primary — visible reflectance for cloud index |
| ABI L2 **ACHA** (cloud top height) | 2 km | 10 min | Parallax correction input |
| ABI L2 **ACM** (clear sky mask) | 2 km | 10 min | QC and background estimation |
| Band 13 (10.3 µm IR) | 2 km | 5 min | Night, and low-stratus discrimination |

Mesoscale sectors give 1-min imagery but are steered to active weather — you cannot rely on Bay Area coverage. Design for 5-min, opportunistically use 1-min.

**Access:** the `noaa-goes18` S3 bucket on AWS Open Data (`us-east-1`). Free, no credentials:

```bash
aws s3 ls --no-sign-request s3://noaa-goes18/ABI-L1b-RadC/
```

Subset to a Bay Area bounding box on read via `s3fs` + `xarray` — a Band 2 Bay subset is a few hundred KB per frame. There's also an **SNS topic** (`arn:aws:sns:us-east-1:123901341784:NewGOES18Object`) that fires on every new object, if you'd rather event-drive the pipeline than poll.

Don't write the downloader yourself: **`goes2go`** (Brian Blaylock) handles bucket paths, product naming, and time-range downloads, and returns xarray objects directly.

**Prior art worth reading before you build:** *SunCast: Solar Irradiance Nowcasting from Geosynchronous Satellite Data* ([arXiv:2201.06173](https://arxiv.org/pdf/2201.06173.pdf)) — a Berkeley group doing essentially your Stage 3–4 with ML. Also NOAA's own [Beginner's Guide to GOES-R Series Data](https://noaa-goes16.s3.amazonaws.com/Beginners_Guide_to_GOES-R_Series_Data.pdf), which is the clearest explanation of the fixed-grid projection you'll find.

### 4.2 Reprojection

GOES data arrives on the ABI fixed-grid (a geostationary projection). Reproject to a local UTM Zone 10N grid at 500 m using `pyproj` / `rioxarray`. Do this once per frame, cache to zarr.

### 4.3 Cloud index (Heliosat method)

1. **Ground albedo background.** For each pixel and each solar-time slot, take a low percentile (≈5th) of observed reflectance over a trailing 30-day window. This estimates the clear-ground reflectance ρ_ground.

   *SF-specific hazards:* (a) land and water pixels have wildly different albedo — mask the bay and ocean separately; (b) in the Outer Sunset, persistent summer stratus may mean the "clearest" observation in a 30-day window still has cloud in it, biasing ρ_ground high and suppressing your cloud index exactly where you need it most. Consider a longer window with seasonal detrending for coastal pixels, or seeding the background from a known-clear winter period.

2. **Cloud index:** `n = (ρ − ρ_ground) / (ρ_cloud − ρ_ground)`, with ρ_cloud an upper-percentile saturation value.

3. **Clear-sky index:** `k = f(n)`. Start with the Heliosat-2 piecewise mapping, then refit the coefficients against your own ground truth. The published coefficients were tuned for Europe; refitting for coastal California stratus is a legitimate, small, publishable-flavored contribution.

### 4.4 Parallax correction — mandatory, and there are two of them

This is the step most hobby projects skip and it will smear your entire map if you do.

**Displacement 1 — viewing parallax.** A cloud at height *h* is reported at the pixel where the satellite's line of sight intersects the *ground ellipsoid*, not where the cloud actually is. Correct by walking back along the satellite view vector by *h*.

**Displacement 2 — shadow projection.** The cloud's shadow falls where the *solar* vector from the cloud hits the ground. Displacement ≈ `h / tan(solar_elevation)`, in the anti-solar azimuth direction.

Magnitudes for SF:
- Marine-layer stratus at ~400 m: viewing shift ≈ 400 m; shadow shift at 30° elevation ≈ 700 m. Both comparable to your 500 m grid — correctable but not ignorable.
- Cirrus at 8 km: shifts of 8–14 km. Ignoring this puts your cloud in the wrong neighborhood entirely.

Both corrections need cloud-top height. Use ACHA where available; fall back to a Band 13 brightness-temperature-to-height lookup against a temperature profile.

### 4.5 GHI → DNI/DHI separation

`GHI = k × GHI_clear`, then split with a separation model — `pvlib.irradiance.dirint` or Engerer2 (Engerer2 was fitted on Australian data and generally performs well at sub-hourly resolution). Under thick fog the split goes ~100% diffuse, which the models handle fine; the risky regime is broken cloud, where the split is genuinely bimodal and no deterministic model does well.

---

## 5. Stages 4–6 — Irradiance to power

All of this is `pvlib-python`. The chain, per archetype:

1. **Solar position** — `pvlib.solarposition.get_solarposition` (NREL SPA).
2. **Clear-sky GHI/DNI/DHI** — `Location.get_clearsky(model='ineichen')` with Linke turbidity climatology. Compute per 1 km cell; SF's elevation range is large enough to matter slightly.
3. **Horizon mask** — zero the beam component where solar elevation < horizon (§3.2).
4. **Transposition** — Perez model → plane-of-array irradiance. Apply sky view factor to the isotropic diffuse term; ground albedo ≈ 0.15–0.2 for urban.
5. **IAM / reflection losses** — ASHRAE or physical model. A few percent, cheap to include.
6. **Cell temperature** — Faiman or SAPM. Needs ambient temperature and wind speed; source from HRRR analysis or nearby METARs (KSFO, KOAK, KHWD) plus city mesonet stations. Temperature coefficient ≈ −0.35 %/°C.
7. **DC power** — `pvwatts_dc`, or the CEC single-diode model if module types are known.
8. **System losses** — soiling (SF's dry summer means real accumulation; ~2–3 % is defensible, and rain events reset it), mismatch, wiring, availability.
9. **Degradation** — 0.5 %/yr from `install_year`. Across a fleet spanning ~2005–2026 this is not negligible in aggregate.
10. **Inverter** — `pvlib.inverter.pvwatts` with clipping at `ac_kw`. **Do not skip clipping.** DC:AC ratios of 1.2–1.3 are standard, and clipping bites exactly at the peak hours you most care about.

**Aggregation:** sum `archetype_unit_output_kW × archetype_capacity_kW` over all archetypes. Emit citywide MW plus a per-neighborhood breakdown for the map.

---

## 6. The marine layer — where the interesting work is

Everything above is standard practice, competently assembled. What makes *this* project non-generic is that San Francisco is one of the hardest small domains in the country for solar nowcasting:

- **Sharp spatial gradients.** The Outer Sunset can be socked in while the Mission is in full sun, ~5 km apart. Your 500 m grid is barely adequate; a coarse irradiance API would completely miss this.
- **Dissipation, not advection.** Stratus burns off *in place*, from the inland edge westward, on a 30–90 minute timescale. Optical-flow advection assumes cloud features translate — for burn-off, that assumption is simply false, and pure advection will be systematically wrong every summer morning. This is the single most interesting failure mode in the project.
- **Low cloud top.** ~300–600 m means parallax is small but not zero, and cloud-top height retrieval is least reliable for low cloud.
- **Strong diurnal and seasonal structure.** Burn-off timing is predictable-ish from inversion strength, onshore pressure gradient, and season — which means a **statistical burn-off model** conditioned on those variables could beat advection outright in the summer months.

Suggested depth investment: build the advection nowcast, demonstrate it fails on burn-off mornings, then build a separate burn-off model and show the improvement. That is a much better portfolio story than a model that just works.

---

## 7. Stage 3b — Advection (v2)

1. Optical flow (Farnebäck, or TV-L1 for smoother fields) between consecutive cloud-index frames.
2. Smooth/regularize the motion field — raw per-pixel flow is noisy over near-uniform stratus.
3. Semi-Lagrangian forward advection at 15-min steps out to ~4 h.
4. Blend toward NWP (HRRR) with a weight ramping over roughly 2–4 h. Satellite advection wins at short horizons; NWP wins beyond. Find your actual crossover empirically rather than assuming it.
5. Apply the burn-off model (§6) as a separate term during stratus regimes.

Expect skill to decay fast. Beating persistence at 15 min is easy; beating it at 3 h is the real test.

---

## 8. Validation

**Plan this before building anything.** A beautiful model you can't score is worthless.

**Ground truth, in order of usefulness:**

| Source | Notes |
|---|---|
| **CSI 15-Minute Interval Data** | See below — the best historical ground truth available, and free |
| **PVOutput.org** | Hobbyists publish live 5-min output with location and system size. Free API. Several Bay Area systems. Your workhorse for *live* validation. |
| Enphase / SolarEdge / Tesla APIs | If you or friends have a system — highest quality, lowest N |
| CSI Measured Production Data Set | Monthly totals for PBI systems. Too coarse for nowcasting, but a good annual-energy bias check |
| SFPUC city-owned installations | Sunset Reservoir (~5 MW) and others; check what's published |
| CAISO behind-the-meter solar estimate | Statewide only — use for aggregate shape sanity, not scoring |

### 8.1 The CSI 15-minute dataset — check this on day one

Buried on the DG Stats downloads page and easy to miss. The CSI program had Itron install **production meters on 504 systems in 2010**, collecting **15-minute generation data through 2016**. Two versions are published:

- **Raw Interval Dataset** — the actual measured data. This is what you want.
- **Simulated 15-minute Interval Data** — CPR/CEC filled gaps and bad data with simulation, based on 414 of the 504 systems. Downloadable **by ZIP code** or by CSI system.

Why this matters: it's a multi-year, 15-minute, metered, geolocated PV production archive for PG&E territory, free, already assembled. If any meaningful number of those 504 systems sit in SF ZIP codes, you can **backtest your entire pipeline against six years of real fog mornings** before you write a single line of live-ingest code. That completely changes the development order — you could validate the physics and the cloud-index mapping offline, on your own schedule, without waiting for weather.

Caveats: ends in 2016, so no overlap with GOES-18 (launched 2022) — you'd need GOES-15 or reanalysis irradiance for that era, which weakens it for validating the *satellite* stage specifically. But it's excellent for validating Stages 1, 2, and 5 (geometry, archetypes, PV physics), which is exactly where silent bugs hide. **Use the raw dataset, not the simulated one, for validation** — validating your model against someone else's model is circular.

**Action:** download the by-ZIP-code archive, filter to SF ZIPs (94102–94134), and count systems. That number determines whether this is a footnote or the backbone of your validation plan.

**Metrics:** MAE and RMSE normalized by installed capacity, plus **skill score** against each baseline. Report skill, not raw error — raw error looks impressive at night.

**Baselines to beat, in order:**

1. Persistence (last observed output)
2. Clear-sky × climatological mean clear-sky index
3. A commercial irradiance API (Solcast has a free hobbyist tier with 5-min nowcast)

**Stratify results by** fog vs. clear regime, hour of day, season, neighborhood, and geocode precision. Aggregate citywide error will look great through spatial error cancellation. The per-site, foggy-morning number is the honest one — report both, and say which is which.

---

## 9. Stack

- **Python.** `pvlib`, `xarray` + `zarr`, `rioxarray` / `pyproj`, `s3fs`, `geopandas`, `numpy` / `scipy`, `opencv` (optical flow), `pdal` or `laspy` (LiDAR).
- **Storage.** Zarr for the satellite time series, Parquet for fleet and results, DuckDB for ad-hoc querying.
- **Orchestration.** A cron'd script. Do not reach for Airflow/Prefect/Dagster until it hurts.
- **Serving.** FastAPI + deck.gl or MapLibre. Precompute per-neighborhood MW; don't serve site-level.
- **Compute.** All of this fits on a laptop for the SF domain. A Band 2 Bay-Area subset is small; the only heavy step is the one-time LiDAR horizon computation.

---

## 10. Milestones

| # | Deliverable | Est. |
|---|---|---|
| M0a | Download CSI 15-min raw data, count SF systems, plot one clear day | an evening |
| M0 | One real system, one clear day, pvlib output vs. measured within ~10% | a weekend |
| M1 | Fleet inventory + archetypes + citywide clear-sky curve | 1–2 weeks |
| M2 | GOES ingest + cloud index, no parallax. Validate vs. baselines. | 2–3 weeks |
| M3 | Parallax + separation model + refit k(n) coefficients | 2 weeks |
| M4 | Live 5-min pipeline + public map | 2 weeks |
| M5 | Advection forecast | open-ended |
| M6 | Burn-off model + the "advection fails on fog" writeup | the interesting part |

M0 is the important one. If pvlib on one known system doesn't match reality on a cloudless day, nothing downstream can work, and you'll have found out in two days instead of two months.

---

## 11. Risks and open questions

| Risk | Mitigation |
|---|---|
| **Geocode precision.** ZIP-only records make fine spatial resolution illusory. | Track precision per record; aggregate to neighborhood for anything published; stratify validation by it. |
| **PVOutput sample bias.** Enthusiast-owned, well-maintained, possibly better-sited than average. | Acknowledge explicitly. Cross-check against aggregate CAISO shape. |
| **Clear-sky background over persistent-fog pixels.** Biases cloud index low exactly where fog matters. | Longer background window for coastal pixels; seed from known-clear winter dates. |
| **Invisible behind-meter behavior.** Curtailment, batteries, outages, snow-free but soiled panels. | Irreducible. Quantify as a residual bias term rather than pretending it away. |
| **Fleet completeness.** Interconnection data misses unpermitted and very old systems. | Bound it: compare fleet total to CEC/PG&E county capacity. Treat the gap as a scaling factor. |
| **Duplicate/retired systems.** Records may include decommissioned installs. | Filter on status fields; check for capacity inflation vs. official totals. |

**Open questions to resolve early:**

- Are you modeling **gross generation** or **net export**? The latter requires a load model and is a substantially bigger project. Recommend gross for v1.
- Instantaneous kW or interval energy (kWh per 5 min)? Affects how you compare against PVOutput, which reports both.
- How much of the fleet actually has usable tilt/azimuth in the interconnection data? This single number determines whether the LiDAR work is a nice-to-have or the critical path — **worth checking on day one.**
- How big is the SF fleet actually? Likely on the order of a few thousand systems and tens of MW — SF is dense, renter-heavy, and roof-poor compared to the suburbs. If it's genuinely that small, the archetype clustering in §3.3 may be unnecessary optimization, and you could simulate every system directly. Measure before optimizing.

---

## 12. Verified as of Aug 2026

Checked live while writing this:

- ✅ DG Stats downloads page live; Project Sites and Applications datasets current as of Apr 30, 2026; tilt/azimuth columns confirmed present in the CPUC read-me.
- ✅ GOES-18 confirmed operational GOES-West since Jan 10, 2023; `s3://noaa-goes18/` public, no-sign-request.
- ✅ GOES-16 confirmed off-air for East since Apr 4, 2025 (superseded by GOES-19) — the original plan's satellite choice needed updating.
- ✅ CSI 15-minute interval datasets (raw and simulated) confirmed available for download.
- ⚠️ SF installed capacity and system count **not verified** — no current county-level figure found in public search. Get this from the data itself.
- ⚠️ PVOutput Bay Area system count **not verified** — check the API directly.
