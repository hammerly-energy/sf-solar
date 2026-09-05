"""
Level 3 -- how much does getting tilt/azimuth wrong actually cost?

The building-level CV detour exists to measure per-system tilt and azimuth
from imagery. Before spending more on it, this answers the question nobody
had asked: if every system's geometry were wrong by X degrees, how far off
would the citywide MW number be?

Two error models, because they behave completely differently in an aggregate:

  BIAS    every system wrong in the same direction (a systematic pipeline
          error -- e.g. small footprints reading too steep). Does NOT cancel.
  SCATTER each system wrong independently (measurement noise). Largely
          cancels across 13,485 systems.

Reported both as annual clear-sky energy error and as instantaneous MW error
at specific times of day, because those answer different questions. Annual
energy is forgiving of tilt; an instantaneous nowcast is not forgiving of
azimuth.

CONVENTIONS (see CLAUDE.md -- these are the bug-prone ones)
  Time      all arithmetic in UTC; America/Los_Angeles only for labels.
  Azimuth   pvlib surface_azimuth: 0=N, 90=E, 180=S, 270=W. CA DG Stats uses
            the same convention (verified against the data key), so no
            conversion. But flat systems record azimuth=0 as a PLACEHOLDER,
            not a real bearing -- 30.0 of the fleet's 81.1 MW sits on it.
  Units     capacity_kw is power; poa_wm2 is irradiance; mw is power.

Run: uv run python scripts/level3_geometry_sensitivity.py
"""

import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

from level3_fleet import load_sf_fleet, resolve_tilt_azimuth, circular_median_deg

# Level 1's reference location -- SF is small enough that solar position and
# clear-sky irradiance barely vary across it (see level3_fleet.py docstring).
SF_LAT, SF_LON, SF_ALT = 37.7749, -122.4194, 16.0
SF_TZ = "America/Los_Angeles"

# DC -> AC derate. Only scales the absolute MW; every percentage error in this
# script is a ratio, so the derate cancels out of the sensitivity result.
DERATE = 0.80

# A system recorded as flat has no meaningful azimuth (the dataset stores 0).
# If an error model tilts it, it has to face *something*, and 0 (due north)
# would be the worst orientation in SF -- using the placeholder literally
# would massively overstate the sensitivity. Instead, a flat system that gets
# tilted is assumed to tilt the way SF actually tilts things: the capacity-
# weighted circular median azimuth of the genuinely-tilted fleet.
FLAT_TILT_THRESHOLD_DEG = 0.5

# The alternative assumption: instead of giving a tilted-by-error flat system
# a plausible south-facing bearing, exclude flat systems from tilt error
# entirely. That isolates the question to the 57.3% of capacity where tilt is
# genuinely measured. main() reports both for one case so the choice is not
# invisible -- see "assumption check" in the output.
FLAT_STAYS_FLAT = False


def build_unique_geometries(fleet):
    """Collapse the fleet to unique (tilt, azimuth) pairs with summed capacity.

    POA irradiance depends only on the orientation, not on which system has
    it, so simulating 13,485 systems individually would repeat the same
    transposition thousands of times. Grouping first makes a full-year hourly
    sweep tractable.
    """
    df = fleet.copy()
    df["tilt_deg"] = pd.to_numeric(df["Tilt"], errors="coerce")
    df["azimuth_deg"] = pd.to_numeric(df["Azimuth"], errors="coerce")
    df["capacity_kw"] = pd.to_numeric(df["System Size DC"], errors="coerce")
    df = df.dropna(subset=["tilt_deg", "azimuth_deg", "capacity_kw"])

    df["tilt_deg"] = df["tilt_deg"].round(0)
    df["azimuth_deg"] = df["azimuth_deg"].round(0) % 360

    grouped = (
        df.groupby(["tilt_deg", "azimuth_deg"], as_index=False)["capacity_kw"]
        .sum()
        .sort_values("capacity_kw", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def poa_global(times_utc, solpos, clearsky, dni_extra, tilt_deg, azimuth_deg):
    """Plane-of-array global irradiance (W/m2) for one orientation."""
    return pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )["poa_global"]


def fleet_mw(geoms, times_utc, solpos, clearsky, dni_extra, tilts, azimuths):
    """Citywide AC MW timeseries for a given set of per-geometry orientations.

    tilts/azimuths are arrays parallel to geoms, so an error model can shift
    them without touching the capacity weights.

    Still pvlib's own transposition -- the only trick is shape. Times go down
    the rows as (n_times, 1) and orientations across the columns as
    (1, n_geoms), so pvlib broadcasts to the full (n_times, n_geoms) matrix in
    one call instead of 2275 separate ones. Verified against the per-geometry
    loop in poa_global() to 2e-13 MW; see check_vectorisation().
    """
    zenith = solpos["apparent_zenith"].to_numpy()[:, None]
    solar_az = solpos["azimuth"].to_numpy()[:, None]
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilts[None, :],
        surface_azimuth=azimuths[None, :],
        solar_zenith=zenith,
        solar_azimuth=solar_az,
        dni=clearsky["dni"].to_numpy()[:, None],
        ghi=clearsky["ghi"].to_numpy()[:, None],
        dhi=clearsky["dhi"].to_numpy()[:, None],
        dni_extra=dni_extra.to_numpy()[:, None],
        model="haydavies",
    )["poa_global"]
    capacity_kw = geoms["capacity_kw"].to_numpy()[None, :]
    return (poa * capacity_kw / 1000.0 * DERATE).sum(axis=1) / 1000.0


def check_vectorisation(geoms, times_utc, solpos, clearsky, dni_extra, tilts, azimuths):
    """Confirm the broadcast form matches the plain per-geometry loop.

    Cheap insurance against a silent broadcasting mistake -- a transposed axis
    here would produce a smooth, plausible, completely wrong curve.
    """
    loop_kw = np.zeros(len(times_utc))
    for i in range(len(geoms)):
        poa = poa_global(times_utc, solpos, clearsky, dni_extra, tilts[i], azimuths[i])
        loop_kw += geoms["capacity_kw"].iloc[i] * (poa.to_numpy() / 1000.0) * DERATE
    return np.abs(loop_kw / 1000.0 - fleet_mw(geoms, times_utc, solpos, clearsky,
                                              dni_extra, tilts, azimuths)).max()


def apply_error(geoms, d_tilt, d_azimuth, flat_azimuth, rng=None, scatter=False):
    """Perturb every geometry's tilt/azimuth.

    scatter=False -> systematic bias: every orientation shifted the same way.
    scatter=True  -> independent noise: d_tilt/d_azimuth are used as the
                     standard deviation of a per-geometry normal draw.

    `flat_azimuth` carries the fleet's capacity-weighted median azimuth of
    genuinely-tilted systems, for use on rows whose recorded azimuth is the
    flat-system placeholder rather than a real bearing.
    """
    base_tilt = geoms["tilt_deg"].to_numpy(dtype=float)
    base_az = geoms["azimuth_deg"].to_numpy(dtype=float)
    n = len(base_tilt)

    if scatter:
        tilt_shift = rng.normal(0.0, d_tilt, n) if d_tilt > 0 else np.zeros(n)
        az_shift = rng.normal(0.0, d_azimuth, n) if d_azimuth > 0 else np.zeros(n)
    else:
        tilt_shift = np.full(n, float(d_tilt))
        az_shift = np.full(n, float(d_azimuth))

    tilts = np.clip(base_tilt + tilt_shift, 0.0, 90.0)

    azimuths = (base_az + az_shift) % 360.0

    # A system recorded flat stores azimuth=0 as a placeholder, not a bearing
    # -- 30.0 of the fleet's 81.1 MW sits on that placeholder. Taken literally,
    # a positive tilt error would make 37% of SF's capacity face due north,
    # the worst possible orientation, and the sensitivity result would be
    # badly pessimistic for a purely clerical reason. So a flat system that an
    # error model tilts is assumed to tilt the way SF actually tilts things:
    # `flat_azimuth`, the capacity-weighted median azimuth of the genuinely-
    # tilted fleet (which comes out at 180 deg, due south).
    #
    # Systems that stay flat are unaffected either way -- a horizontal plane
    # has no azimuth and pvlib ignores surface_azimuth at tilt=0.
    was_flat = base_tilt <= FLAT_TILT_THRESHOLD_DEG
    if FLAT_STAYS_FLAT:
        tilts = np.where(was_flat, base_tilt, tilts)
    else:
        azimuths = np.where(was_flat, flat_azimuth, azimuths)

    return tilts, azimuths


def gap_slice_worst_case(location, grids):
    """The question this whole script exists to answer.

    SF's fleet data already reports measured tilt/azimuth for 95.9% of
    capacity. Only 376 systems (4.1%) are gap-filled from ZIP-level medians.
    So the most a perfect geometry measurement could buy is however much that
    4.1% slice can move the citywide number -- and this bounds it by making
    that slice as wrong as it can physically be.
    """
    raw = load_sf_fleet()
    gapfilled = (raw["Tilt"].astype(str) == "multiple") | (raw["Azimuth"].astype(str) == "multiple")
    fleet = resolve_tilt_azimuth(raw)
    fleet["gapfilled"] = gapfilled.values
    fleet["tilt_deg"] = pd.to_numeric(fleet["Tilt"], errors="coerce")
    fleet["azimuth_deg"] = pd.to_numeric(fleet["Azimuth"], errors="coerce") % 360
    fleet["capacity_kw"] = pd.to_numeric(fleet["System Size DC"], errors="coerce")
    fleet = fleet.dropna(subset=["tilt_deg", "azimuth_deg", "capacity_kw"]).reset_index(drop=True)

    total_mw = fleet["capacity_kw"].sum() / 1000.0
    gap_mw = fleet.loc[fleet["gapfilled"], "capacity_kw"].sum() / 1000.0
    print("\n" + "=" * 74)
    print("BOUND -- only the gap-filled slice is wrong; the rest is measured data")
    print("=" * 74)
    print(f"Gap-filled: {int(fleet['gapfilled'].sum())} systems, {gap_mw:.2f} MW "
          f"({100 * gap_mw / total_mw:.1f}% of {total_mw:.1f} MW)")

    base_tilt = fleet["tilt_deg"].to_numpy(dtype=float)
    base_az = fleet["azimuth_deg"].to_numpy(dtype=float)
    is_gap = fleet["gapfilled"].to_numpy()

    for season in ["summer", "winter"]:
        t, solpos, cs, dnie = grids[season]
        local = t.tz_convert(SF_TZ)
        base = fleet_mw(fleet, t, solpos, cs, dnie, base_tilt, base_az)
        print(f"\n  {season}: citywide MW error if the gap-filled slice were this wrong")
        for label, d_tilt, d_az in [
            ("+20 deg tilt", 20, 0),
            ("-20 deg tilt", -20, 0),
            ("180 deg azimuth flip", 0, 180),
            ("both, simultaneously", 20, 180),
        ]:
            tl, az = base_tilt.copy(), base_az.copy()
            tl[is_gap] = np.clip(tl[is_gap] + d_tilt, 0.0, 90.0)
            az[is_gap] = (az[is_gap] + d_az) % 360.0
            mw = fleet_mw(fleet, t, solpos, cs, dnie, tl, az)
            cells = []
            for h in [9, 12, 17]:
                i = np.argmin(np.abs(local.hour + local.minute / 60 - h))
                b = base[i]
                cells.append(f"{h:02d}:00 {100 * (mw[i] - b) / b:+6.2f}%" if b > 0.01 else f"{h:02d}:00    n/a")
            print(f"     {label:24s} " + "   ".join(cells))


def main():
    fleet = resolve_tilt_azimuth(load_sf_fleet())
    geoms = build_unique_geometries(fleet)
    total_capacity_mw = geoms["capacity_kw"].sum() / 1000.0
    print(f"Fleet: {len(fleet)} systems, {total_capacity_mw:.1f} MW DC")
    print(f"Unique (tilt, azimuth) pairs: {len(geoms)}")

    tilted = geoms[geoms["tilt_deg"] > FLAT_TILT_THRESHOLD_DEG]
    flat_capacity_mw = geoms.loc[geoms["tilt_deg"] <= FLAT_TILT_THRESHOLD_DEG, "capacity_kw"].sum() / 1000.0
    flat_azimuth = circular_median_deg(
        pd.Series(np.repeat(tilted["azimuth_deg"].to_numpy(),
                            np.maximum(1, tilted["capacity_kw"].round().astype(int))))
    )
    print(f"Recorded flat (tilt <= {FLAT_TILT_THRESHOLD_DEG} deg): {flat_capacity_mw:.1f} MW "
          f"({100 * flat_capacity_mw / total_capacity_mw:.1f}% of capacity)")
    print(f"Capacity-weighted median azimuth of tilted systems: {flat_azimuth:.0f} deg")

    # --- time grids, all UTC ---
    summer = pd.date_range("2026-06-21 00:00", "2026-06-22 00:00", freq="15min", tz="UTC")
    winter = pd.date_range("2026-12-21 00:00", "2026-12-22 00:00", freq="15min", tz="UTC")
    annual = pd.date_range("2026-01-01 00:00", "2027-01-01 00:00", freq="1h", tz="UTC")[:-1]

    location = pvlib.location.Location(SF_LAT, SF_LON, SF_TZ, SF_ALT)

    def prep(times_utc):
        solpos = location.get_solarposition(times_utc)
        clearsky = location.get_clearsky(times_utc, model="ineichen", solar_position=solpos)
        dni_extra = pvlib.irradiance.get_extra_radiation(times_utc)
        return solpos, clearsky, dni_extra

    grids = {name: (t,) + prep(t) for name, t in
             [("summer", summer), ("winter", winter), ("annual", annual)]}

    # --- sanity checks before trusting anything downstream ---
    print("\n--- sanity checks (CLAUDE.md ranges) ---")
    t, solpos, cs, dnie = grids["summer"]
    noon_ghi = cs["ghi"].max()
    print(f"Peak clear-sky GHI, summer:      {noon_ghi:.0f} W/m2   (expect 950-1050)")

    rng = np.random.default_rng(0)
    tilts0, az0 = apply_error(geoms, 0, 0, flat_azimuth, rng)
    mw_summer = fleet_mw(geoms, t, solpos, cs, dnie, tilts0, az0)
    peak_frac = mw_summer.max() / total_capacity_mw
    print(f"Peak AC / nameplate DC:          {100 * peak_frac:.0f}%       (expect 70-85)")

    local = t.tz_convert(SF_TZ)
    night = (local.hour < 4) | (local.hour > 22)
    print(f"Output at night:                 {mw_summer[night].max():.6f} MW  (expect exactly 0)")

    vec_err = check_vectorisation(geoms, t, solpos, cs, dnie, tilts0, az0)
    print(f"Vectorised vs per-geometry loop: {vec_err:.2e} MW  (expect ~0)")

    # --- the sweep ---
    TILT_ERRORS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    AZ_ERRORS = [-60, -40, -20, 0, 20, 40, 60]
    HOURS = [9, 12, 17]

    def mw_at(times_local, series, hour):
        idx = np.argmin(np.abs(times_local.hour + times_local.minute / 60 - hour))
        return series[idx]

    results = []
    for season in ["summer", "winter"]:
        t, solpos, cs, dnie = grids[season]
        local = t.tz_convert(SF_TZ)
        base_mw = fleet_mw(geoms, t, solpos, cs, dnie, *apply_error(geoms, 0, 0, flat_azimuth, rng))
        for mode, scatter in [("bias", False), ("scatter", True)]:
            for d in TILT_ERRORS:
                if scatter and d < 0:
                    continue
                tl, az = apply_error(geoms, abs(d) if scatter else d, 0, flat_azimuth,
                                     np.random.default_rng(1), scatter)
                mw = fleet_mw(geoms, t, solpos, cs, dnie, tl, az)
                for h in HOURS:
                    b, v = mw_at(local, base_mw, h), mw_at(local, mw, h)
                    results.append(dict(season=season, mode=mode, kind="tilt", delta=d,
                                        hour=h, base_mw=b, mw=v,
                                        pct=100 * (v - b) / b if b > 0.01 else np.nan))
            for d in AZ_ERRORS:
                if scatter and d < 0:
                    continue
                tl, az = apply_error(geoms, 0, abs(d) if scatter else d, flat_azimuth,
                                     np.random.default_rng(1), scatter)
                mw = fleet_mw(geoms, t, solpos, cs, dnie, tl, az)
                for h in HOURS:
                    b, v = mw_at(local, base_mw, h), mw_at(local, mw, h)
                    results.append(dict(season=season, mode=mode, kind="azimuth", delta=d,
                                        hour=h, base_mw=b, mw=v,
                                        pct=100 * (v - b) / b if b > 0.01 else np.nan))

    res = pd.DataFrame(results)

    # --- annual clear-sky energy, for contrast with the instantaneous numbers ---
    t, solpos, cs, dnie = grids["annual"]
    base_annual = fleet_mw(geoms, t, solpos, cs, dnie, *apply_error(geoms, 0, 0, flat_azimuth, rng)).sum()
    annual_rows = []
    for kind, deltas in [("tilt", TILT_ERRORS), ("azimuth", AZ_ERRORS)]:
        for d in deltas:
            dt, da = (d, 0) if kind == "tilt" else (0, d)
            tl, az = apply_error(geoms, dt, da, flat_azimuth, np.random.default_rng(1), False)
            e = fleet_mw(geoms, t, solpos, cs, dnie, tl, az).sum()
            annual_rows.append(dict(kind=kind, delta=d, pct=100 * (e - base_annual) / base_annual))
    annual_df = pd.DataFrame(annual_rows)

    print("\n" + "=" * 74)
    print("SYSTEMATIC BIAS -- every system wrong in the same direction")
    print("=" * 74)
    for kind in ["tilt", "azimuth"]:
        print(f"\n  {kind.upper()} error -> % error in citywide MW")
        sub = res[(res["mode"] == "bias") & (res["kind"] == kind)]
        piv = sub.pivot_table(index="delta", columns=["season", "hour"], values="pct")
        ann = annual_df[annual_df["kind"] == kind].set_index("delta")["pct"]
        piv[("annual", "energy")] = ann
        print(piv.round(1).to_string())

    print("\n" + "=" * 74)
    print("RANDOM SCATTER -- each system wrong independently (delta = 1 sigma)")
    print("=" * 74)
    for kind in ["tilt", "azimuth"]:
        print(f"\n  {kind.upper()} scatter -> % error in citywide MW")
        sub = res[(res["mode"] == "scatter") & (res["kind"] == kind)]
        print(sub.pivot_table(index="delta", columns=["season", "hour"], values="pct").round(1).to_string())

    gap_slice_worst_case(location, grids)

    res.to_csv("data/level3_geometry_sensitivity.csv", index=False)
    print("\nSaved data/level3_geometry_sensitivity.csv")
    plot_sensitivity(res, annual_df, "data/level3_geometry_sensitivity.png")


def plot_sensitivity(res, annual_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = {9: "#2a78d6", 12: "#eb6834", 17: "#1baf7a"}
    for ax, kind in zip(axes, ["tilt", "azimuth"]):
        sub = res[(res["mode"] == "bias") & (res["kind"] == kind) & (res["season"] == "summer")]
        for h in [9, 12, 17]:
            s = sub[sub["hour"] == h].sort_values("delta")
            ax.plot(s["delta"], s["pct"], "o-", color=colors[h], label=f"{h}:00 local, summer")
        ann = annual_df[annual_df["kind"] == kind].sort_values("delta")
        ax.plot(ann["delta"], ann["pct"], "s--", color="#52514e", label="annual energy")
        ax.axhline(0, color="#b9b6ae", lw=1)
        ax.axvline(0, color="#b9b6ae", lw=1)
        ax.set_xlabel(f"systematic {kind} error (deg)")
        ax.set_title(f"Citywide MW error vs {kind} bias")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("error in citywide MW (%)")
    axes[1].legend(fontsize=9)
    fig.suptitle("SF fleet: how much does wrong geometry cost? (clear-sky, 81.1 MW DC)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
