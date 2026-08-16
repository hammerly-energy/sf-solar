"""
Level 3 -- the real SF solar fleet.

Loads the CA DG Stats Interconnected Project Sites dataset (PG&E, since
San Francisco is entirely PG&E territory), filters to San Francisco,
and produces a cleaned fleet table plus sanity-check plots.

Data source: https://www.californiadgstats.ca.gov/downloads/
"Interconnected Project Sites Data Set", current as of 2026-05-31.
Excludes pending and decommissioned systems already (per the CPUC read-me),
so every row here is a real, currently-interconnected system.

Azimuth convention, confirmed against the official data key
(data/raw/data_key.xlsx, "Project Site Data Set Key" sheet, row 18):
"For tilted arrays, the azimuth is number of degrees clockwise from true
north" -- i.e. N=0, E=90, S=180, W=270. Matches pvlib's surface_azimuth
convention directly; no conversion needed.

This dataset has no lat/lon or street address -- only Service City/Zip/
County. Since San Francisco is small enough that solar position and
clear-sky irradiance barely vary across it, that's fine: every system
in Level 3 shares Level 1's single reference location, and only each
system's own tilt/azimuth/capacity varies.

Run: uv run python scripts/level3_fleet.py
"""

from pathlib import Path

import contextily as ctx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw/Interconnected_Project_Sites_2026-05-31")
PGE_FILES = [
    RAW_DIR / "PGE_Interconnected_Project_Sites_Historical-Jan2020.csv",
    RAW_DIR / "PGE_Interconnected_Project_Sites_Jan2020-May2026.csv",
]
COLUMNS = [
    "Application Id",
    "Application Status",
    "Service County",
    "Service Zip",
    "Technology Type",
    "System Size DC",
    "System Size AC",
    "Inverter Size (kW AC)",
    "Tilt",
    "Azimuth",
]


def load_sf_fleet():
    """Load PG&E project sites, filter to San Francisco PV systems.

    Excludes non-solar technology types (pure battery storage, fuel
    cells, wind) and rows with non-positive DC capacity, which are
    data errors, not real systems.
    """
    frames = [pd.read_csv(f, usecols=COLUMNS, low_memory=False) for f in PGE_FILES]
    df = pd.concat(frames, ignore_index=True)

    sf = df[df["Service County"].astype(str).str.strip().str.upper() == "SAN FRANCISCO"].copy()

    is_pv = sf["Technology Type"].astype(str).str.contains("Photovoltaic", na=False)
    has_positive_capacity = sf["System Size DC"] > 0
    fleet = sf[is_pv & has_positive_capacity].copy()

    return fleet.reset_index(drop=True)


def resolve_tilt_azimuth(fleet):
    """Fill in a numeric tilt/azimuth for systems recorded as 'multiple'.

    These are multi-array installs (e.g. panels split across a front and
    back roof plane) where the interconnection data doesn't give per-array
    geometry -- see architecture doc §2 "Known data quirks". Affects ~4%
    of citywide DC capacity, so the choice of default matters a little,
    not a lot.

    Returns a copy of fleet with 'Tilt' and 'Azimuth' fully numeric.
    """
    MIN_REFERENCE_SYSTEMS = 5

    fleet = fleet.copy()
    fleet["Tilt"] = pd.to_numeric(fleet["Tilt"], errors="coerce")
    fleet["Azimuth"] = pd.to_numeric(fleet["Azimuth"], errors="coerce")
    is_multiple = fleet["Tilt"].isna() | fleet["Azimuth"].isna()

    known = fleet[~is_multiple]
    # Azimuth is only meaningful for tilted arrays (confirmed against the
    # data key -- flat systems record azimuth=0 as a placeholder), so the
    # azimuth prior must only be built from tilted known systems.
    known_tilted = known[known["Tilt"] > 0]

    citywide_tilt = known["Tilt"].median()
    citywide_azimuth = circular_median_deg(known_tilted["Azimuth"])

    zip_tilt = known.groupby("Service Zip")["Tilt"].median()
    zip_azimuth = known_tilted.groupby("Service Zip")["Azimuth"].apply(circular_median_deg)
    zip_reference_count = known_tilted.groupby("Service Zip").size()

    for idx in fleet.index[is_multiple]:
        zip_code = fleet.at[idx, "Service Zip"]
        has_local_prior = zip_reference_count.get(zip_code, 0) >= MIN_REFERENCE_SYSTEMS
        fleet.at[idx, "Tilt"] = zip_tilt.get(zip_code, citywide_tilt) if has_local_prior else citywide_tilt
        fleet.at[idx, "Azimuth"] = zip_azimuth.get(zip_code, citywide_azimuth) if has_local_prior else citywide_azimuth

    return fleet


def circular_median_deg(degrees):
    """Median of a set of compass-bearing angles, correct across the 0/360 wrap.

    A plain numeric median of e.g. [10, 350] gives 180 (exactly backwards).
    Fix: find the circular mean as a reference direction, re-express every
    angle as a signed offset from it (safely within -180..180, no wrap),
    take the ordinary median of the offsets, then shift back. This keeps
    the median's robustness to outliers (unlike using the mean directly)
    while still handling the wraparound correctly.
    """
    values = degrees.dropna()
    radians = np.deg2rad(values)
    mean_angle_deg = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    offsets = (values - mean_angle_deg + 180) % 360 - 180
    return (mean_angle_deg + offsets.median()) % 360


def _check_circular_median():
    """Regression check for the 0/360 wraparound bug circular_median_deg exists to fix."""
    wrapped = circular_median_deg(pd.Series([350, 10, 355, 5]))
    assert min(wrapped, 360 - wrapped) < 1e-6, f"expected ~0/360, got {wrapped}"

    not_wrapped = circular_median_deg(pd.Series([170, 180, 190]))
    assert abs(not_wrapped - 180) < 1e-6, f"expected ~180, got {not_wrapped}"


def sanity_check(fleet):
    total_dc_mw = fleet["System Size DC"].sum() / 1000
    print(f"SF PV systems: {len(fleet)}")
    print(f"Total DC capacity: {total_dc_mw:.1f} MW")
    print("  expected ballpark (per architecture doc): tens of MW")

    needs_gapfill = (fleet["Tilt"].astype(str) == "multiple") | (fleet["Azimuth"].astype(str) == "multiple")
    gapfill_mw = fleet.loc[needs_gapfill, "System Size DC"].sum() / 1000
    pct = 100 * gapfill_mw / total_dc_mw
    print(f"Systems needing tilt/azimuth gap-fill ('multiple'): {needs_gapfill.sum()} ({pct:.1f}% of capacity)")


def plot_fleet_overview(fleet, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    by_zip = fleet.groupby("Service Zip")["System Size DC"].sum().sort_values(ascending=False) / 1000
    axes[0, 0].bar(by_zip.index.astype(int).astype(str), by_zip.values, color="steelblue")
    axes[0, 0].set_title("DC capacity by ZIP code")
    axes[0, 0].set_ylabel("MW DC")
    axes[0, 0].tick_params(axis="x", rotation=90, labelsize=6)

    tilt_numeric = pd.to_numeric(fleet["Tilt"], errors="coerce")
    axes[0, 1].hist(tilt_numeric.dropna(), bins=30, color="darkorange")
    axes[0, 1].set_title("Tilt distribution (deg)")
    axes[0, 1].set_xlabel("Tilt (deg from horizontal)")

    az_numeric = pd.to_numeric(fleet["Azimuth"], errors="coerce")
    axes[1, 0].hist(az_numeric.dropna(), bins=36, color="seagreen")
    axes[1, 0].set_title("Azimuth distribution (deg, 0=N, 180=S)")
    axes[1, 0].set_xlabel("Azimuth (deg clockwise from N)")
    axes[1, 0].axvline(180, color="gray", linestyle=":", linewidth=1, label="South")
    axes[1, 0].legend()

    axes[1, 1].hist(fleet["System Size DC"], bins=50, color="indianred")
    axes[1, 1].set_title("System size distribution")
    axes[1, 1].set_xlabel("DC nameplate (kW)")
    axes[1, 1].set_yscale("log")

    fig.suptitle(f"San Francisco PV fleet, n={len(fleet)} systems (PG&E interconnection data)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved fleet overview plot to {output_path}")


def load_zcta_centroids(zip_codes):
    """Look up real ZCTA centroid lat/lon from the Census Gazetteer.

    Source: https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/
    These are official centroids of the ZIP Code Tabulation Area polygons,
    not something estimated or hand-typed.
    """
    gaz = pd.read_csv(
        "data/raw/2024_Gaz_zcta_national.txt",
        sep="\t",
        dtype={"GEOID": str},
    )
    gaz.columns = gaz.columns.str.strip()
    gaz = gaz.set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]

    zip_strs = [f"{int(z):05d}" for z in zip_codes]
    missing = [z for z in zip_strs if z not in gaz.index]
    if missing:
        print(f"Warning: {len(missing)} ZIPs not found in gazetteer, will be dropped from map: {missing}")

    return gaz.reindex(zip_strs)


def plot_zip_map(fleet, output_path):
    """Bubble map of DC capacity by ZIP centroid.

    This is the most honest map the data supports: the dataset has no
    street address or lat/lon, only Service Zip, so this shows ZIP-level
    location, not individual rooftops.
    """
    by_zip = fleet.groupby("Service Zip")["System Size DC"].agg(["sum", "count"])
    by_zip["sum_mw"] = by_zip["sum"] / 1000

    centroids = load_zcta_centroids(by_zip.index)
    by_zip.index = [f"{int(z):05d}" for z in by_zip.index]
    plot_df = by_zip.join(centroids).dropna(subset=["INTPTLAT", "INTPTLONG"])

    fig, ax = plt.subplots(figsize=(8, 9))
    scatter = ax.scatter(
        plot_df["INTPTLONG"],
        plot_df["INTPTLAT"],
        s=plot_df["sum_mw"] * 60,
        c=plot_df["sum_mw"],
        cmap="YlOrRd",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.85,
    )
    for zip_code, row in plot_df.iterrows():
        ax.annotate(
            zip_code,
            (row["INTPTLONG"], row["INTPTLAT"]),
            fontsize=6,
            ha="center",
            va="center",
        )

    # Real street/coastline basemap under the bubbles, so the plot reads as
    # an actual map of SF rather than bare lat/lon points. contextily
    # reprojects OSM-derived tiles to match our WGS84 (lat/lon) axes.
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.3)  # rough correction for latitude at SF (~37.7N)
    fig.suptitle("SF PV fleet by ZIP code centroid", fontsize=13)
    ax.set_title(
        "bubble size/color = DC capacity (MW) -- ZIP-level only, no street addresses in this data",
        fontsize=9,
    )
    fig.colorbar(scatter, ax=ax, label="MW DC")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    print(f"Saved ZIP map to {output_path}")


def main():
    _check_circular_median()

    fleet = load_sf_fleet()
    sanity_check(fleet)
    plot_fleet_overview(fleet, "data/level3_fleet_overview.png")
    plot_zip_map(fleet, "data/level3_zip_map.png")

    fleet = resolve_tilt_azimuth(fleet)
    print(f"\nAfter gap-fill: {fleet['Tilt'].isna().sum()} systems still missing tilt/azimuth (expect 0)")


if __name__ == "__main__":
    main()
