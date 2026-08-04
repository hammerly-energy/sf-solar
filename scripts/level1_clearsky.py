"""
Level 1 — one hypothetical SF rooftop, clear-sky output, one summer day.

System: 5 kW DC, 20 deg tilt, south-facing (180 deg in pvlib's
N=0/E=90/S=180/W=270 convention), 4 kW AC inverter (1.25 DC:AC ratio,
a standard design ratio -- this is why the AC curve flattens near
solar noon instead of following the DC curve all the way up).

All timestamps are computed in UTC and converted to Pacific only for
the printed sanity checks and the plot, per the time convention in
CLAUDE.md.

Run: uv run python scripts/level1_clearsky.py
"""

import pandas as pd
import pvlib
import matplotlib.pyplot as plt

# --- Site ---
LATITUDE = 37.7749
LONGITUDE = -122.4194
ELEVATION_M = 16
TIMEZONE = "America/Los_Angeles"

# --- Hypothetical system ---
TILT_DEG = 20
AZIMUTH_DEG = 180  # pvlib convention: 180 = south
DC_NAMEPLATE_W = 5000
AC_NAMEPLATE_W = 4000  # DC:AC ratio 1.25
TEMP_COEFF_PCT_PER_C = -0.35  # matches CLAUDE.md's stated value

# No real weather yet (that's a later level) -- assumed constant for
# a clear, mild SF summer day.
AMBIENT_TEMP_C = 20
WIND_SPEED_MS = 2

LOCAL_DATE = "2026-06-21"  # near summer solstice


def main():
    site = pvlib.location.Location(
        LATITUDE, LONGITUDE, tz=TIMEZONE, altitude=ELEVATION_M
    )

    # Build the day as local midnight-to-midnight, then convert to UTC
    # and do all physics in UTC.
    local_start = pd.Timestamp(LOCAL_DATE, tz=TIMEZONE)
    local_end = local_start + pd.Timedelta(days=1)
    times_utc = pd.date_range(
        local_start.tz_convert("UTC"), local_end.tz_convert("UTC"), freq="5min"
    )

    solar_position = site.get_solarposition(times_utc)
    clearsky = site.get_clearsky(times_utc, model="ineichen")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT_DEG,
        surface_azimuth=AZIMUTH_DEG,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )

    cell_temp = pvlib.temperature.faiman(
        poa["poa_global"], AMBIENT_TEMP_C, WIND_SPEED_MS
    )

    dc_power_w = pvlib.pvsystem.pvwatts_dc(
        poa["poa_global"],
        cell_temp,
        pdc0=DC_NAMEPLATE_W,
        gamma_pdc=TEMP_COEFF_PCT_PER_C / 100,
    )

    ac_power_w = pvlib.inverter.pvwatts(dc_power_w, pdc0=AC_NAMEPLATE_W)

    # --- sanity checks, per the table in CLAUDE.md ---
    peak_ac_w = ac_power_w.max()
    peak_pct_of_dc = 100 * peak_ac_w / DC_NAMEPLATE_W
    daily_energy_kwh = ac_power_w.sum() * (5 / 60) / 1000  # 5-min steps -> kWh
    night_max_w = ac_power_w[
        (solar_position["apparent_elevation"] < -2)
    ].max()

    print(f"Peak AC output: {peak_ac_w:.0f} W ({peak_pct_of_dc:.0f}% of {DC_NAMEPLATE_W} W DC nameplate)")
    print("  expected: 70-85% of nameplate")
    print(f"Daily energy: {daily_energy_kwh:.1f} kWh")
    print("  expected: ~25-35 kWh for a 5 kW system on a clear summer day")
    print(f"Max output while sun is below horizon: {night_max_w:.4f} W")
    print("  expected: exactly 0")

    # --- plot, in local time for display ---
    dc_power_kw = dc_power_w / 1000
    ac_power_kw = ac_power_w / 1000
    dc_power_kw.index = dc_power_kw.index.tz_convert(TIMEZONE)
    ac_power_kw.index = ac_power_kw.index.tz_convert(TIMEZONE)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dc_power_kw.index, dc_power_kw, linestyle="--", label="DC output")
    ax.plot(ac_power_kw.index, ac_power_kw, label="AC output (after inverter clipping)")
    ax.axhline(DC_NAMEPLATE_W / 1000, color="gray", linewidth=0.8, linestyle=":", label="DC nameplate (5 kW)")
    ax.set_xlabel(f"Time ({TIMEZONE})")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Hypothetical SF rooftop, clear-sky output -- {LOCAL_DATE}\n5 kW DC, 20° tilt, south-facing")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = "data/level1_clearsky.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
