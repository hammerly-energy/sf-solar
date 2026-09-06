"""Visual audit: draw each detected array's segmentation OUTLINE over native 10cm SWISSIMAGE.

One panel per stitched array (largest first), not one per tile -- see
find_pv_arrays() in level3_cv_detection.py.

Reads the .npz cache written by level3_cv_detection.py so this doesn't re-run
model inference. Outline (not a filled overlay) on purpose -- a translucent
fill hides the roof texture you need in order to tell a PV module from a
skylight, which is the whole point of looking.

    uv run --isolated --with rasterio --with requests --with numpy \
      --with matplotlib python scripts/audit_detections.py zurich
"""
import sys
import numpy as np
import requests
import rasterio
import rasterio.warp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from affine import Affine
from rasterio.merge import merge as merge_rasters

TARGET_MPP = 0.2359  # must match level3_cv_detection.py


def fetch_native_crop(bounds_2056):
    """Native-resolution (10cm) SWISSIMAGE for a bbox given in EPSG:2056."""
    lon_lat = rasterio.warp.transform_bounds("EPSG:2056", "EPSG:4326", *bounds_2056)
    resp = requests.get(
        "https://data.geo.admin.ch/api/stac/v1/search",
        params={"collections": "ch.swisstopo.swissimage-dop10",
                "bbox": ",".join(str(b) for b in lon_lat)},
        timeout=60,
    )
    resp.raise_for_status()
    items = sorted(resp.json()["features"], key=lambda i: i["properties"]["datetime"], reverse=True)
    hrefs = [next(v["href"] for k, v in i["assets"].items() if k.endswith("_0.1_2056.tif")) for i in items]
    data, _ = merge_rasters(hrefs, bounds=bounds_2056, indexes=[1, 2, 3])
    return np.transpose(data, (1, 2, 0))


def draw_mask_outline(ax, mask, extent, **kwargs):
    """Closed outline of a boolean mask. contour() cannot close a line that
    runs off the edge of its array, and a blob's bounding box has mask pixels
    on all four edges by definition -- so pad a ring of False around it and
    widen the extent by that same pixel to keep the geometry aligned."""
    left, right, bottom, top = extent
    px = (right - left) / mask.shape[1]
    py = (top - bottom) / mask.shape[0]
    ax.contour(np.pad(mask, 1).astype(float), levels=[0.5],
               extent=(left - px, right + px, bottom - py, top + py),
               origin="upper", **kwargs)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "zurich"
    cache = np.load(f"data/level3_cv_{mode}_detections.npz")
    transform = Affine.from_gdal(*cache["transform"])
    n = int(cache["n_arrays"])

    dets = []
    for i in range(n):
        p_pv, y0, x0, area, lat, lon, tilt, az, n_tiles, truncated = cache[f"meta_{i}"]
        mask = cache[f"mask_{i}"].astype(bool)
        if mask.sum() == 0:
            continue
        dets.append(dict(i=i, p_pv=p_pv, y0=int(y0), x0=int(x0), area=area,
                         lat=lat, lon=lon, tilt=tilt, az=az, mask=mask,
                         n_tiles=int(n_tiles), truncated=bool(truncated)))
    dets.sort(key=lambda d: -d["area"])
    print(f"{mode}: {n} detections cached, {len(dets)} with PV pixels")

    ncols = 4
    nrows = int(np.ceil(len(dets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, d in zip(axes, dets):
        h, w = d["mask"].shape
        # tile extent in EPSG:2056 metres
        left, top = transform * (d["x0"], d["y0"])
        right, bottom = transform * (d["x0"] + w, d["y0"] + h)
        # pad outward so context around the array is visible. The proportional
        # term alone leaves small arrays cropped flush to their own edge, so
        # there is an absolute floor too (must match CONTEXT_MARGIN_* in
        # level3_cv_detection.py).
        pad = max(0.35 * max(right - left, top - bottom), 10.0)
        crop = (left - pad, bottom - pad, right + pad, top + pad)
        img = fetch_native_crop(crop)
        ax.imshow(img, extent=(crop[0], crop[2], crop[1], crop[3]), origin="upper")
        draw_mask_outline(ax, d["mask"], (left, right, bottom, top),
                          colors="#eb6834", linewidths=2.0)
        tilt_s = "--" if np.isnan(d["tilt"]) else f"{d['tilt']:.1f}/{d['az']:.0f}"
        seam = []
        if d["n_tiles"] > 1:
            seam.append(f"{d['n_tiles']} tiles")
        if d["truncated"]:
            seam.append("cut at seam?")
        seam_s = f"   [{', '.join(seam)}]" if seam else ""
        ax.set_title(f"p={d['p_pv']:.3f}   {d['area']:.0f} m²   tilt/az={tilt_s}\n"
                     f"{crop[2]-crop[0]:.0f} m across{seam_s}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"  p={d['p_pv']:.3f} {d['area']:.0f}m2 {crop[2]-crop[0]:.0f}m crop")

    for ax in axes[len(dets):]:
        ax.axis("off")

    fig.suptitle(f"{mode}: segmentation outline (orange) over SWISSIMAGE 10 cm native "
                 f"-- is the outlined thing actually PV?", fontsize=15)
    fig.tight_layout()
    out = f"data/level3_cv_{mode}_audit_outlines.png"
    fig.savefig(out, dpi=95, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
