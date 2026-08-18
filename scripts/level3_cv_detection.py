"""
EXPERIMENTAL -- Level 3 side quest, not part of the main pipeline.
Full context: CLAUDE.md, "Building-level detection detour" section.

Detect rooftop solar panels using DeepSolar-3M's published model (Prabha et
al., ICLR 2025 Climate Change AI workshop), then fit each detected array's
roof plane (tilt/azimuth) from LiDAR. Three stages: classify (ViT-MAE-large
+ LoRA) -> segment (SegFormer) -> fit a plane to the segmented footprint's
DSM pixels.

Four modes (first CLI arg):
  zurich     -- Wiedikon, Zurich. WORKING, but turned out almost entirely
             near-flat commercial/industrial roofs -- not much of a stress
             test for the plane-fitting code. SWISSIMAGE (10cm/px) +
             swissSURFACE3D DSM (0.5m/px), both openly licensed. Classifier
             clears its own 0.5 decision threshold here; full pipeline runs
             end-to-end, including DSM mosaicking across all tiles covering
             the bbox.
  altstetten -- Altstetten, Zurich (Kreis 9). Picked via OpenStreetMap's
             Overpass API (real generator:source=solar tags, not a guess) to
             find a residential area with pitched roofs and confirmed real
             solar, as a contrast to Wiedikon's flat roofs.
  smoke      -- SF Richmond (94118/94121), centered on a permit-verified
             address. KNOWN BROKEN: NAIP is only 0.6m/px vs the ~0.24m/px
             (Google Maps zoom>=19) the model expects; every tile scored
             near zero even at a real positive location.
  control    -- SF Sunset Reservoir solar farm (positive control, ~24,000
             panels, unmistakable in the imagery). KNOWN BROKEN for the same
             resolution reason -- max score 0.10 even here, proving the SF
             failure is resolution (and/or ground-mount-vs-rooftop domain
             mismatch), not a pipeline bug. This is what motivated the
             Zurich pivot.

Model weights: data/raw/deepsolar_weights/ (gitignored, ~1.2GB). Re-download
via gdown from the Google Drive link in the DeepSolar-3M GitHub README if
missing (github.com/rajanieprabha/DeepSolar-3M).

Run (note --isolated -- this repo's own .venv has a newer NumPy that
conflicts with these pins otherwise):
  uv run --isolated --with "torch>=2.3" --with transformers==4.46.3 \
    --with rasterio --with pystac-client --with planetary-computer \
    --with matplotlib --with requests --with scipy \
    python scripts/level3_cv_detection.py zurich
"""

import json
import sys

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
import torch
import torch.nn as nn
from rasterio.merge import merge as merge_rasters
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from transformers import SegformerImageProcessor, ViTForImageClassification, ViTImageProcessor

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Physical scale the model expects (Google Maps zoom 19 at SF's latitude).
TARGET_MPP = 0.2359
NAIP_NATIVE_MPP = 0.6
TILE_PX = 256

DETECTION_WEIGHTS = "data/raw/deepsolar_weights/detection_pytorch_model.bin"
SEGMENTATION_WEIGHTS = "data/raw/deepsolar_weights/segformer_full.pt"


class LoRAConv1DWrapper(nn.Module):
    """Copied from DeepSolar-3M's inference notebook -- required to load their
    fine-tuned state dict, which has LoRA adapter weights on these layers."""

    def __init__(self, conv1dmodule, lora_rank):
        super().__init__()
        self.base_module = conv1dmodule
        in_features = self.base_module.weight.shape[0]
        out_features = self.base_module.weight.shape[1]
        self.lora_A = nn.Parameter(torch.empty((in_features, lora_rank)))
        self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank)))
        nn.init.zeros_(self.lora_B)
        nn.init.normal_(self.lora_A)

    def forward(self, x):
        Wx = self.base_module(x)
        BAt = self.lora_B @ self.lora_A.T
        BAtx = x @ BAt
        return Wx + BAtx


def load_detection_model():
    model = ViTForImageClassification.from_pretrained(
        "facebook/vit-mae-large", num_labels=2, ignore_mismatched_sizes=True
    )
    for m in model.vit.encoder.layer:
        m.output.dense = LoRAConv1DWrapper(m.output.dense, 4)
        m.intermediate.dense = LoRAConv1DWrapper(m.intermediate.dense, 4)
    model.load_state_dict(torch.load(DETECTION_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_segmentation_model():
    """Load the pickled SegformerForSemanticSegmentation object directly.

    This is a full pickled model (not a state dict), so it's tightly coupled
    to the exact transformers version DeepSolar-3M used (4.46.3) -- see
    CLAUDE.md's "Building-level detection detour" section for the version
    conflicts this caused.
    """
    model = torch.load(SEGMENTATION_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()
    return model


def segment_tile(tile_pixels, seg_model, image_processor):
    """Return a boolean (256, 256) mask of PV pixels for one tile."""
    from PIL import Image

    img = Image.fromarray(np.transpose(tile_pixels, (1, 2, 0)))
    pixel_values = image_processor(img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        outputs = seg_model(pixel_values=pixel_values)
    pred = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
    return (pred.cpu().numpy() == 1)


def get_smoke_test_bbox():
    """~300m box centered on a cluster of permit-verified addresses."""
    permits = json.load(open("data/raw/richmond_solar_permits.json"))
    lats = [p["location"]["coordinates"][1] for p in permits if p.get("location")]
    lons = [p["location"]["coordinates"][0] for p in permits if p.get("location")]

    # pick the address at index 0 as the center; ~150m padding in degrees
    center_lat, center_lon = lats[0], lons[0]
    dlat = 150 / 111_320  # meters -> degrees latitude
    dlon = 150 / (111_320 * np.cos(np.radians(center_lat)))
    bbox = [center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat]
    print(f"Smoke test centered on permit at {center_lat:.6f}, {center_lon:.6f}")
    print(f"bbox: {bbox}")
    return bbox, (center_lon, center_lat)


# SFPUC Sunset Reservoir North Basin solar farm: ~24,000 panels, ground-mounted
# on the reservoir cover, no tree cover, operating since 2010. Positive-control
# case -- if the model can't find PV here, resolution is the fundamental blocker,
# not occlusion or array size.
SUNSET_RESERVOIR = (-122.483605, 37.7500956)


def get_control_test_bbox():
    center_lon, center_lat = SUNSET_RESERVOIR
    radius_m = 400
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * np.cos(np.radians(center_lat)))
    bbox = [center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat]
    print(f"Control test centered on Sunset Reservoir solar farm at {center_lat:.6f}, {center_lon:.6f}")
    print(f"bbox: {bbox}")
    return bbox, (center_lon, center_lat)


def fetch_naip_chip(bbox):
    """Read the NAIP window covering bbox, resampled to TARGET_MPP."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["naip"], bbox=bbox, datetime="2020-01-01/2026-01-01")
    items = sorted(search.items(), key=lambda i: i.datetime, reverse=True)
    item = items[0]
    href = item.assets["image"].href
    print(f"Using NAIP item: {item.id} ({item.datetime.date()})")

    with rasterio.open(href) as src:
        window = from_bounds(*rasterio.warp.transform_bounds("EPSG:4326", src.crs, *bbox), src.transform)
        data = src.read([1, 2, 3], window=window)
        window_transform = src.window_transform(window)
        src_crs = src.crs

    scale = NAIP_NATIVE_MPP / TARGET_MPP
    new_height = int(data.shape[1] * scale)
    new_width = int(data.shape[2] * scale)

    dst_transform = window_transform * window_transform.scale(
        data.shape[2] / new_width, data.shape[1] / new_height
    )
    upsampled = np.zeros((3, new_height, new_width), dtype=data.dtype)
    for band in range(3):
        reproject(
            source=data[band],
            destination=upsampled[band],
            src_transform=window_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.bilinear,
        )
    return upsampled, dst_transform, src_crs


SWISSIMAGE_NATIVE_MPP = 0.1

# A dense residential district in Zurich (Wiedikon), comparable scale to the
# Richmond test area. Switzerland has very high per-capita rooftop solar
# adoption, so a broad scan is more likely to find real positives than
# betting on one pre-picked address -- unlike the SF Richmond test, this is
# not centered on a permit-verified system.
ZURICH_WIEDIKON = (8.5259, 47.3706)


def get_zurich_bbox():
    center_lon, center_lat = ZURICH_WIEDIKON
    radius_m = 300
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * np.cos(np.radians(center_lat)))
    bbox = [center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat]
    print(f"Zurich test centered on Wiedikon at {center_lat:.6f}, {center_lon:.6f}")
    print(f"bbox: {bbox}")
    return bbox, (center_lon, center_lat)


# Wiedikon turned out to be almost entirely near-flat commercial/industrial
# roofs -- not a useful stress test for the plane-fitting code, which needs
# genuine pitch variation to be worth anything. Altstetten (a former village
# annexed into the city, Kreis 9) was picked instead of guessing: queried
# OpenStreetMap's Overpass API for real generator:source=solar tags across
# the whole city. First pass just took the rawest point-density hotspot,
# which turned out to be ~90 individually-mapped panel nodes on ONE flat
# industrial/service roof (a false signal -- one enthusiastic OSM mapper,
# not neighborhood-wide adoption), landing the bbox mostly on a sports
# field and a highway. Redone properly: queried actual building footprints
# to find genuinely dense small-building residential blocks, then
# cross-referenced those against nearby solar-point density to find a
# cluster with both -- 95 distinct buildings and 91 nearby confirmed solar
# installations within 300m, a real residential grid, not a guess.
ZURICH_ALTSTETTEN = (8.4747, 47.4012)


def get_zurich_altstetten_bbox():
    center_lon, center_lat = ZURICH_ALTSTETTEN
    radius_m = 300
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * np.cos(np.radians(center_lat)))
    bbox = [center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat]
    print(f"Zurich test centered on Altstetten at {center_lat:.6f}, {center_lon:.6f}")
    print(f"bbox: {bbox}")
    return bbox, (center_lon, center_lat)


def fetch_swissimage_chip(bbox):
    """Read the SWISSIMAGE 10cm window covering bbox, resampled to TARGET_MPP.

    Mosaics all SWISSIMAGE tiles covering the bbox -- a bbox can straddle a
    tile boundary (SWISSIMAGE tiles are ~1km), and reading only one tile
    would silently truncate the fetch to whatever fraction of the bbox that
    tile happens to cover. fetch_dsm_chip() had this exact bug; see CLAUDE.md.
    Where tiles overlap, rasterio.merge keeps the first dataset's pixels, so
    sorting hrefs newest-first preserves the original "prefer most recent
    acquisition" intent.

    Attribution per swisstopo terms of use: (C) swisstopo.
    """
    resp = requests.get(
        "https://data.geo.admin.ch/api/stac/v1/search",
        params={"collections": "ch.swisstopo.swissimage-dop10", "bbox": ",".join(str(b) for b in bbox)},
    )
    resp.raise_for_status()
    items = resp.json()["features"]
    items.sort(key=lambda i: i["properties"]["datetime"], reverse=True)
    hrefs = [next(v["href"] for k, v in item["assets"].items() if k.endswith("_0.1_2056.tif")) for item in items]
    print(f"Mosaicking {len(hrefs)} SWISSIMAGE tile(s), newest: {items[0]['id']}")

    with rasterio.open(hrefs[0]) as src:
        src_crs = src.crs
    merge_bounds = rasterio.warp.transform_bounds("EPSG:4326", src_crs, *bbox)
    data, window_transform = merge_rasters(hrefs, bounds=merge_bounds, indexes=[1, 2, 3])

    scale = SWISSIMAGE_NATIVE_MPP / TARGET_MPP  # ~0.42 -- downsampling, not up
    new_height = int(data.shape[1] * scale)
    new_width = int(data.shape[2] * scale)

    dst_transform = window_transform * window_transform.scale(
        data.shape[2] / new_width, data.shape[1] / new_height
    )
    resampled = np.zeros((3, new_height, new_width), dtype=data.dtype)
    for band in range(3):
        reproject(
            source=data[band],
            destination=resampled[band],
            src_transform=window_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.bilinear,
        )
    return resampled, dst_transform, src_crs


def tile_chip(image, transform, crs, center_lonlat=None):
    """Slice a (3, H, W) array into TILE_PX x TILE_PX tiles with per-tile geo bounds.

    If center_lonlat is given, the tile grid is offset so one tile is
    centered as closely as possible on that point (rather than starting
    blindly from the top-left corner, which could split a target roof
    across a tile boundary).
    """
    _, h, w = image.shape

    y_offset = 0
    x_offset = 0
    if center_lonlat is not None:
        target_x, target_y = rasterio.warp.transform("EPSG:4326", crs, [center_lonlat[0]], [center_lonlat[1]])
        target_col, target_row = ~transform * (target_x[0], target_y[0])
        y_offset = int(target_row) % TILE_PX - TILE_PX // 2
        x_offset = int(target_col) % TILE_PX - TILE_PX // 2

    tiles = []
    for y0 in range(max(0, y_offset), h - TILE_PX, TILE_PX):
        for x0 in range(max(0, x_offset), w - TILE_PX, TILE_PX):
            tile = image[:, y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
            utm_x, utm_y = rasterio.transform.xy(transform, y0 + TILE_PX // 2, x0 + TILE_PX // 2)
            lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [utm_x], [utm_y])
            tiles.append({"pixels": tile, "center_lon": lon[0], "center_lat": lat[0], "y0": y0, "x0": x0})
    return tiles


def classify_tiles(tiles, model, feature_extractor):
    image_mean, image_std = feature_extractor.image_mean, feature_extractor.image_std
    results = []
    with torch.no_grad():
        for t in tiles:
            img = np.transpose(t["pixels"], (1, 2, 0))  # CHW -> HWC
            pixel_values = feature_extractor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)
            logits = model(pixel_values).logits
            probs = torch.softmax(logits, dim=1)[0]
            results.append(
                {
                    "center_lon": t["center_lon"],
                    "center_lat": t["center_lat"],
                    "y0": t["y0"],
                    "x0": t["x0"],
                    "p_pv": float(probs[1]),
                }
            )
    return results


def save_tile_grid(tiles, results, target_lonlat, output_path, target_label="target", title="NAIP tiles fed to DeepSolar-3M classifier"):
    """Save all tiles as a labeled grid PNG so we can SEE what the model saw,
    per CLAUDE.md's 'plot it, don't just trust the number' rule."""
    import matplotlib.pyplot as plt

    p_pv_by_pos = {(r["y0"], r["x0"]): r["p_pv"] for r in results}
    ys = sorted(set(t["y0"] for t in tiles))
    xs = sorted(set(t["x0"] for t in tiles))

    # which tile is physically closest to the known permit address
    def dist(t):
        return (t["center_lon"] - target_lonlat[0]) ** 2 + (t["center_lat"] - target_lonlat[1]) ** 2

    target_tile = min(tiles, key=dist)

    fig, axes = plt.subplots(len(ys), len(xs), figsize=(3 * len(xs), 3 * len(ys)))
    axes = np.atleast_2d(axes)
    for t in tiles:
        row = ys.index(t["y0"])
        col = xs.index(t["x0"])
        ax = axes[row, col]
        img = np.transpose(t["pixels"], (1, 2, 0))
        ax.imshow(img)
        p_pv = p_pv_by_pos[(t["y0"], t["x0"])]
        is_target = t["y0"] == target_tile["y0"] and t["x0"] == target_tile["x0"]
        title = f"p_pv={p_pv:.3f}" + (f" <- {target_label}" if is_target else "")
        ax.set_title(title, fontsize=8, color="red" if is_target else "black")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved tile grid to {output_path}")


def find_pv_arrays(tiles, results, transform, crs, threshold=0.2):
    """Run segmentation on tiles the classifier flagged as plausible PV,
    and return one record per tile with a real-world centroid for its
    detected panel pixels (empty mask -> centroid is None, classifier's
    threshold guess was likely wrong)."""
    print("Loading segmentation model...")
    seg_model = load_segmentation_model()
    seg_processor = SegformerImageProcessor(reduce_labels=False)

    tiles_by_pos = {(t["y0"], t["x0"]): t for t in tiles}
    arrays = []
    for r in results:
        if r["p_pv"] < threshold:
            continue
        tile = tiles_by_pos[(r["y0"], r["x0"])]
        mask = segment_tile(tile["pixels"], seg_model, seg_processor)
        pv_px = int(mask.sum())
        centroid_lon = centroid_lat = None
        if pv_px > 0:
            rows, cols = np.nonzero(mask)
            abs_row = tile["y0"] + rows.mean()
            abs_col = tile["x0"] + cols.mean()
            utm_x, utm_y = rasterio.transform.xy(transform, abs_row, abs_col)
            centroid_lon, centroid_lat = rasterio.warp.transform(crs, "EPSG:4326", [utm_x], [utm_y])
            centroid_lon, centroid_lat = centroid_lon[0], centroid_lat[0]
        arrays.append(
            {
                "p_pv": r["p_pv"],
                "mask": mask,
                "pv_pixel_count": pv_px,
                "pv_area_m2": pv_px * (TARGET_MPP**2),
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "y0": tile["y0"],
                "x0": tile["x0"],
                "pixels": tile["pixels"],
            }
        )
    return arrays


def save_segmentation_overlays(arrays, output_path):
    import matplotlib.pyplot as plt

    arrays_with_pv = [a for a in arrays if a["pv_pixel_count"] > 0]
    if not arrays_with_pv:
        print("No tiles had any segmented PV pixels -- skipping overlay plot.")
        return

    fig, axes = plt.subplots(1, len(arrays_with_pv), figsize=(4 * len(arrays_with_pv), 4.5))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arrays_with_pv):
        img = np.transpose(a["pixels"], (1, 2, 0))
        ax.imshow(img)
        overlay = np.zeros((*a["mask"].shape, 4))
        overlay[a["mask"]] = [1, 1, 0, 0.5]  # yellow, semi-transparent
        ax.imshow(overlay)
        ax.set_title(f"p_pv={a['p_pv']:.3f}\n{a['pv_area_m2']:.0f} m^2 panel area", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Segmented PV pixels (yellow overlay)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved segmentation overlay to {output_path}")


def fetch_dsm_chip(bbox):
    """Read the swissSURFACE3D DSM (0.5m native res) window covering bbox.
    No resampling -- we want the DSM at its own native resolution for plane
    fitting, not matched to the imagery's resampled scale."""
    resp = requests.get(
        "https://data.geo.admin.ch/api/stac/v1/search",
        params={"collections": "ch.swisstopo.swisssurface3d-raster", "bbox": ",".join(str(b) for b in bbox)},
    )
    resp.raise_for_status()
    items = resp.json()["features"]
    hrefs = []
    for item in items:
        href = next((v["href"] for k, v in item["assets"].items() if k.endswith(".tif")), None)
        if href:
            hrefs.append(href)
    print(f"DSM tiles covering bbox: {len(hrefs)}")

    # bbox commonly spans several DSM tiles (they're smaller than the
    # SWISSIMAGE tiles) -- mosaic all of them, then crop to bbox, so no
    # array footprint silently falls outside a single tile's window.
    with rasterio.open(hrefs[0]) as src:
        crs = src.crs
    merge_bounds = rasterio.warp.transform_bounds("EPSG:4326", crs, *bbox)
    mosaic, transform = merge_rasters(hrefs, bounds=merge_bounds)
    elevation = mosaic[0]  # single band
    return elevation, transform, crs


OUTLIER_RESIDUAL_M = 0.3  # a point this far from the fitted plane is treated as a stray LiDAR return, not roof
MAX_OUTLIER_ITERATIONS = 3
MAX_OUTLIER_FRACTION = 0.2  # refuse to discard more than this share of a cluster as "artifacts"


def fit_roof_plane(elevation, transform, footprint_mask=None):
    """Least-squares plane fit z = a*x + b*y + c over a DSM window, with
    iterative outlier rejection.

    footprint_mask, if given, is a boolean array the same shape as elevation
    that's True only for DSM pixels actually under the segmented PV
    footprint. Without it, every pixel in the (rectangular) window is used,
    which can include off-roof terrain the window's bounding box happens to
    cover -- e.g. a courtyard or street visible at the window's edges.

    A real roof surface is locally smooth; a single DSM return more than
    OUTLIER_RESIDUAL_M off the fitted plane is far more likely to be a
    sensor artifact (a stray return off an antenna, vent, wire, or bird)
    than an actual roof feature, so it's dropped and the plane refit --
    up to MAX_OUTLIER_ITERATIONS times, or until nothing more is dropped.

    But a handful of stray returns and "this cluster genuinely isn't one
    flat plane" (e.g. a rougher roof, or a segmentation mask that spans more
    than one physical roof) look the same point-by-point -- the only real
    signal is how MUCH gets rejected. So this stops and keeps the last
    accepted fit, rather than trimming further, if the next round would (a)
    strip more than MAX_OUTLIER_FRACTION of the cluster's original points,
    since a fit that only "works" by discarding a large chunk of real data
    isn't trustworthy, or (b) leave fewer than 9 points, since a plane fit
    on fewer than that isn't meaningful.

    Returns (tilt_deg, azimuth_deg, (a, b, c), used_mask). used_mask is the
    same shape as elevation, True only for the points in the final fit
    (footprint pixels minus rejected outliers) -- useful for showing exactly
    what was excluded and why.
    """
    rows, cols = np.indices(elevation.shape)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    xs, ys, zs = np.array(xs), np.array(ys), elevation.ravel()

    valid = np.isfinite(zs) & (zs > -1000)  # DSM nodata is often a large negative sentinel
    if footprint_mask is not None:
        valid &= footprint_mask.ravel()
    n_initial = valid.sum()

    for _ in range(MAX_OUTLIER_ITERATIONS):
        A = np.column_stack([xs[valid], ys[valid], np.ones(valid.sum())])
        (a, b, c), *_ = np.linalg.lstsq(A, zs[valid], rcond=None)
        residual = zs - (a * xs + b * ys + c)
        tightened = valid & (np.abs(residual) <= OUTLIER_RESIDUAL_M)
        n_would_reject = n_initial - tightened.sum()
        if tightened.sum() == valid.sum():
            break  # converged
        if n_would_reject > MAX_OUTLIER_FRACTION * n_initial or tightened.sum() < 9:
            break  # this isn't a few artifacts -- keep the pre-trim fit instead of over-trimming
        valid = tightened

    tilt_deg = np.degrees(np.arctan(np.hypot(a, b)))
    azimuth_deg = np.degrees(np.arctan2(-a, -b)) % 360
    return tilt_deg, azimuth_deg, (a, b, c), valid.reshape(elevation.shape)


def add_roof_geometry(arrays, image_transform, dsm_elevation, dsm_transform, image_crs, dsm_crs, margin_px=2):
    """For each detected array, fit a roof plane to the DSM pixels under its
    segmented footprint and attach tilt/azimuth. The footprint's DSM pixels
    are split into spatially-connected clusters first (see MIN_CLUSTER_PIXELS
    below) and each cluster gets its own plane fit -- see a["roof_planes"].

    Both SWISSIMAGE and the DSM are natively EPSG:2056, so this is a direct
    coordinate transform, not a reprojection.
    """
    from scipy import ndimage

    for a in arrays:
        if a["pv_pixel_count"] == 0:
            a["tilt_deg"] = a["azimuth_deg"] = None
            continue

        rows, cols = np.nonzero(a["mask"])
        abs_rows = a["y0"] + rows
        abs_cols = a["x0"] + cols

        # footprint pixel centers -> real-world (x, y) in EPSG:2056
        xs, ys = rasterio.transform.xy(image_transform, abs_rows, abs_cols)
        if image_crs != dsm_crs:
            xs, ys = rasterio.warp.transform(image_crs, dsm_crs, xs, ys)

        # real-world (x, y) -> DSM pixel indices, then a small bounding window
        dsm_rows, dsm_cols = rasterio.transform.rowcol(dsm_transform, xs, ys)
        dsm_rows, dsm_cols = np.array(dsm_rows), np.array(dsm_cols)
        r0, r1 = dsm_rows.min() - margin_px, dsm_rows.max() + margin_px + 1
        c0, c1 = dsm_cols.min() - margin_px, dsm_cols.max() + margin_px + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        window = dsm_elevation[r0:r1, c0:c1]
        window_transform = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), dsm_transform
        )

        # rasterize the segmentation footprint onto the DSM window: True only for
        # DSM pixels actually touched by a mask pixel, not just inside the window's
        # bounding box (which can include off-roof terrain at coarser DSM resolution)
        footprint_mask = np.zeros(window.shape, dtype=bool)
        local_rows, local_cols = dsm_rows - r0, dsm_cols - c0
        in_bounds = (
            (local_rows >= 0) & (local_rows < window.shape[0]) & (local_cols >= 0) & (local_cols < window.shape[1])
        )
        footprint_mask[local_rows[in_bounds], local_cols[in_bounds]] = True

        # a single detected array's footprint can span more than one physical
        # roof plane -- either genuinely (two roof faces at different pitches)
        # or spuriously (a disconnected false-positive blob elsewhere in the
        # tile, e.g. street markings the classifier mistook for panels). Fit
        # a separate plane per spatially-connected cluster of DSM pixels
        # rather than one plane across all of them.
        MIN_CLUSTER_PIXELS = 9
        labels, n_clusters = ndimage.label(footprint_mask, structure=np.ones((3, 3)))
        planes = []
        for label_id in range(1, n_clusters + 1):
            cluster_mask = labels == label_id
            if cluster_mask.sum() < MIN_CLUSTER_PIXELS:
                continue
            tilt_deg, azimuth_deg, plane_coeffs, fit_mask = fit_roof_plane(window, window_transform, cluster_mask)
            planes.append(
                {
                    "tilt_deg": tilt_deg,
                    "azimuth_deg": azimuth_deg,
                    "plane_coeffs": plane_coeffs,
                    "mask": cluster_mask,
                    "fit_mask": fit_mask,  # cluster pixels actually used, after outlier rejection
                    "n_dsm_pixels": int(fit_mask.sum()),
                    "n_outliers_rejected": int(cluster_mask.sum() - fit_mask.sum()),
                }
            )
        planes.sort(key=lambda p: -p["n_dsm_pixels"])

        if not planes:  # every cluster too small to fit a plane meaningfully
            a["tilt_deg"] = a["azimuth_deg"] = None
            continue

        # the largest cluster is treated as "the" roof plane for this array
        # (existing single-plane consumers), with the full cluster list kept
        # alongside for anything that wants to see all of them
        a["tilt_deg"] = planes[0]["tilt_deg"]
        a["azimuth_deg"] = planes[0]["azimuth_deg"]
        a["plane_coeffs"] = planes[0]["plane_coeffs"]
        a["roof_planes"] = planes
        a["dsm_window"] = window
        a["dsm_window_transform"] = window_transform
        a["dsm_footprint_mask"] = footprint_mask


def save_roof_geometry_windrose(arrays, output_path, location_label="Wiedikon, Zurich"):
    """Polar plot of fitted roof geometry: azimuth as compass angle, tilt as
    radius, marker size by panel area, color by classifier confidence.

    Azimuth is unreliable on near-flat roofs (small elevation noise flips the
    fitted downhill direction by a lot), so those points are marked with a
    dashed ring rather than silently plotted as if the angle were solid.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fitted = [a for a in arrays if a.get("tilt_deg") is not None]
    if not fitted:
        print("No arrays with fitted roof geometry -- skipping wind-rose plot.")
        return

    FLAT_TILT_THRESHOLD_DEG = 2.0  # below this, azimuth is more noise than signal
    blue_ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]  # palette steps 100/200/350/500/700
    cmap = mcolors.LinearSegmentedColormap.from_list("confidence_blue", blue_ramp)

    azimuth_deg = np.array([a["azimuth_deg"] for a in fitted])
    r = np.array([a["tilt_deg"] for a in fitted])
    area = np.array([a["pv_area_m2"] for a in fitted])
    p_pv = np.array([a["p_pv"] for a in fitted])
    is_flat = r < FLAT_TILT_THRESHOLD_DEG

    r_max = max(r.max() * 1.35, 10)
    r_label = r_max * 0.86  # ring where decluttered labels sit, outside all data points

    # Labels sharing near-identical azimuth (common near the center, where several
    # roofs happen to face the same way) would overlap if placed at their raw
    # angle. Push any label less than MIN_GAP_DEG from the previous one (going
    # around the circle) further along the ring -- a leader line still points
    # each label back to its true marker, so nothing is misrepresented, just
    # decluttered. This is circular, not linear: with enough points, naively
    # pushing forward from azimuth 0 can walk past 360 deg and collide with
    # wherever it started. So instead start the walk right after the single
    # largest natural gap between points -- the one place nothing needs to
    # wrap around -- and shrink the gap if there are too many points to fit
    # MIN_GAP_DEG apart at all.
    n = len(azimuth_deg)
    min_gap_deg = min(16, 340 / n)
    sorted_az = np.sort(azimuth_deg)
    circular_gaps = np.diff(np.concatenate([sorted_az, [sorted_az[0] + 360]]))
    start_idx = np.argmax(circular_gaps)
    walk_order = np.roll(np.argsort(azimuth_deg), -1 - start_idx)  # start just after the biggest gap

    # unwrap the RAW azimuths along the walk first (they cross the 360->0
    # boundary at most once, right where the walk started) -- keeping this
    # separate from the push step below matters: comparing each point's raw
    # azimuth to the previous point's already-pushed label position would
    # misfire (a raw angle can easily be "less than" an inflated label without
    # any real wraparound having happened), which is what produced the
    # original bug -- labels landing back on top of earlier, unrelated ones.
    unwrapped_raw = azimuth_deg[walk_order].astype(float).copy()
    wrap_at = np.where(np.diff(unwrapped_raw) < 0)[0]
    if wrap_at.size:
        unwrapped_raw[wrap_at[0] + 1 :] += 360

    label_walk = np.empty(n)
    label_walk[0] = unwrapped_raw[0]
    for i in range(1, n):
        label_walk[i] = max(unwrapped_raw[i], label_walk[i - 1] + min_gap_deg)

    label_azimuth_deg = np.empty(n)
    label_azimuth_deg[walk_order] = label_walk

    theta = np.radians(azimuth_deg)
    label_theta = np.radians(label_azimuth_deg)

    fig, ax = plt.subplots(figsize=(8.5, 7.8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_ylim(0, r_max)
    ax.set_ylabel("")
    ax.set_title(
        f"Fitted roof geometry -- {location_label}\n(azimuth = compass direction, radius = tilt, from swissSURFACE3D LiDAR)",
        pad=28,
    )

    marker_area_pts2 = area * 0.6  # scale factor chosen so the largest (1318 m^2) fits the axes
    sc = ax.scatter(
        theta, r, s=marker_area_pts2, c=p_pv, cmap=cmap, vmin=0.2, vmax=1.0,
        edgecolors="#0b0b0b", linewidths=0.75, alpha=0.9, zorder=3,
    )
    if is_flat.any():
        ax.scatter(
            theta[is_flat], r[is_flat], s=marker_area_pts2[is_flat] * 1.6,
            facecolors="none", edgecolors="#52514e", linewidths=1.25,
            linestyles="dashed", zorder=2,
        )

    for t, rad, lt, a in zip(theta, r, label_theta, fitted):
        ax.plot([t, lt], [rad, r_label], color="#c3c2b7", linewidth=0.75, zorder=1)
        ax.text(
            lt, r_label, f"{a['pv_area_m2']:.0f} m²" + (" *" if a["tilt_deg"] < FLAT_TILT_THRESHOLD_DEG else ""),
            fontsize=8, color="#0b0b0b", ha="center", va="center",
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1.5},
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.6, location="right")
    cbar.set_label("Classifier confidence p(PV)")

    size_handles = [
        plt.scatter([], [], s=v * 0.6, facecolors="#9ec5f4", edgecolors="#0b0b0b", linewidths=0.75)
        for v in [100, 500, 1300]
    ]
    fig.legend(
        size_handles, ["100 m²", "500 m²", "1,300 m²"], title="Panel area",
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False,
    )

    fig.text(
        0.5, -0.005,
        "* dashed ring = tilt < 2°, near-flat roof -- fitted azimuth is noise-dominated, don't read the exact angle",
        ha="center", fontsize=8, color="#52514e",
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved roof geometry wind-rose to {output_path}")


def save_building_detail(array, output_path):
    """One building, two views of the same footprint: the SWISSIMAGE tile
    with its segmented PV mask, next to the raw swissSURFACE3D DSM heights
    for that footprint and the plane fitted through them.

    Requires add_roof_geometry() to have run on this array first (needs
    dsm_window / dsm_window_transform / plane_coeffs).
    """
    import matplotlib.pyplot as plt

    if array.get("tilt_deg") is None:
        print("Selected array has no fitted roof geometry -- skipping detail plot.")
        return

    elevation = array["dsm_window"]
    transform = array["dsm_window_transform"]
    planes = array["roof_planes"]

    rows, cols = np.indices(elevation.shape)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    xs, ys = np.array(xs).reshape(elevation.shape), np.array(ys).reshape(elevation.shape)
    valid = np.isfinite(elevation) & (elevation > -1000)

    # recenter to small local offsets (real-world EPSG:2056 coords are ~2.6M/1.2M) and
    # express height relative to the footprint's own mean, so axis numbers are readable
    x0, y0, z0 = xs[valid].mean(), ys[valid].mean(), elevation[valid].mean()
    X, Y, Z = xs - x0, ys - y0, np.where(valid, elevation - z0, np.nan)

    # fixed categorical order (palette.md) -- identity of "which plane", not magnitude
    cluster_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    fig = plt.figure(figsize=(13, 6.5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.transpose(array["pixels"], (1, 2, 0)))
    overlay = np.zeros((*array["mask"].shape, 4))
    overlay[array["mask"]] = [1, 1, 0, 0.5]  # yellow, semi-transparent
    ax1.imshow(overlay)
    ax1.set_title(f"SWISSIMAGE tile + segmented PV mask\np(PV)={array['p_pv']:.3f}, area={array['pv_area_m2']:.0f} m²")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X, Y, Z, color="#c3c2b7", edgecolor="none", alpha=0.4)  # neutral terrain context
    any_outliers = False
    for i, plane in enumerate(planes):
        color = cluster_colors[i % len(cluster_colors)]
        a_coef, b_coef, c_coef = plane["plane_coeffs"]
        plane_Z = (a_coef * xs + b_coef * ys + c_coef) - z0
        used = plane["fit_mask"] & valid
        ax2.scatter(
            X[used], Y[used], Z[used], color=color, edgecolors="#0b0b0b", linewidths=0.3, s=20, depthshade=False,
            label=f"Plane {i + 1}: tilt={plane['tilt_deg']:.1f}°, az={plane['azimuth_deg']:.0f}° ({plane['n_dsm_pixels']} px)",
        )
        rejected = plane["mask"] & ~plane["fit_mask"] & valid
        if rejected.any():
            any_outliers = True
            ax2.scatter(
                X[rejected], Y[rejected], Z[rejected], color="#d03b3b", marker="x", s=30,
                linewidths=1.2, depthshade=False, zorder=5,
            )
        # only draw each fitted plane surface over the pixels actually used in its
        # fit (post outlier-rejection), not the whole window -- otherwise unrelated
        # planes and rejected outliers overplot each other
        plane_Z_masked = np.where(plane["fit_mask"], plane_Z, np.nan)
        ax2.plot_surface(X, Y, plane_Z_masked, color=color, alpha=0.3, edgecolor="none")
    if any_outliers:
        ax2.scatter([], [], [], color="#d03b3b", marker="x", label=f"Rejected as artifact (>{OUTLIER_RESIDUAL_M * 100:.0f}cm from fit)")
    ax2.legend(loc="upper left", fontsize=7, frameon=False)
    n_planes_note = f" -- {len(planes)} distinct roof planes found" if len(planes) > 1 else ""
    ax2.set_title(f"swissSURFACE3D DSM heights + fitted plane(s){n_planes_note}")
    ax2.set_xlabel("East-West (m)")
    ax2.set_ylabel("North-South (m)")
    ax2.set_zlabel("Height relative to footprint mean (m)")

    fig.suptitle(f"Building detail -- lat={array['centroid_lat']:.6f}, lon={array['centroid_lon']:.6f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved building detail to {output_path}")


def save_registration_check(array, image_transform, output_path, margin_m=3):
    """Sanity check that SWISSIMAGE and the swissSURFACE3D DSM are actually
    co-registered, rather than just assumed to be because both claim
    EPSG:2056. Takes the ONE segmentation footprint (the ground truth) and
    draws its outline on both datasets, in the same shared real-world
    coordinates -- if the two datasets agree on where things are, the same
    outline should trace the same physical edges (roofline, courtyard gap)
    in both panels. A consistent offset between the two would mean the
    datasets are misaligned; this makes that checkable by eye instead of
    assumed from the CRS tag.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if array.get("dsm_footprint_mask") is None:
        print("Selected array has no DSM footprint -- skipping registration check.")
        return

    tile_transform = rasterio.windows.transform(
        rasterio.windows.Window(array["x0"], array["y0"], TILE_PX, TILE_PX), image_transform
    )

    dsm_window = array["dsm_window"]
    dsm_transform = array["dsm_window_transform"]
    dsm_left, dsm_bottom, dsm_right, dsm_top = rasterio.transform.array_bounds(*dsm_window.shape, dsm_transform)
    crop_left, crop_right = dsm_left - margin_m, dsm_right + margin_m
    crop_bottom, crop_top = dsm_bottom - margin_m, dsm_top + margin_m

    # crop the RGB tile down to the same real-world extent as the DSM window,
    # so both panels show the same physical area at comparable scale
    rows, cols = rasterio.transform.rowcol(tile_transform, [crop_left, crop_right], [crop_top, crop_bottom])
    r0, r1 = sorted((max(int(rows[0]), 0), min(int(rows[1]), TILE_PX)))
    c0, c1 = sorted((max(int(cols[0]), 0), min(int(cols[1]), TILE_PX)))
    rgb_crop = np.transpose(array["pixels"][:, r0:r1, c0:c1], (1, 2, 0))
    mask_crop = array["mask"][r0:r1, c0:c1]
    rgb_crop_transform = rasterio.windows.transform(rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), tile_transform)
    rgb_bounds = rasterio.transform.array_bounds(r1 - r0, c1 - c0, rgb_crop_transform)

    # shared local origin so both panels plot in the identical coordinate frame --
    # this, not each panel's own data, is what makes the comparison meaningful
    ox, oy = crop_left, crop_bottom

    def to_extent(bounds):
        left, bottom, right, top = bounds
        return [left - ox, right - ox, bottom - oy, top - oy]

    blue_ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("elevation_blue", blue_ramp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(rgb_crop, extent=to_extent(rgb_bounds), origin="upper")
    ax1.contour(mask_crop.astype(float), levels=[0.5], extent=to_extent(rgb_bounds), origin="upper", colors="#eda100", linewidths=2)
    ax1.set_title("SWISSIMAGE orthophoto\n+ segmentation mask outline")
    ax1.set_xlabel("East-West (m)")
    ax1.set_ylabel("North-South (m)")

    im = ax2.imshow(
        dsm_window, cmap=cmap, extent=to_extent((dsm_left, dsm_bottom, dsm_right, dsm_top)), origin="upper"
    )
    ax2.contour(
        array["dsm_footprint_mask"].astype(float), levels=[0.5],
        extent=to_extent((dsm_left, dsm_bottom, dsm_right, dsm_top)), origin="upper", colors="#eda100", linewidths=2,
    )
    ax2.set_title("swissSURFACE3D DSM height\n+ SAME mask outline")
    ax2.set_xlabel("East-West (m)")
    fig.colorbar(im, ax=ax2, label="Height (m)", shrink=0.8)

    for ax in (ax1, ax2):
        ax.set_xlim(crop_left - ox, crop_right - ox)
        ax.set_ylim(crop_bottom - oy, crop_top - oy)

    fig.suptitle("Registration check -- same footprint outline, both datasets, shared coordinates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved registration check to {output_path}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    print(f"Device: {DEVICE}, mode: {mode}")

    ZURICH_MODES = {"zurich", "altstetten"}

    if mode == "control":
        bbox, target_lonlat = get_control_test_bbox()
        output_path = "data/level3_cv_control_test_tiles.png"
        target_label = "reservoir center"
    elif mode == "zurich":
        bbox, target_lonlat = get_zurich_bbox()
        output_path = "data/level3_cv_zurich_test_tiles.png"
        target_label = "bbox center"
    elif mode == "altstetten":
        bbox, target_lonlat = get_zurich_altstetten_bbox()
        output_path = "data/level3_cv_altstetten_test_tiles.png"
        target_label = "bbox center"
    else:
        bbox, target_lonlat = get_smoke_test_bbox()
        output_path = "data/level3_cv_smoke_test_tiles.png"
        target_label = "permit addr"

    if mode in ZURICH_MODES:
        image, transform, crs = fetch_swissimage_chip(bbox)
        title = "SWISSIMAGE tiles fed to DeepSolar-3M classifier (downsampled 0.1m/px -> 0.24m/px)"
    else:
        image, transform, crs = fetch_naip_chip(bbox)
        title = "NAIP tiles fed to DeepSolar-3M classifier (upsampled 0.6m/px -> 0.24m/px)"
    print(f"Fetched + resampled chip: {image.shape}, crs={crs}")

    tiles = tile_chip(image, transform, crs, center_lonlat=target_lonlat)
    print(f"Tiled into {len(tiles)} tiles of {TILE_PX}x{TILE_PX}px")

    print("Loading detection model...")
    model = load_detection_model()
    feature_extractor = ViTImageProcessor.from_pretrained("facebook/vit-mae-large")

    print("Running classification...")
    results = classify_tiles(tiles, model, feature_extractor)

    save_tile_grid(tiles, results, target_lonlat, output_path, target_label=target_label, title=title)

    results.sort(key=lambda r: -r["p_pv"])
    print("\nTop 10 tiles by P(PV):")
    for r in results[:10]:
        print(f"  p_pv={r['p_pv']:.3f}  lat={r['center_lat']:.6f}  lon={r['center_lon']:.6f}")

    if mode in ZURICH_MODES:
        print("\nRunning segmentation on plausible-PV tiles...")
        arrays = find_pv_arrays(tiles, results, transform, crs, threshold=0.2)
        save_segmentation_overlays(arrays, f"data/level3_cv_{mode}_segmentation.png")

        print("\nFetching swissSURFACE3D DSM for roof-plane fitting...")
        dsm_elevation, dsm_transform, dsm_crs = fetch_dsm_chip(bbox)
        add_roof_geometry(arrays, transform, dsm_elevation, dsm_transform, crs, dsm_crs)

        print("\nDetected PV arrays -- location, size, and fitted roof geometry:")
        for a in arrays:
            if a["centroid_lon"] is None:
                print(f"  p_pv={a['p_pv']:.3f}  no PV pixels segmented (classifier likely wrong)")
            elif a["tilt_deg"] is None:
                print(
                    f"  p_pv={a['p_pv']:.3f}  area={a['pv_area_m2']:.0f}m^2  "
                    f"lat={a['centroid_lat']:.6f}  lon={a['centroid_lon']:.6f}  (footprint too small to fit a plane)"
                )
            else:
                print(
                    f"  p_pv={a['p_pv']:.3f}  area={a['pv_area_m2']:.0f}m^2  "
                    f"lat={a['centroid_lat']:.6f}  lon={a['centroid_lon']:.6f}  "
                    f"tilt={a['tilt_deg']:.1f}deg  azimuth={a['azimuth_deg']:.0f}deg"
                )
                if len(a["roof_planes"]) > 1 or a["roof_planes"][0]["n_outliers_rejected"] > 0:
                    for i, plane in enumerate(a["roof_planes"]):
                        outlier_note = f"  ({plane['n_outliers_rejected']} outlier px rejected)" if plane["n_outliers_rejected"] else ""
                        print(
                            f"      plane {i + 1}: tilt={plane['tilt_deg']:.1f}deg  "
                            f"azimuth={plane['azimuth_deg']:.0f}deg  n_dsm_px={plane['n_dsm_pixels']}{outlier_note}"
                        )

        location_label = "Altstetten, Zurich" if mode == "altstetten" else "Wiedikon, Zurich"
        save_roof_geometry_windrose(arrays, f"data/level3_cv_{mode}_roof_geometry.png", location_label)

        fitted = [a for a in arrays if a.get("tilt_deg") is not None]
        if fitted:
            best = max(fitted, key=lambda a: a["p_pv"])
            save_building_detail(best, f"data/level3_cv_{mode}_building_detail.png")
            save_registration_check(best, transform, f"data/level3_cv_{mode}_registration_check.png")


if __name__ == "__main__":
    main()
