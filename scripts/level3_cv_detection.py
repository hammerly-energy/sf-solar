"""
EXPERIMENTAL -- Level 3 side quest, not part of the main pipeline.
Full context: CLAUDE.md, "Building-level detection detour" section.

Detect rooftop solar panels using DeepSolar-3M's published model (Prabha et
al., ICLR 2025 Climate Change AI workshop), then fit each detected array's
roof plane (tilt/azimuth) from LiDAR. Three stages: classify (ViT-MAE-large
+ LoRA) -> segment (SegFormer) -> fit a plane to the segmented footprint's
DSM pixels.

Three modes (first CLI arg):
  zurich  -- Wiedikon, Zurich. WORKING. SWISSIMAGE (10cm/px) + swissSURFACE3D
             DSM (0.5m/px), both openly licensed. Classifier clears its own
             0.5 decision threshold here; full pipeline runs end-to-end.
             Known bug: fetch_dsm_chip() only uses the first of several DSM
             tiles covering the bbox instead of mosaicking -- footprints
             outside that one tile wrongly report "too small to fit".
  smoke   -- SF Richmond (94118/94121), centered on a permit-verified
             address. KNOWN BROKEN: NAIP is only 0.6m/px vs the ~0.24m/px
             (Google Maps zoom>=19) the model expects; every tile scored
             near zero even at a real positive location.
  control -- SF Sunset Reservoir solar farm (positive control, ~24,000
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
    --with matplotlib --with requests \
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


def fetch_swissimage_chip(bbox):
    """Read the SWISSIMAGE 10cm window covering bbox, resampled to TARGET_MPP.

    Attribution per swisstopo terms of use: (C) swisstopo.
    """
    resp = requests.get(
        "https://data.geo.admin.ch/api/stac/v1/search",
        params={"collections": "ch.swisstopo.swissimage-dop10", "bbox": ",".join(str(b) for b in bbox)},
    )
    resp.raise_for_status()
    items = resp.json()["features"]
    # prefer the most recent acquisition covering this bbox
    items.sort(key=lambda i: i["properties"]["datetime"], reverse=True)
    item = items[0]
    href = next(v["href"] for k, v in item["assets"].items() if k.endswith("_0.1_2056.tif"))
    print(f"Using SWISSIMAGE item: {item['id']}")

    with rasterio.open(href) as src:
        window = from_bounds(*rasterio.warp.transform_bounds("EPSG:4326", src.crs, *bbox), src.transform)
        data = src.read([1, 2, 3], window=window)
        window_transform = src.window_transform(window)
        src_crs = src.crs

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

    # bbox typically spans one DSM tile for our test areas; if it spans more,
    # just use the first -- good enough for a single-array plane fit.
    with rasterio.open(hrefs[0]) as src:
        window = from_bounds(*rasterio.warp.transform_bounds("EPSG:4326", src.crs, *bbox), src.transform)
        elevation = src.read(1, window=window)
        transform = src.window_transform(window)
        crs = src.crs
    return elevation, transform, crs


def fit_roof_plane(elevation, transform):
    """Least-squares plane fit z = a*x + b*y + c over a DSM window.

    Returns (tilt_deg, azimuth_deg). Azimuth is the compass direction the
    roof slopes DOWN toward (its aspect) -- same convention as pvlib's
    surface_azimuth (0=N, 90=E, 180=S, 270=W), matching the convention
    confirmed against the CA DG Stats data key in level3_fleet.py.
    """
    rows, cols = np.indices(elevation.shape)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    xs, ys, zs = np.array(xs), np.array(ys), elevation.ravel()

    valid = np.isfinite(zs) & (zs > -1000)  # DSM nodata is often a large negative sentinel
    A = np.column_stack([xs[valid], ys[valid], np.ones(valid.sum())])
    (a, b, c), *_ = np.linalg.lstsq(A, zs[valid], rcond=None)

    tilt_deg = np.degrees(np.arctan(np.hypot(a, b)))
    azimuth_deg = np.degrees(np.arctan2(-a, -b)) % 360
    return tilt_deg, azimuth_deg


def add_roof_geometry(arrays, image_transform, dsm_elevation, dsm_transform, image_crs, dsm_crs, margin_px=2):
    """For each detected array, fit a roof plane to the DSM pixels under its
    segmented footprint and attach tilt/azimuth.

    Both SWISSIMAGE and the DSM are natively EPSG:2056, so this is a direct
    coordinate transform, not a reprojection.
    """
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
        r0, r1 = min(dsm_rows) - margin_px, max(dsm_rows) + margin_px + 1
        c0, c1 = min(dsm_cols) - margin_px, max(dsm_cols) + margin_px + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        window = dsm_elevation[r0:r1, c0:c1]
        window_transform = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), dsm_transform
        )

        if window.size < 9:  # too small a footprint to fit a plane meaningfully
            a["tilt_deg"] = a["azimuth_deg"] = None
            continue

        a["tilt_deg"], a["azimuth_deg"] = fit_roof_plane(window, window_transform)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    print(f"Device: {DEVICE}, mode: {mode}")

    if mode == "control":
        bbox, target_lonlat = get_control_test_bbox()
        output_path = "data/level3_cv_control_test_tiles.png"
        target_label = "reservoir center"
    elif mode == "zurich":
        bbox, target_lonlat = get_zurich_bbox()
        output_path = "data/level3_cv_zurich_test_tiles.png"
        target_label = "bbox center"
    else:
        bbox, target_lonlat = get_smoke_test_bbox()
        output_path = "data/level3_cv_smoke_test_tiles.png"
        target_label = "permit addr"

    if mode == "zurich":
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

    if mode == "zurich":
        print("\nRunning segmentation on plausible-PV tiles...")
        arrays = find_pv_arrays(tiles, results, transform, crs, threshold=0.2)
        save_segmentation_overlays(arrays, "data/level3_cv_zurich_segmentation.png")

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


if __name__ == "__main__":
    main()
