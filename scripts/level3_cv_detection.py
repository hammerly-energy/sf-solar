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


MIN_BLOB_PX = 30  # ~1.7 m^2 at TARGET_MPP -- roughly one solar module


def combine_tile_confidence(pixel_confidences):
    """Collapse a stitched blob's per-pixel classifier confidences into one
    number for the whole array.

    A blob that spans several tiles inherits a different p_pv from each tile
    it touches. `pixel_confidences` is a 1-D array with one entry per blob
    pixel: the p_pv of whichever tile that pixel was segmented in.

    Area-weighted mean, which is what taking the mean over PIXELS already
    gives us for free -- a blob with 3000 pixels in a confident tile and 200
    in a doubtful one is pulled mostly by the confident tile, in proportion
    to how much of the array actually sits there.

    Not max(): "one tile was sure, so the whole array counts as sure" is
    exactly the failure mode we already have evidence for -- the
    highest-scoring Wiedikon detection (p=0.952) turned out to be four dormer
    skylights, while the two largest unambiguous real arrays scored 0.348 and
    0.260. Nothing about this classifier earns the benefit of the doubt.
    Not min() either: that would punish a large, genuine array for extending
    a few pixels into one unconvincing tile.

    NOTE: the result is a summary statistic, not a calibrated probability.
    The model outputs p(PV) for a whole 60m tile; averaging several of those
    does not produce p(PV) for one roof. Read it as a relative ranking
    between arrays, and keep a["tile_p_pv"] if you need the raw values.
    """
    return float(np.mean(pixel_confidences))


def _pad_slice(sl, shape, pad=1):
    """Grow a 2-D bounding-box slice by `pad` pixels, clipped to the array."""
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, dim)) for s, dim in zip(sl, shape))


def find_pv_arrays(image, tiles, results, transform, crs, threshold=0.2):
    """Segment every tile the classifier flagged, stitch the tile-local masks
    back into ONE full-image canvas, and return one record per spatially-
    connected blob of panel pixels -- not one record per tile.

    The 256x256 tile exists only because the models were trained on that
    input size. It is an input constraint, not a unit of measurement. Before
    this stitched, a roof crossing a tile seam became two or three separate
    "arrays", each with its own area, centroid and plane fit: on the
    Altstetten test area 12 of 13 detections were fragments like that, and
    fragments of one surface disagreed by up to 179 degrees in azimuth.

    Each returned record uses the blob's own bounding box as its frame:
      mask   -- boolean, blob-local (the bounding box of this blob only)
      y0/x0  -- top-left of that box in full-image pixel coords
      pixels -- the RGB crop of the full image over the same box
    which is the same (mask, y0, x0, pixels) contract the tile-based records
    had, so everything downstream keeps working unchanged -- it now just
    describes one contiguous surface instead of one square of image.
    """
    from scipy import ndimage

    print("Loading segmentation model...")
    seg_model = load_segmentation_model()
    seg_processor = SegformerImageProcessor(reduce_labels=False)

    tiles_by_pos = {(t["y0"], t["x0"]): t for t in tiles}
    flagged = [r for r in results if r["p_pv"] >= threshold]

    _, img_h, img_w = image.shape
    canvas = np.zeros((img_h, img_w), dtype=bool)
    # which pixels the segmenter actually looked at -- a blob that runs up
    # against an unsegmented tile may have been cut short at that seam
    segmented = np.zeros((img_h, img_w), dtype=bool)
    # which flagged tile each pixel came from, so a blob can be traced back
    # to the tiles (and the confidences) that produced it
    tile_index = np.full((img_h, img_w), -1, dtype=np.int32)

    n_empty = 0
    for i, r in enumerate(flagged):
        tile = tiles_by_pos[(r["y0"], r["x0"])]
        mask = segment_tile(tile["pixels"], seg_model, seg_processor)
        ys = slice(tile["y0"], tile["y0"] + TILE_PX)
        xs = slice(tile["x0"], tile["x0"] + TILE_PX)
        canvas[ys, xs] |= mask
        segmented[ys, xs] = True
        tile_index[ys, xs] = i
        if not mask.any():
            n_empty += 1
    print(
        f"Segmented {len(flagged)} flagged tiles "
        f"({n_empty} produced no PV pixels at all -- classifier likely wrong there)"
    )
    tile_conf = np.array([r["p_pv"] for r in flagged], dtype=float)

    # ONE connected-component pass over the whole canvas. Panel pixels either
    # side of a seam are literally adjacent here, so they join into one blob
    # and the tile boundary stops existing.
    labels, n_blobs = ndimage.label(canvas, structure=np.ones((3, 3)))

    arrays = []
    n_dropped = 0
    n_pieces = 0
    for blob_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = labels[sl] == blob_id
        n_px = int(sub.sum())
        if n_px < MIN_BLOB_PX:
            n_dropped += 1
            continue

        y0, x0 = sl[0].start, sl[1].start
        rows, cols = np.nonzero(sub)
        utm_x, utm_y = rasterio.transform.xy(transform, y0 + rows.mean(), x0 + cols.mean())
        centroid_lon, centroid_lat = rasterio.warp.transform(crs, "EPSG:4326", [utm_x], [utm_y])

        pixel_tiles = tile_index[sl][sub]
        contributing = np.unique(pixel_tiles)
        n_pieces += len(contributing)

        # did this blob stop at the edge of the segmented area? if so it may
        # be a real array cut short, not a complete one -- overlapping
        # inference (stride < TILE_PX) is what would fix that, not stitching
        psl = _pad_slice(sl, canvas.shape)
        grown = ndimage.binary_dilation(labels[psl] == blob_id, structure=np.ones((3, 3)))
        truncated = bool((grown & ~segmented[psl]).any())

        arrays.append(
            {
                "p_pv": combine_tile_confidence(tile_conf[pixel_tiles]),
                "tile_p_pv": tile_conf[contributing],
                "n_tiles": len(contributing),
                "possibly_truncated": truncated,
                "mask": sub,
                "pixels": image[:, sl[0], sl[1]],
                "pv_pixel_count": n_px,
                "pv_area_m2": n_px * (TARGET_MPP**2),
                "centroid_lon": centroid_lon[0],
                "centroid_lat": centroid_lat[0],
                "y0": y0,
                "x0": x0,
            }
        )

    arrays.sort(key=lambda a: -a["pv_area_m2"])
    n_multi = sum(1 for a in arrays if a["n_tiles"] > 1)
    n_trunc = sum(1 for a in arrays if a["possibly_truncated"])
    print(f"Stitched into {len(arrays)} arrays ({n_dropped} blobs dropped as smaller than ~1 module)")
    print(f"  assembled from {n_pieces} tile-level pieces; {n_multi} arrays span more than one tile")
    print(f"  {n_trunc} run up against a tile that was never segmented -- possibly cut short at a seam")
    return arrays


def save_detection_cache(arrays, image_transform, image_crs, output_path):
    """Dump every detection (segmentation mask + geometry) to a .npz.

    The full pipeline takes ~10 minutes, mostly model inference. Anything that
    wants to re-examine the detections afterwards -- a visual audit, a
    threshold sweep, a different plot -- should read this cache instead of
    re-running inference. Masks are blob-local; y0/x0 place each blob's
    bounding box in the full image, and image_transform maps full-image
    pixels to CRS coords.
    """
    payload = {
        "transform": np.array(image_transform.to_gdal(), dtype=float),
        "crs": np.array(str(image_crs)),
        "n_arrays": np.array(len(arrays)),
    }
    for i, a in enumerate(arrays):
        payload[f"mask_{i}"] = a["mask"].astype(np.uint8)
        payload[f"meta_{i}"] = np.array(
            [
                a["p_pv"],
                a["y0"],
                a["x0"],
                a["pv_area_m2"],
                a["centroid_lat"] if a["centroid_lat"] is not None else np.nan,
                a["centroid_lon"] if a["centroid_lon"] is not None else np.nan,
                a.get("tilt_deg") if a.get("tilt_deg") is not None else np.nan,
                a.get("azimuth_deg") if a.get("azimuth_deg") is not None else np.nan,
                a["n_tiles"],
                float(a["possibly_truncated"]),
            ],
            dtype=float,
        )
    np.savez_compressed(output_path, **payload)
    print(f"Saved detection cache ({len(arrays)} arrays) to {output_path}")


MAX_OVERLAY_PANELS = 12


def save_segmentation_overlays(arrays, image, image_transform, output_path):
    """One panel per detected array, largest first, each cropped to that
    array plus a margin of surrounding roof -- so panels differ in size and
    shape (they are arrays now, not uniform tiles) but you can still see what
    the mask stopped short of."""
    import matplotlib.pyplot as plt

    if not arrays:
        print("No arrays detected -- skipping overlay plot.")
        return

    shown = arrays[:MAX_OVERLAY_PANELS]  # arrays arrive sorted largest-first
    ncols = min(4, len(shown))
    nrows = int(np.ceil(len(shown) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, a in zip(axes, shown):
        rgb, mask_crop, _ = crop_around_array(
            a, image, image_transform, array_bounds_with_margin(a, image_transform)
        )
        ax.imshow(rgb)
        # outline, not a fill -- a translucent fill hides exactly the roof
        # texture you need to tell a PV module from a skylight
        h, w = mask_crop.shape
        draw_mask_outline(ax, mask_crop, (-0.5, w - 0.5, h - 0.5, -0.5), colors="#eb6834", linewidths=1.6)
        seam_note = f", {a['n_tiles']} tiles" if a["n_tiles"] > 1 else ""
        ax.set_title(f"p_pv={a['p_pv']:.3f}{seam_note}\n{a['pv_area_m2']:.0f} m^2 panel area", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(shown):]:
        ax.axis("off")

    omitted = f" -- showing the {len(shown)} largest of {len(arrays)}" if len(arrays) > len(shown) else ""
    fig.suptitle(f"Segmented PV pixels (yellow overlay){omitted}")
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
    # Every tile exists in BOTH a 2018 and a 2024 acquisition, and the API
    # returns 2018 first. rasterio.merge keeps the FIRST dataset's pixels
    # where tiles overlap, so without this sort we'd silently fit planes to
    # 2018 elevation under 2025 imagery -- any array installed since 2018
    # would be fitted to the bare roof as it was before the panels existed.
    items.sort(key=lambda i: i["properties"]["datetime"], reverse=True)
    hrefs = []
    for item in items:
        href = next((v["href"] for k, v in item["assets"].items() if k.endswith(".tif")), None)
        if href:
            hrefs.append(href)
    years = sorted({i["properties"]["datetime"][:4] for i in items}, reverse=True)
    print(f"DSM tiles covering bbox: {len(hrefs)} (acquisitions available: {', '.join(years)}; using newest first)")

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


def fit_group_planes(window, window_transform, group_mask, min_pixels):
    """Split one spatially-connected group of DSM pixels into multiple plane
    facets, where the group doesn't sit on a single plane -- e.g. several
    individually-tilted rack-mounted rows that touch each other in the mask.

    Sequential-plane-peeling, the same idea as RANSAC-style multi-plane
    segmentation: fit one plane (fit_roof_plane already does its own
    within-plane outlier rejection), keep it, then look at whatever's left
    over. If that leftover still contains a spatially-connected group of at
    least min_pixels, it's not noise -- it's probably a whole separate
    facet -- so queue it up and fit it too. Repeats until nothing sizeable
    is left.

    Returns a list of facet dicts (tilt_deg, azimuth_deg, plane_coeffs,
    fit_mask, n_dsm_pixels), largest first. Only the primary (largest)
    facet carries n_outliers_rejected / outlier_mask -- points that ended up
    in a sibling facet aren't outliers, they're just a different plane, and
    only points nothing ever claimed count as genuine unexplained leftovers.
    """
    from scipy import ndimage

    facets = []
    queue = [group_mask]
    while queue:
        candidate = queue.pop(0)
        if candidate.sum() < min_pixels:
            continue
        tilt_deg, azimuth_deg, plane_coeffs, fit_mask = fit_roof_plane(window, window_transform, candidate)
        if fit_mask.sum() < min_pixels:
            continue
        facets.append(
            {"tilt_deg": tilt_deg, "azimuth_deg": azimuth_deg, "plane_coeffs": plane_coeffs, "fit_mask": fit_mask}
        )
        leftover = candidate & ~fit_mask
        if leftover.sum() >= min_pixels:
            sub_labels, n_sub = ndimage.label(leftover, structure=np.ones((3, 3)))
            for label_id in range(1, n_sub + 1):
                sub_mask = sub_labels == label_id
                if sub_mask.sum() >= min_pixels:
                    queue.append(sub_mask)

    if not facets:
        return []

    claimed = np.zeros_like(group_mask)
    for f in facets:
        claimed |= f["fit_mask"]
    outlier_mask = group_mask & ~claimed

    facets.sort(key=lambda f: -f["fit_mask"].sum())
    for i, f in enumerate(facets):
        f["n_dsm_pixels"] = int(f["fit_mask"].sum())
        f["outlier_mask"] = outlier_mask if i == 0 else np.zeros_like(group_mask)
        f["n_outliers_rejected"] = int(outlier_mask.sum()) if i == 0 else 0
    return facets


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
        # roof plane -- either genuinely (two roof faces at different pitches,
        # or several individually-tilted rack-mounted rows sitting close
        # together) or spuriously (a disconnected false-positive blob
        # elsewhere in the tile, e.g. street markings the classifier mistook
        # for panels). Connected-component labeling alone only catches the
        # spurious case -- it splits by adjacency, not by whether pixels
        # actually sit on the same plane, so several touching tilted rows
        # still come out as one blob. fit_group_planes() (below) handles
        # that: within each spatially-connected group, it keeps peeling off
        # the best-fitting plane and re-clustering whatever's left, so e.g.
        # four adjacent tilted panel rows become four separate plane fits
        # instead of one meaningless average.
        MIN_CLUSTER_PIXELS = 9
        labels, n_clusters = ndimage.label(footprint_mask, structure=np.ones((3, 3)))
        planes = []
        for label_id in range(1, n_clusters + 1):
            cluster_mask = labels == label_id
            if cluster_mask.sum() < MIN_CLUSTER_PIXELS:
                continue
            planes.extend(fit_group_planes(window, window_transform, cluster_mask, MIN_CLUSTER_PIXELS))
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


# How much roof to show around a detected array. A blob's bounding box stops
# at its last panel pixel, so without a margin every plot is cropped flush to
# the array and there is nothing left to judge it against. Proportional alone
# is not enough -- 25% of a 5 m^2 blob is about a metre -- hence the floor.
CONTEXT_MARGIN_FRAC = 0.35
CONTEXT_MARGIN_MIN_M = 10.0


def array_bounds_with_margin(array, image_transform):
    """The array's bounding box in CRS coords, grown by a visible margin.
    Returns (left, bottom, right, top)."""
    h, w = array["mask"].shape
    left, top = image_transform * (array["x0"], array["y0"])
    right, bottom = image_transform * (array["x0"] + w, array["y0"] + h)
    pad = max(CONTEXT_MARGIN_FRAC * max(right - left, top - bottom), CONTEXT_MARGIN_MIN_M)
    return left - pad, bottom - pad, right + pad, top + pad


def draw_mask_outline(ax, mask, extent, **kwargs):
    """Draw a boolean mask's boundary as a CLOSED outline.

    matplotlib's contour() will not close a contour that runs off the edge of
    the array it is given -- it has no data beyond the edge to place the line
    against. A blob's bounding box has mask pixels touching all four edges by
    definition, so contouring one directly leaves the outline hanging open
    wherever the array reaches its own edge. Padding a ring of False around
    the mask, and widening the extent by that same one pixel so the geometry
    still lines up with the image underneath, gives the contour somewhere to
    close. `extent` is matplotlib's (left, right, bottom, top).
    """
    left, right, bottom, top = extent
    px = (right - left) / mask.shape[1]
    py = (top - bottom) / mask.shape[0]
    ax.contour(
        np.pad(mask, 1).astype(float), levels=[0.5],
        extent=(left - px, right + px, bottom - py, top + py),
        origin="upper", **kwargs,
    )


def crop_around_array(array, image, image_transform, bounds):
    """RGB crop of the FULL image over `bounds` (left, bottom, right, top in
    the image CRS), with this array's blob mask pasted into the same frame.

    Needed because an array's bounding box is now tight around its own panel
    pixels -- any context around it (the rest of the roof, the neighbouring
    building) has to be re-cut from the full image, not read off the record.
    Returns (rgb HWC, mask, bounds actually covered).
    """
    left, bottom, right, top = bounds
    _, img_h, img_w = image.shape
    rows, cols = rasterio.transform.rowcol(image_transform, [left, right], [top, bottom])
    r0, r1 = sorted((max(int(rows[0]), 0), min(int(rows[1]), img_h)))
    c0, c1 = sorted((max(int(cols[0]), 0), min(int(cols[1]), img_w)))
    rgb = np.transpose(image[:, r0:r1, c0:c1], (1, 2, 0))

    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mh, mw = array["mask"].shape
    my0, mx0 = array["y0"] - r0, array["x0"] - c0  # blob box, in crop-local coords
    sy0, sx0 = max(my0, 0), max(mx0, 0)
    sy1, sx1 = min(my0 + mh, r1 - r0), min(mx0 + mw, c1 - c0)
    if sy1 > sy0 and sx1 > sx0:
        mask[sy0:sy1, sx0:sx1] = array["mask"][sy0 - my0 : sy1 - my0, sx0 - mx0 : sx1 - mx0]

    crop_transform = rasterio.windows.transform(
        rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), image_transform
    )
    return rgb, mask, rasterio.transform.array_bounds(r1 - r0, c1 - c0, crop_transform)


def save_building_detail(array, image, image_transform, output_path):
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

    rgb, mask_crop, _ = crop_around_array(
        array, image, image_transform, array_bounds_with_margin(array, image_transform)
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb)
    overlay = np.zeros((*mask_crop.shape, 4))
    overlay[mask_crop] = [1, 1, 0, 0.5]  # yellow, semi-transparent
    ax1.imshow(overlay)
    seam_note = f" (stitched from {array['n_tiles']} tiles)" if array["n_tiles"] > 1 else ""
    ax1.set_title(
        f"SWISSIMAGE + segmented PV mask{seam_note}\n"
        f"p(PV)={array['p_pv']:.3f}, area={array['pv_area_m2']:.0f} m²"
    )
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
        rejected = plane["outlier_mask"] & valid
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


def save_registration_check(array, image, image_transform, output_path, margin_m=3):
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


    dsm_window = array["dsm_window"]
    dsm_transform = array["dsm_window_transform"]
    dsm_left, dsm_bottom, dsm_right, dsm_top = rasterio.transform.array_bounds(*dsm_window.shape, dsm_transform)
    crop_left, crop_right = dsm_left - margin_m, dsm_right + margin_m
    crop_bottom, crop_top = dsm_bottom - margin_m, dsm_top + margin_m

    # cut the RGB to the same real-world extent as the DSM window, so both
    # panels show the same physical area at comparable scale
    rgb_crop, mask_crop, rgb_bounds = crop_around_array(
        array, image, image_transform, (crop_left, crop_bottom, crop_right, crop_top)
    )

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
    draw_mask_outline(ax1, mask_crop, to_extent(rgb_bounds), colors="#eda100", linewidths=2)
    ax1.set_title("SWISSIMAGE orthophoto\n+ segmentation mask outline")
    ax1.set_xlabel("East-West (m)")
    ax1.set_ylabel("North-South (m)")

    im = ax2.imshow(
        dsm_window, cmap=cmap, extent=to_extent((dsm_left, dsm_bottom, dsm_right, dsm_top)), origin="upper"
    )
    draw_mask_outline(
        ax2, array["dsm_footprint_mask"], to_extent((dsm_left, dsm_bottom, dsm_right, dsm_top)),
        colors="#eda100", linewidths=2,
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


def save_outlier_check(array, image, image_transform, output_path, margin_m=3):
    """Show exactly where fit_roof_plane()'s rejected outlier points sit, on
    both the orthophoto and the raw DSM, in the same shared real-world
    coordinates -- so it's checkable by eye whether a rejected point
    corresponds to something real and visible (a chimney, vent, antenna,
    overhanging tree) or looks like plain sensor noise.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    planes = array.get("roof_planes")
    if not planes:
        print("Selected array has no fitted roof planes -- skipping outlier check.")
        return

    dsm_window = array["dsm_window"]
    dsm_transform = array["dsm_window_transform"]
    rejected_mask = np.zeros(dsm_window.shape, dtype=bool)
    for plane in planes:
        rejected_mask |= plane["outlier_mask"]
    n_rejected = int(rejected_mask.sum())
    if n_rejected == 0:
        print("Selected array had zero rejected outliers -- skipping outlier check.")
        return


    dsm_left, dsm_bottom, dsm_right, dsm_top = rasterio.transform.array_bounds(*dsm_window.shape, dsm_transform)
    crop_left, crop_right = dsm_left - margin_m, dsm_right + margin_m
    crop_bottom, crop_top = dsm_bottom - margin_m, dsm_top + margin_m

    rgb_crop, mask_crop, rgb_bounds = crop_around_array(
        array, image, image_transform, (crop_left, crop_bottom, crop_right, crop_top)
    )

    ox, oy = crop_left, crop_bottom

    def to_extent(bounds):
        left, bottom, right, top = bounds
        return [left - ox, right - ox, bottom - oy, top - oy]

    rej_rows, rej_cols = np.nonzero(rejected_mask)
    rej_xs, rej_ys = rasterio.transform.xy(dsm_transform, rej_rows, rej_cols)
    rej_x_local = np.array(rej_xs) - ox
    rej_y_local = np.array(rej_ys) - oy

    blue_ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("elevation_blue", blue_ramp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(rgb_crop, extent=to_extent(rgb_bounds), origin="upper")
    draw_mask_outline(ax1, mask_crop, to_extent(rgb_bounds), colors="#eda100", linewidths=1.5)
    ax1.scatter(rej_x_local, rej_y_local, color="#d03b3b", marker="x", s=60, linewidths=2, zorder=5, label="Rejected outlier")
    ax1.set_title("SWISSIMAGE orthophoto\n+ rejected outlier locations")
    ax1.set_xlabel("East-West (m)")
    ax1.set_ylabel("North-South (m)")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)

    im = ax2.imshow(
        dsm_window, cmap=cmap, extent=to_extent((dsm_left, dsm_bottom, dsm_right, dsm_top)), origin="upper"
    )
    draw_mask_outline(
        ax2, array["dsm_footprint_mask"], to_extent((dsm_left, dsm_bottom, dsm_right, dsm_top)),
        colors="#eda100", linewidths=1.5,
    )
    ax2.scatter(rej_x_local, rej_y_local, facecolors="none", edgecolors="#d03b3b", marker="o", s=100, linewidths=2, zorder=5)
    ax2.set_title(f"swissSURFACE3D DSM height\n+ {n_rejected} rejected outlier px (>{OUTLIER_RESIDUAL_M * 100:.0f}cm from fit)")
    ax2.set_xlabel("East-West (m)")
    fig.colorbar(im, ax=ax2, label="Height (m)", shrink=0.8)

    for ax in (ax1, ax2):
        ax.set_xlim(crop_left - ox, crop_right - ox)
        ax.set_ylim(crop_bottom - oy, crop_top - oy)

    fig.suptitle("Rejected outliers -- same points, orthophoto + LiDAR height, shared coordinates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved outlier check to {output_path}")


def _shifted_overlap(a, b, dr, dc):
    """The overlapping slices of a and b when b is shifted by (dr, dc)
    relative to a (positive dr = b moved down, positive dc = b moved
    right). No wraparound -- pixels that fall off the edge are just
    dropped from the comparison, unlike np.roll."""
    h, w = a.shape
    ar0, ar1 = max(0, dr), min(h, h + dr)
    ac0, ac1 = max(0, dc), min(w, w + dc)
    br0, br1 = max(0, -dr), min(h, h - dr)
    bc0, bc1 = max(0, -dc), min(w, w - dc)
    return a[ar0:ar1, ac0:ac1], b[br0:br1, bc0:bc1]


def _normalized_cross_correlation(a, b):
    a, b = a.ravel(), b.ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def check_registration_offset(arrays, max_shift_px=4):
    """Quantitative version of save_registration_check(): rather than
    eyeballing one building, test whether SWISSIMAGE-derived masks and the
    DSM are CONSISTENTLY offset from each other across every detected array.

    For each array, computes two independent "edge strength" maps over the
    same DSM pixel grid -- gradient magnitude of the DSM heights (real roof
    edges are height jumps) and gradient magnitude of the rasterized mask
    (the segmentation footprint's own boundary) -- then finds the pixel
    shift that maximizes their normalized cross-correlation (NCC). NCC is
    bounded in [-1, 1] and measures whether the two edge PATTERNS line up,
    not just their magnitude, so unlike a raw gradient-magnitude search it
    has a genuine interior peak instead of drifting toward whatever noisy
    terrain happens to have more texture.

    If most arrays agree on roughly the same non-zero shift, that's
    evidence of a real, fixable spatial offset between the two datasets. If
    the best shift is scattered across arrays (or clusters at zero), or the
    correlation is weak everywhere, there's no reliable single offset to
    correct for.
    """
    results = []
    for a in arrays:
        footprint_mask = a.get("dsm_footprint_mask")
        dsm_window = a.get("dsm_window")
        if footprint_mask is None or dsm_window is None or footprint_mask.sum() < 12:
            continue

        dgy, dgx = np.gradient(dsm_window)
        dsm_edge = np.hypot(dgx, dgy)
        mgy, mgx = np.gradient(footprint_mask.astype(float))
        mask_edge = np.hypot(mgx, mgy)
        if mask_edge.sum() == 0:
            continue

        best_shift, best_ncc, baseline_ncc = (0, 0), -1.0, None
        for dr in range(-max_shift_px, max_shift_px + 1):
            for dc in range(-max_shift_px, max_shift_px + 1):
                d_ov, m_ov = _shifted_overlap(dsm_edge, mask_edge, dr, dc)
                if d_ov.size < 20:  # too little overlap left to mean anything
                    continue
                ncc = _normalized_cross_correlation(d_ov, m_ov)
                if dr == 0 and dc == 0:
                    baseline_ncc = ncc
                if ncc > best_ncc:
                    best_ncc, best_shift = ncc, (dr, dc)

        if baseline_ncc is None:
            continue
        results.append(
            {
                "label": f"{a['pv_area_m2']:.0f}m^2 @ ({a['centroid_lat']:.5f},{a['centroid_lon']:.5f})",
                "best_shift_px": best_shift,
                "best_ncc": best_ncc,
                "baseline_ncc": baseline_ncc,
            }
        )

    print(f"\nRegistration offset check across {len(results)} arrays (NCC, shift in DSM pixels, 0.5m each):")
    for r in results:
        flag = "" if r["best_shift_px"] == (0, 0) else "  <-- non-zero"
        print(
            f"  {r['label']:55s} best_shift={r['best_shift_px']}  "
            f"NCC={r['best_ncc']:.2f} (vs {r['baseline_ncc']:.2f} at zero shift){flag}"
        )

    # a weak correlation even at its best (loose threshold: 0.15) means this array's
    # edges are too noisy/ambiguous for the shift estimate to be trustworthy at all
    confident = [r for r in results if r["best_ncc"] >= 0.15]
    print(f"\n{len(confident)}/{len(results)} arrays had a strong enough edge match (NCC >= 0.15) to trust their shift estimate.")
    if confident:
        shift_counts = {}
        for r in confident:
            shift_counts[r["best_shift_px"]] = shift_counts.get(r["best_shift_px"], 0) + 1
        mode_shift, mode_count = max(shift_counts.items(), key=lambda kv: kv[1])
        print(
            f"Most common best-shift among those: {mode_shift} in {mode_count}/{len(confident)}. "
            + (
                "Zero (or near-zero) shift dominates -- no evidence of a consistent spatial offset."
                if mode_shift == (0, 0)
                else "A non-zero shift dominates -- this looks like a real registration offset worth correcting."
                if mode_count / len(confident) > 0.5
                else "No shift dominates even among the confident results -- still no reliable evidence of a fixed offset."
            )
        )


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
        arrays = find_pv_arrays(image, tiles, results, transform, crs, threshold=0.2)
        save_segmentation_overlays(arrays, image, transform, f"data/level3_cv_{mode}_segmentation.png")

        print("\nFetching swissSURFACE3D DSM for roof-plane fitting...")
        dsm_elevation, dsm_transform, dsm_crs = fetch_dsm_chip(bbox)
        add_roof_geometry(arrays, transform, dsm_elevation, dsm_transform, crs, dsm_crs)

        print("\nDetected PV arrays -- location, size, and fitted roof geometry:")
        for a in arrays:
            notes = []
            if a["n_tiles"] > 1:
                notes.append(f"stitched from {a['n_tiles']} tiles")
            if a["possibly_truncated"]:
                notes.append("may be cut short at a seam")
            note = f"  [{'; '.join(notes)}]" if notes else ""
            if a["tilt_deg"] is None:
                print(
                    f"  p_pv={a['p_pv']:.3f}  area={a['pv_area_m2']:.0f}m^2  "
                    f"lat={a['centroid_lat']:.6f}  lon={a['centroid_lon']:.6f}  "
                    f"(footprint too small to fit a plane){note}"
                )
            else:
                print(
                    f"  p_pv={a['p_pv']:.3f}  area={a['pv_area_m2']:.0f}m^2  "
                    f"lat={a['centroid_lat']:.6f}  lon={a['centroid_lon']:.6f}  "
                    f"tilt={a['tilt_deg']:.1f}deg  azimuth={a['azimuth_deg']:.0f}deg{note}"
                )
                if len(a["roof_planes"]) > 1 or a["roof_planes"][0]["n_outliers_rejected"] > 0:
                    for i, plane in enumerate(a["roof_planes"]):
                        outlier_note = f"  ({plane['n_outliers_rejected']} outlier px rejected)" if plane["n_outliers_rejected"] else ""
                        print(
                            f"      plane {i + 1}: tilt={plane['tilt_deg']:.1f}deg  "
                            f"azimuth={plane['azimuth_deg']:.0f}deg  n_dsm_px={plane['n_dsm_pixels']}{outlier_note}"
                        )

        save_detection_cache(arrays, transform, crs, f"data/level3_cv_{mode}_detections.npz")

        location_label = "Altstetten, Zurich" if mode == "altstetten" else "Wiedikon, Zurich"
        save_roof_geometry_windrose(arrays, f"data/level3_cv_{mode}_roof_geometry.png", location_label)

        fitted = [a for a in arrays if a.get("tilt_deg") is not None]
        if fitted:
            best = max(fitted, key=lambda a: a["p_pv"])
            save_building_detail(best, image, transform, f"data/level3_cv_{mode}_building_detail.png")
            save_registration_check(best, image, transform, f"data/level3_cv_{mode}_registration_check.png")

            most_outliers = max(fitted, key=lambda a: sum(p["n_outliers_rejected"] for p in a["roof_planes"]))
            save_outlier_check(most_outliers, image, transform, f"data/level3_cv_{mode}_outlier_check.png")

            check_registration_offset(fitted)


if __name__ == "__main__":
    main()
