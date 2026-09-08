"""Roof geometry from segmentation masks and elevation rasters.

Split out of level3_cv_detection.py so this half can be tested without the
model stack: everything here needs only numpy, scipy and rasterio, where the
pipeline needs torch and transformers as well. That matters because this is
where the silent-wrongness bugs live -- a sign error in an orientation, a
misaligned grid, a plane fit over the wrong pixels -- and none of them crash.
They return smooth, plausible, wrong numbers, so they are found by tests and
by plotting, which have to be cheap to run.

Nothing in here fetches, downloads, or plots. Tests: scripts/test_roof_geometry.py
"""

import numpy as np
import rasterio


# Physical scale the model expects (Google Maps zoom 19 at SF's latitude).
TARGET_MPP = 0.2359


def combine_tile_confidence(pixel_confidences):
    """Collapse a stitched blob's per-pixel classifier confidences into one
    number for the whole array.

    A blob that spans several tile positions inherits a different p_pv from
    each. `pixel_confidences` is a 1-D array with one entry per blob pixel:
    the mean p_pv of the tile positions that segmented that pixel (with
    non-overlapping stride there is exactly one, so this is just "the tile it
    came from", as before).

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


# Panel pixels a few metres apart belong to the same installation, not to two.
# The segmenter drops out across walkways, shadow and its own uncertainty, and
# on the Wiedikon test roof that split one array into three detections whose
# nearest mask pixels were 1.5, 4.3 and 7.2 m apart. Closing the canvas
# (dilate, then erode by the same amount) bridges gaps up to roughly this
# radius before the connected-component pass decides what "one array" means.
#
# 12 px = 2.8 m, taken off a measured plateau rather than picked by eye:
# sweeping the radius over both test areas, the detection count is stable from
# 9 px through 16 px in BOTH (Wiedikon 16 arrays, Altstetten 8), and only
# starts merging further at 21 px in Altstetten and 30 px in Wiedikon. 12 sits
# in the middle of the range where the two areas agree.
#
# CRITICAL: the closed canvas decides GROUPING ONLY. Every returned mask is
# still the original segmented pixels. If the bridged gaps counted as panels,
# every area -- and so every capacity estimate built on one -- would be
# inflated by the walkways between rows. The invariant is that total panel
# area is IDENTICAL at every radius including zero; it held exactly (3021.8 m^2
# Wiedikon, 6778.9 m^2 Altstetten) across 0-30 px when this was chosen.
GAP_BRIDGE_PX = 12


def bridge_mask_gaps(canvas, radius_px):
    """Morphological closing of the panel canvas: dilate by radius_px, then
    erode by the same. Small gaps get filled; the outer boundary comes back
    to roughly where it started.

    Used only to decide which pixels belong to the same array -- never
    returned as a mask. See GAP_BRIDGE_PX.
    """
    from scipy import ndimage

    if radius_px <= 0:
        return canvas
    # pad first, or the erosion pass eats the mask back at the array border
    pad = radius_px + 1
    padded = np.pad(canvas, pad)
    grown = ndimage.binary_dilation(padded, structure=np.ones((3, 3)), iterations=radius_px)
    shrunk = ndimage.binary_erosion(grown, structure=np.ones((3, 3)), iterations=radius_px)
    return shrunk[pad:-pad, pad:-pad]


# A standard crystalline PV module is about 1.7 x 1.0 m. Snapping a footprint
# to the SHORT edge lets the grid represent modules laid either way round at
# ~1 m granularity; snapping to 1.7 m would be coarser and would shave real
# edges. See sweep_regularization_cell() -- this is checked against a plateau,
# not asserted.
MODULE_SHORT_EDGE_M = 1.0


# Blur applied before measuring edge direction. See dominant_orientation_deg().
ORIENTATION_SMOOTH_PX = 2.0


def dominant_orientation_deg(mask):
    """The angle, 0-90 deg, of the mask boundary's most common edge direction.

    A solar array is rectilinear: panels are rectangles in rows on rectangular
    racks, so its outline is dominated by two edge directions 90 deg apart.
    Folding the histogram mod 90 collapses those two into one peak.

    Voting over every boundary pixel, weighted by gradient magnitude, rather
    than taking the minimum-area bounding rectangle: the bounding rectangle is
    decided by whichever few pixels stick out furthest, so one stray blob of
    segmentation noise rotates the whole answer. A vote is decided by the bulk
    of the outline and barely notices the same blob.

    The returned angle is in ndimage.rotate()'s sense: rotating the mask by
    -theta brings its dominant edges onto the axes.

    Returns 0.0 for a mask with no usable boundary (empty, or a single pixel).
    """
    from scipy import ndimage

    # MUST smooth before differencing. np.gradient on a BINARY mask uses a
    # two-pixel stencil on 0/1 data, so the only directions it can produce are
    # 0 and 45 deg -- no rotated rectangle is recoverable, and the function
    # returns a plausible-looking wrong answer rather than failing. Blurring
    # first makes the boundary a ramp and the directions continuous.
    # sigma is a compromise: 1.0 leaves ~7 deg of error, 3.0 is slightly more
    # accurate but smears a small footprint (a 10 m^2 detection is barely 13 px
    # across). Small footprints have unreliable orientation either way.
    smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=ORIENTATION_SMOOTH_PX)
    gy, gx = np.gradient(smoothed)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() <= 0:
        return 0.0
    # arctan2 gives the direction of steepest ascent, perpendicular to the
    # edge -- but perpendicular is +90 deg, which mod 90 is the same bin, so
    # no correction is needed for that. The NEGATION is needed: array rows
    # increase downwards, so this frame is left-handed relative to the one
    # ndimage.rotate() uses, and an un-negated angle would make
    # regularize_footprint() rotate the wrong way and double the misalignment.
    angle = -np.degrees(np.arctan2(gy, gx)) % 90.0
    hist = np.zeros(90)
    np.add.at(hist, angle.astype(int) % 90, magnitude)
    # 'wrap' because 89 deg and 0 deg are adjacent directions, not opposite ends
    hist = ndimage.gaussian_filter1d(hist, sigma=1.5, mode="wrap")
    return float(np.argmax(hist))


def _pad_square(mask):
    """Centre a mask in a square big enough that rotation cannot clip a corner."""
    h, w = mask.shape
    side = int(np.ceil(np.hypot(h, w))) + 2
    pad_y, pad_x = (side - h) // 2, (side - w) // 2
    out = np.zeros((side, side), dtype=bool)
    out[pad_y : pad_y + h, pad_x : pad_x + w] = mask
    return out, (pad_y, pad_x, h, w)


def regularize_footprint(mask, mpp=TARGET_MPP, cell_m=MODULE_SHORT_EDGE_M):
    """Snap a blobby segmentation mask to a rectilinear footprint aligned with
    the array's own dominant orientation and quantised to the module grid.

    Three steps, each boring on its own:
      1. find the dominant edge direction and rotate the mask onto the axes
      2. lay a cell_m grid over it and keep every cell at least half covered
      3. rotate back

    Step 2 is what makes the result rectilinear: a union of grid cells has
    right angles by construction, so there is no contour simplification and no
    vertex-angle snapping to tune. "At least half covered" is the same
    inclusive-on-ties rule as combine_views(), for the same reason -- for an
    aggregate capacity total, dropping real panel area is the worse error.

    IMPORTANT -- this is not automatically an improvement, and is deliberately
    NOT wired into the pipeline. Every pixel it adds beyond the true panel edge
    is a pixel fit_roof_plane() will then fit a roof plane through. The
    documented leverage bias means a handful of contaminating roof-edge DSM
    pixels swings a small footprint's tilt by degrees, so a "tidier" footprint
    can buy a worse geometry. Validate against measured tilt before adopting.

    Returns (regularized_mask, theta_deg).
    """
    from scipy import ndimage

    if not mask.any():
        return mask.copy(), 0.0

    theta = dominant_orientation_deg(mask)
    square, (pad_y, pad_x, h, w) = _pad_square(mask)

    # order=0 keeps the array boolean; reshape=False keeps the round trip's
    # shapes identical so the result can be cropped straight back
    rotated = ndimage.rotate(square.astype(np.uint8), -theta, reshape=False, order=0) > 0

    cell = max(1, int(round(cell_m / mpp)))
    coarse = np.zeros_like(rotated)
    ys, xs = np.nonzero(rotated)
    if len(ys):
        # Anchor the grid on the mask's own top-left corner, NOT on the canvas
        # origin. Otherwise the array's edges fall at an arbitrary offset into
        # a cell, every edge cell is part-covered, and the >= 0.5 rule inflates
        # a clean rectangle by ~13% -- growth that would go straight into
        # fit_roof_plane(). Anchoring also makes the result translation-
        # invariant: the same array shifted two pixels regularizes the same.
        r0, c0 = int(ys.min()), int(xs.min())
        sub = rotated[r0 : int(ys.max()) + 1, c0 : int(xs.max()) + 1]
        nh = int(np.ceil(sub.shape[0] / cell))
        nw = int(np.ceil(sub.shape[1] / cell))
        # Anchored at the mask's corner, NOT centred on it. Centring was tried
        # (to avoid the grid always padding towards the bottom-right) and is
        # worse: it makes the grid origin depend on the mask's SIZE as well as
        # its corner, so a one-pixel change in the bounding box repartitions
        # every cell. Measured, centring took re-application IoU from 0.93 to
        # 0.87 and area stability from 0.1% to 10%. Corner anchoring costs a
        # ~1.4 px displacement on re-application, which is real but bounded;
        # centring costs stability, which is not.
        region = np.zeros((nh * cell, nw * cell), dtype=bool)
        region[: sub.shape[0], : sub.shape[1]] = sub
        keep = region.reshape(nh, cell, nw, cell).mean(axis=(1, 3)) >= 0.5
        grid = np.repeat(np.repeat(keep, cell, axis=0), cell, axis=1)
        hh = min(grid.shape[0], coarse.shape[0] - r0)
        ww = min(grid.shape[1], coarse.shape[1] - c0)
        coarse[r0 : r0 + hh, c0 : c0 + ww] = grid[:hh, :ww]

    back = ndimage.rotate(coarse.astype(np.uint8), theta, reshape=False, order=0) > 0
    return back[pad_y : pad_y + h, pad_x : pad_x + w], theta


def footprint_change(original, regularized, mpp=TARGET_MPP):
    """What regularization did, in the terms that matter downstream.

    `added_m2` is the number to watch: those are pixels no view called panel,
    which fit_roof_plane() would nonetheless fit a plane through.
    """
    inter = int((original & regularized).sum())
    union = int((original | regularized).sum())
    return {
        "iou": inter / union if union else 1.0,
        "added_m2": int((regularized & ~original).sum()) * mpp**2,
        "removed_m2": int((original & ~regularized).sum()) * mpp**2,
        "area_before_m2": int(original.sum()) * mpp**2,
        "area_after_m2": int(regularized.sum()) * mpp**2,
    }


def combine_views(votes, views):
    """Decide, per pixel, whether the overlapping segmentation views call it panel.

    With TILE_STRIDE_PX < TILE_PX a pixel is segmented up to 4 times, once per
    overlapping tile position, and those views can disagree. `votes` counts how
    many called it panel; `views` counts how many looked at it.

    Majority vote. The alternatives and why not:

    - UNION (panel if any view says so) can only ever GROW the area, and every
      view's hallucinations compound. Given this classifier's known false
      positives -- the highest-confidence Wiedikon detection is four dormer
      skylights -- a rule that cannot subtract is the wrong rule.
    - AVERAGED PROBABILITY would be cleanest statistically, but segment_tile()
      returns a hard argmax, and switching to soft output introduces a new
      threshold with no ground truth anywhere in Zurich to set it against.
      Majority vote adds no tunable parameter at all.

    The denominator is views that ACTUALLY RAN the segmenter, not every view
    that geometrically covers the pixel. An unflagged tile is the classifier's
    judgement about a whole 60 m square; counting it as a "not panel" vote for
    each of its 65k pixels over-reads a tile-level score, and would make the
    result depend on the classification threshold twice over.

    Ties go to panel (2 of 4 counts as panel): for an aggregate MW total,
    missed capacity is the bias that actually matters, so where the views are
    evenly split, err toward including it.

    At TILE_STRIDE_PX == TILE_PX every segmented pixel has views == 1, so this
    reduces to votes >= 1 -- exactly the boolean OR canvas it replaces. The
    change is a generalisation of the old behaviour, not a substitute for it,
    and running at stride 256 should reproduce the previous run exactly.
    """
    return (views > 0) & (votes * 2 >= views)


def report_view_agreement(votes, views, canvas):
    """How often do overlapping views of the same pixel agree?

    This is the real prize in overlapping inference, and it is free once the
    tiles overlap. Until now every mask this pipeline produced was a single
    unrepeatable measurement -- there was no way to ask how stable the
    segmenter is, only whether its output looked right. Four views per pixel
    turns that into a number.

    Read it as: if agreement is high, the masks are reproducible and the known
    under-coverage of large arrays is a genuine model limitation. If agreement
    is low, the segmenter is substantially noisy and every area figure in this
    branch carries an error bar nobody has ever quoted.
    """
    px_m2 = TARGET_MPP**2
    n_seg = int((views > 0).sum())
    n_over = int((views >= 2).sum())
    if n_over == 0:
        print("  every pixel was segmented by at most one view -- no agreement to measure "
              "(set TILE_STRIDE_PX < TILE_PX)")
        return

    print(f"  {n_over / n_seg:.0%} of segmented pixels were seen by 2+ views "
          f"(mean {views[views > 0].mean():.2f} views/px)")

    claimed = (views >= 2) & (votes >= 1)  # at least one overlapping view said panel
    n = int(claimed.sum())
    unanimous = int((claimed & (votes == views)).sum())
    kept = int((claimed & canvas).sum())
    print(f"  of {n} contested px ({n * px_m2:.0f} m^2) called panel by >=1 of 2+ views:")
    print(f"    {unanimous / n:.1%} unanimous, {kept / n:.1%} survived the majority vote, "
          f"{(n - kept) / n:.1%} outvoted and dropped")

    union_m2 = int((votes >= 1).sum()) * px_m2
    major_m2 = int(canvas.sum()) * px_m2
    print(f"  estimator choice is worth {union_m2 - major_m2:.0f} m^2 here: "
          f"union would give {union_m2:.0f} m^2, majority vote gives {major_m2:.0f} m^2")


OUTLIER_RESIDUAL_M = 0.3  # a point this far from the fitted plane is treated as a stray LiDAR return, not roof


MAX_OUTLIER_ITERATIONS = 3


MAX_OUTLIER_FRACTION = 0.2  # refuse to discard more than this share of a cluster as "artifacts"


# --- plausibility gates -------------------------------------------------
#
# A rooftop array sits metres above the surrounding terrain; a parking deck,
# a courtyard or a forecourt sits on it. DSM (the surface, buildings included)
# minus DTM (bare terrain) is that height. Measured over Wiedikon on
# 2026-09-06: every detection on a real roof came out 10-38 m up, and the two
# confirmed false positives came out at 0.01 m and -0.03 m. That is not a
# marginal threshold call, which is why 2 m is a safe place to draw it.
#
# But the HEADROOM is a property of the neighbourhood, not of the method.
# Altstetten's lowest real detection sits at 2.7 m -- industrial sheds and
# treatment-plant structures are low buildings -- so the margin there is 0.7 m,
# not Wiedikon's 8.9 m. In a genuinely low-rise area (carports, single-storey
# garages, bungalows) this threshold would start clipping real rooftop arrays.
# Check the height distribution before trusting this gate in a new city.
MIN_HEIGHT_ABOVE_GROUND_M = 2.0


# Above this a "roof" is a wall. Real roof pitches top out near 50 deg and
# rack tilts near 40, so 60 leaves room for genuinely steep alpine roofs.
MAX_PLAUSIBLE_TILT_DEG = 60.0


def apply_plausibility_gates(arrays):
    """Flag detections that cannot be what the pipeline says they are.

    NOTHING IS DELETED. Both gates have real exceptions -- ground-mounted PV
    is genuinely at grade (SF's Sunset Reservoir farm, this project's original
    positive control, is ground-mounted on a reservoir cover), and
    facade-mounted PV is genuinely near-vertical. A flag lets a downstream
    consumer decide; a silent drop would quietly delete real capacity, which
    is the one direction an aggregate MW total cannot afford to be wrong in.
    """
    for a in arrays:
        h = a.get("height_above_ground_m")
        tilt = a.get("tilt_deg")
        a["at_grade"] = h is not None and h < MIN_HEIGHT_ABOVE_GROUND_M
        a["implausible_tilt"] = tilt is not None and tilt > MAX_PLAUSIBLE_TILT_DEG
        a["no_height"] = h is None
        a["plausible"] = not (a["at_grade"] or a["implausible_tilt"])

    at_grade = [a for a in arrays if a["at_grade"]]
    steep = [a for a in arrays if a["implausible_tilt"]]
    unknown = [a for a in arrays if a["no_height"]]
    kept = [a for a in arrays if a["plausible"]]

    print(f"\nPlausibility gates ({len(arrays)} detections in):")
    print(f"  at grade (< {MIN_HEIGHT_ABOVE_GROUND_M:.0f} m above bare terrain): {len(at_grade)}")
    for a in at_grade:
        print(f"      {a['pv_area_m2']:.0f} m^2  p_pv={a['p_pv']:.3f}  "
              f"height={a['height_above_ground_m']:+.2f} m  at ({a['centroid_lat']:.5f},{a['centroid_lon']:.5f})")
    print(f"  tilt > {MAX_PLAUSIBLE_TILT_DEG:.0f} deg (a wall, not a roof): {len(steep)}")
    for a in steep:
        print(f"      {a['pv_area_m2']:.0f} m^2  p_pv={a['p_pv']:.3f}  tilt={a['tilt_deg']:.1f} deg")
    if unknown:
        print(f"  height could not be measured (no DSM/DTM coverage): {len(unknown)}")
    area_all = sum(a["pv_area_m2"] for a in arrays)
    area_kept = sum(a["pv_area_m2"] for a in kept)
    print(f"  -> {len(kept)} plausible, {len(arrays) - len(kept)} flagged "
          f"({area_kept:.0f} of {area_all:.0f} m^2 kept, {100 * (1 - area_kept / area_all):.1f}% flagged out)")
    print("  (flagged, not deleted -- ground-mounted and facade PV are real; see apply_plausibility_gates)")
    return kept


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
