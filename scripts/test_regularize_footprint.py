"""Unit tests for footprint regularization, on synthetic shapes with known answers.

Run standalone (same isolated env as the pipeline -- level3_cv_detection.py
imports torch at module level, so importing two numpy functions out of it pulls
in the whole stack; not worth restructuring the flat script over):
    uv run --isolated --with "torch>=2.3" --with transformers==4.46.3 --with rasterio \
      --with pystac-client --with planetary-computer --with matplotlib --with requests \
      --with scipy python scripts/test_regularize_footprint.py

Every shape here is built by hand, so the right answer is known before the
model sees it. That is the point: on the real detections there is no ground
truth, so anything the model does there is unfalsifiable. Half of these tests
assert what regularization does NOT do -- the boundaries of the method matter
more than its successes, because a footprint that quietly grows past the true
panel edge feeds extra roof pixels into fit_roof_plane().
"""
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from level3_cv_detection import (  # noqa: E402
    TARGET_MPP,
    dominant_orientation_deg,
    footprint_change,
    regularize_footprint,
)

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- shape makers
def rect(h, w, canvas=200):
    m = np.zeros((canvas, canvas), bool)
    y, x = (canvas - h) // 2, (canvas - w) // 2
    m[y : y + h, x : x + w] = True
    return m


def rotate_shape(mask, deg):
    return ndimage.rotate(mask.astype(np.uint8), deg, reshape=False, order=0) > 0


def angle_error(got, want):
    """Smallest difference mod 90 -- orientation is only defined that far."""
    d = abs(got - want) % 90.0
    return min(d, 90.0 - d)


def min_area_rect_angle(mask):
    """Brute-force minimum-area bounding rectangle angle, for comparison."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    best, best_area = 0.0, np.inf
    for deg in range(90):
        t = np.radians(deg)
        r = pts @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if area < best_area:
            best, best_area = float(deg), area
    return best


def rectilinearity(mask):
    """Share of boundary gradient that runs along the two dominant directions.

    1.0 for a perfect axis-aligned rectangle, low for a blob. Measured in the
    mask's own frame, so it does not presuppose which angle is 'right'.
    """
    gy, gx = np.gradient(mask.astype(float))
    mag = np.hypot(gx, gy)
    if mag.sum() == 0:
        return 0.0
    ang = np.degrees(np.arctan2(gy, gx)) % 90.0
    theta = dominant_orientation_deg(mask)
    off = np.minimum(np.abs(ang - theta), 90.0 - np.abs(ang - theta))
    return float(mag[off <= 5.0].sum() / mag.sum())


# ------------------------------------------------------------------- the tests
print("dominant_orientation_deg()")

check("axis-aligned rectangle reads 0 deg (mod 90)",
      angle_error(dominant_orientation_deg(rect(80, 40)), 0) <= 1,
      f"got {dominant_orientation_deg(rect(80, 40)):.0f}")

for want in (15, 30, 45, 60, 75):
    got = dominant_orientation_deg(rotate_shape(rect(80, 40), want))
    check(f"rectangle rotated {want} deg is recovered", angle_error(got, want) <= 2,
          f"got {got:.0f}, error {angle_error(got, want):.1f} deg")

a0, a90 = dominant_orientation_deg(rect(80, 40)), dominant_orientation_deg(rect(40, 80))
check("orientation is defined mod 90 (a rectangle and its transpose agree)",
      angle_error(a0, a90) <= 1, f"{a0:.0f} vs {a90:.0f}")

# a vote over the whole outline should shrug off one stray blob; the minimum-area
# bounding rectangle, decided by extreme points, should not
noisy = rect(80, 40).copy()
noisy[20:26, 150:156] = True            # detached speck, far off-axis
vote, mar = dominant_orientation_deg(noisy), min_area_rect_angle(noisy)
check("a stray blob does not move the vote", angle_error(vote, 0) <= 2, f"vote {vote:.0f} deg")
check("...where the minimum-area rectangle it replaces IS moved",
      angle_error(mar, 0) > 2, f"min-area-rect {mar:.0f} deg vs vote {vote:.0f} deg")

print("\nrotation round trip (the control -- cost of resampling, not of the model)")
for deg in (0, 30, 45):
    r = rotate_shape(rotate_shape(rect(80, 40), deg), -deg)
    iou = (r & rect(80, 40)).sum() / (r | rect(80, 40)).sum()
    check(f"rotate {deg} deg and back keeps IoU >= 0.97", iou >= 0.97, f"IoU {iou:.4f}")

print("\nregularize_footprint() -- what it preserves")

clean = rect(80, 40)
reg, th = regularize_footprint(clean)
ch = footprint_change(clean, reg)
check("a clean rectangle survives nearly untouched", ch["iou"] >= 0.95,
      f"IoU {ch['iou']:.3f}, {ch['area_before_m2']:.1f} -> {ch['area_after_m2']:.1f} m2")

areas = []
for deg in (0, 17, 33, 45, 61):
    r, _ = regularize_footprint(rotate_shape(clean, deg))
    areas.append(r.sum() * TARGET_MPP**2)
spread = (max(areas) - min(areas)) / np.mean(areas)
check("measured area is rotation-invariant to within 10%", spread < 0.10,
      f"spread {spread:.1%} over {[f'{a:.0f}' for a in areas]} m2")

# Idempotence is the test that caught the rotation round-trip offset: before it
# was corrected, IoU here was bimodal -- exactly 1.000 at 0/45/90 deg and
# exactly 0.929 everywhere else, with identical areas, which is the signature of
# a whole-pixel translation rather than a shape change.
ious, areas = [], []
for deg in (0, 11, 22, 33, 45, 56, 67, 78):
    once, _ = regularize_footprint(rotate_shape(clean, deg))
    twice, _ = regularize_footprint(once)
    ious.append((once & twice).sum() / (once | twice).sum())
    areas.append(abs(int(twice.sum()) - int(once.sum())) / max(1, int(once.sum())))
# Characterised, not demanded: snapping a ROTATED raster to a grid cannot be
# exactly idempotent, and re-application is not a real use case anyway -- a
# footprint is regularized once. What must hold is that the second pass changes
# the SIZE of nothing, only its registration by about a pixel. Centring the
# grid to remove even that displacement was tried and made both numbers worse
# (IoU 0.87, area 10%), so ~1.4 px is the accepted cost of a stable grid.
check("re-application displaces the footprint but does not resize it",
      max(areas) <= 0.01, f"worst area change {max(areas):.2%} over 8 angles")
check("...and the displacement stays around one pixel (IoU >= 0.92)",
      min(ious) >= 0.92, f"worst IoU {min(ious):.3f}; identical shape, ~1.4 px over")

print("\nregularize_footprint() -- what it cleans up")

# The rule is simply: a cell survives if at least half of it is covered. Every
# behaviour below follows from that one line, including the awkward ones.
thin = rect(80, 40).copy()
thin[100:130, 120:121] = True           # 1px whisker = 25% of a 4px cell
reg_thin, _ = regularize_footprint(thin)
check("a whisker covering under half a cell is voted away",
      int(reg_thin[100:130, 120:121].sum()) < 8,
      f"{int(thin[100:130, 120:121].sum())} -> {int(reg_thin[100:130, 120:121].sum())} px")

nick = rect(80, 40).copy()
nick[100:102, 100:102] = False          # 2x2 hole = 25% of a cell, so 75% remains
reg_nick, _ = regularize_footprint(nick)
check("a hole leaving the cell over half covered is closed",
      int(reg_nick[100:102, 100:102].sum()) >= 3,
      f"{int(reg_nick[100:102, 100:102].sum())}/4 px restored")

check("regularization raises rectilinearity",
      rectilinearity(reg_thin) > rectilinearity(thin),
      f"{rectilinearity(thin):.3f} -> {rectilinearity(reg_thin):.3f}")

print("\nregularize_footprint() -- what it deliberately does NOT do")

# an L-shaped array must stay L-shaped. This is the test that separates a grid
# snap from a bounding rectangle, which would swallow the whole notch.
L = rect(60, 60).copy()
ys, xs = np.nonzero(L)
L[ys.min() : ys.min() + 30, xs.min() : xs.min() + 30] = False
reg_L, _ = regularize_footprint(L)
bbox_px = 60 * 60
check("an L-shaped array stays L-shaped, not filled to its bounding box",
      reg_L.sum() < 0.80 * bbox_px, f"{int(reg_L.sum())} px vs {bbox_px} px bounding box")
check("...while keeping the L itself", reg_L.sum() > 0.85 * L.sum(),
      f"{int(reg_L.sum())} px vs {int(L.sum())} px original")

# gap filling is bridge_mask_gaps()'s job, at 12px. This reaches half a cell.
big_bite = rect(80, 40).copy()
big_bite[60:75, 80:95] = False          # 15px hole, far bigger than a cell
reg_bite, _ = regularize_footprint(big_bite)
check("a hole much larger than one cell is NOT filled (that is bridge_mask_gaps' job)",
      reg_bite[62:73, 82:93].sum() < 20,
      f"{int(reg_bite[62:73, 82:93].sum())} px of 121 filled")

# The surprising corollary, asserted so nobody has to rediscover it: a hole
# that drops a cell just under half coverage removes the WHOLE cell, so the
# hole gets bigger. Grid snapping cannot fill; it only keeps or drops cells.
mid = rect(80, 40).copy()
mid[100:103, 100:103] = False           # 3x3 = 9/16, leaves 44% -> cell dropped
reg_mid, _ = regularize_footprint(mid)
check("a hole leaving the cell just UNDER half coverage widens to the full cell",
      int(reg_mid[100:103, 100:103].sum()) <= 1,
      f"{int(reg_mid[100:103, 100:103].sum())}/9 px kept -- the notch grew")

# only meaningful now the grid is anchored on the mask, not the canvas
shifted = np.zeros_like(clean)
shifted[2:, 2:] = clean[:-2, :-2]
a_area = regularize_footprint(clean)[0].sum()
b_area = regularize_footprint(shifted)[0].sum()
check("shifting the same array by 2px does not change its regularized area",
      abs(a_area - b_area) <= 0.02 * a_area, f"{int(a_area)} px vs {int(b_area)} px")

# documents the limit rather than hiding it: orientation needs a big enough shape
small = rotate_shape(rect(12, 8), 30)
got_small = dominant_orientation_deg(small)
check("orientation on a ~10 m2 footprint is NOT reliable (documented, not fixed)",
      angle_error(got_small, 30) > 2,
      f"got {got_small:.0f} deg for a 30 deg shape -- consistent with the "
      f"documented small-footprint leverage problem")

empty = np.zeros((50, 50), bool)
reg_empty, th_empty = regularize_footprint(empty)
check("an empty mask returns empty, not a crash", not reg_empty.any() and th_empty == 0.0)

single = np.zeros((50, 50), bool)
single[25, 25] = True
reg_single, _ = regularize_footprint(single)
check("a single pixel is voted away rather than grown to a cell",
      reg_single.sum() == 0, f"{int(reg_single.sum())} px")

print("\n" + (f"{len(FAILS)} FAILED: {FAILS}" if FAILS else "all checks passed"))
sys.exit(1 if FAILS else 0)
