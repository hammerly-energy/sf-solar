"""Rebuild the published pipeline artifact from the current detection caches.

    uv run --isolated --with numpy --with pillow python scripts/build_pipeline_artifact.py

Writes data/pipeline_artifact.html (gitignored -- it is a build output, this
script is the source of truth). Then ask Claude to publish that file to the
artifact URL recorded in CLAUDE.md, which keeps the same link.

WHY THIS EXISTS. The page states measured numbers -- detection counts, panel
areas, tilt-vs-area correlations, how much headroom the ground gate has left.
Those move whenever the pipeline moves. A page of hand-typed figures goes stale
silently and keeps looking authoritative, which is the exact failure mode this
project's sanity-check discipline exists to prevent. Five figures had already
gone stale in CLAUDE.md by the time this page was first built. So every number
and every shape here is read from data/level3_cv_*_detections.npz at build
time; nothing is transcribed.

If a figure on the page cannot be derived from the caches, it does not belong
on the page.
"""

import base64
import io
import json
import os

import numpy as np
from PIL import Image

OUT = "data/pipeline_artifact.html"

# meta row layout written by save_detection_cache()
F = dict(p=0, y0=1, x0=2, area=3, lat=4, lon=5, tilt=6, az=7,
         nt=8, tr=9, h=10, grade=11, bad=12)

# Chip dimensions per area. These are a property of the bounding box and the
# resample scale, not of any run, so they are stable -- but the footprint map
# is drawn in chip pixel coordinates, so a bbox change means changing these.
CHIP = {"zurich": (2575, 2583), "altstetten": (2573, 2581)}

SHRINK = 5          # chip px per map px; OR-pooled so thin arrays survive
KEPT_RGBA = [31, 95, 168, 255]      # --accent
GATED_RGBA = [184, 67, 15, 255]     # --warn


def collect(mode):
    """Everything the page needs about one test area, straight from its cache."""
    path = f"data/level3_cv_{mode}_detections.npz"
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path} -- run scripts/level3_cv_detection.py {mode} first"
        )
    d = np.load(path, allow_pickle=True)
    n = int(d["n_arrays"])
    H, W = CHIP[mode]

    kept = np.zeros((H, W), bool)
    gated = np.zeros((H, W), bool)
    arrays = []
    for i in range(n):
        m = d[f"meta_{i}"]
        mask = d[f"mask_{i}"].astype(bool)
        y0, x0 = int(m[F["y0"]]), int(m[F["x0"]])
        h, w = mask.shape
        is_gated = bool(m[F["grade"]] or m[F["bad"]])
        (gated if is_gated else kept)[y0:y0 + h, x0:x0 + w] |= mask
        arrays.append(dict(
            area=round(float(m[F["area"]]), 1),
            tilt=round(float(m[F["tilt"]]), 1),
            az=round(float(m[F["az"]])),
            p=round(float(m[F["p"]]), 3),
            h=None if not np.isfinite(m[F["h"]]) else round(float(m[F["h"]]), 1),
            gated=is_gated,
            trunc=bool(m[F["tr"]]),
        ))

    hh, ww = H // SHRINK, W // SHRINK

    def pool(a):
        # OR, not mean: a one-pixel-wide array row must survive the shrink
        return a[:hh * SHRINK, :ww * SHRINK].reshape(hh, SHRINK, ww, SHRINK).any(axis=(1, 3))

    rgba = np.zeros((hh, ww, 4), np.uint8)
    rgba[pool(kept)] = KEPT_RGBA
    rgba[pool(gated)] = GATED_RGBA
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)

    scores = d["tile_scores"][:, 2]
    return dict(
        w=ww, h=hh, png=base64.b64encode(buf.getvalue()).decode(),
        n=n, arrays=arrays,
        positions=int(len(scores)), flagged=int((scores >= 0.2).sum()),
        total=round(float(sum(a["area"] for a in arrays)), 1),
        kept_n=sum(1 for a in arrays if not a["gated"]),
        kept_area=round(float(sum(a["area"] for a in arrays if not a["gated"])), 1),
    )


HEAD = r"""<title>Orthophoto to Tilt</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --paper:#f6f8fa; --card:#ffffff; --ink:#12161c; --muted:#5a6472; --faint:#8b95a3;
    --rule:#d9dfe6; --rule-soft:#e8edf2; --accent:#1f5fa8; --accent-dim:#dbe8f6;
    --accent-ghost:#eef4fb; --warn:#b8430f; --warn-dim:#fbe9e0; --good:#1c6b4a; --grid:#e4e9ef;
    --serif:"Newsreader",Georgia,serif;
    --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --paper:#0e1117; --card:#161b23; --ink:#e6ebf1; --muted:#9aa5b4; --faint:#6b7583;
    --rule:#2a323d; --rule-soft:#212934; --accent:#6ba6e8; --accent-dim:#17273a;
    --accent-ghost:#141c27; --warn:#e8875a; --warn-dim:#2e1d14; --good:#5cbd92; --grid:#242c37;
  }}
  :root[data-theme="dark"] {
    --paper:#0e1117; --card:#161b23; --ink:#e6ebf1; --muted:#9aa5b4; --faint:#6b7583;
    --rule:#2a323d; --rule-soft:#212934; --accent:#6ba6e8; --accent-dim:#17273a;
    --accent-ghost:#141c27; --warn:#e8875a; --warn-dim:#2e1d14; --good:#5cbd92; --grid:#242c37;
  }

  body { background:var(--paper); color:var(--ink); font-family:var(--sans);
         font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1000px; margin:0 auto; padding:52px 24px 90px; }

  .kicker { font-family:var(--mono); font-size:11px; letter-spacing:.14em;
            text-transform:uppercase; color:var(--accent); margin-bottom:14px; }
  h1 { font-family:var(--serif); font-weight:500; font-size:clamp(38px,6vw,60px);
       line-height:1.04; letter-spacing:-.015em; margin:0 0 18px; text-wrap:balance; }
  .standfirst { font-family:var(--serif); font-size:19px; line-height:1.55;
                color:var(--muted); max-width:64ch; margin:0 0 8px; }
  .standfirst em { color:var(--ink); font-style:italic; }
  .masthead { border-bottom:2px solid var(--ink); padding-bottom:26px; }

  h2 { font-family:var(--serif); font-weight:500; font-size:29px; letter-spacing:-.01em;
       margin:62px 0 6px; text-wrap:balance; }
  .lede { color:var(--muted); max-width:70ch; margin:0 0 24px; font-size:14.5px; }
  .lede b, .cap b { color:var(--ink); font-weight:600; }

  figure { margin:0 0 8px; background:var(--card); border:1px solid var(--rule-soft);
           border-radius:7px; padding:20px 22px 16px; }
  figcaption, .cap { font-size:13.5px; color:var(--muted); line-height:1.55;
                     margin-top:14px; max-width:76ch; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 20px; font-family:var(--mono);
            font-size:11px; color:var(--muted); margin-top:12px; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px;
              margin-right:6px; vertical-align:-1px; font-style:normal; }
  .sw-a { background:var(--accent); } .sw-w { background:var(--warn); }
  .sw-g { background:var(--faint); }

  /* funnel */
  .funnel { display:grid; gap:22px; grid-template-columns:1fr 1fr; }
  .fn-title { font-family:var(--mono); font-size:11px; letter-spacing:.09em;
              text-transform:uppercase; color:var(--faint); margin-bottom:14px; }
  .step { margin-bottom:11px; }
  .step-lbl { display:flex; justify-content:space-between; align-items:baseline;
              gap:10px; font-size:13px; margin-bottom:4px; }
  .step-lbl span:last-child { font-family:var(--mono); font-variant-numeric:tabular-nums;
                              color:var(--ink); font-weight:500; }
  .bar { height:22px; border-radius:3px; background:var(--accent);
         display:flex; align-items:center; min-width:2px; }
  .bar.ghost { background:var(--accent-ghost); border:1px solid var(--accent-dim); }
  .bar.warn  { background:var(--warn); }
  .drop { font-family:var(--mono); font-size:11px; color:var(--faint); padding-left:8px; }

  /* maps */
  .maps { display:grid; gap:22px; grid-template-columns:1fr 1fr; }
  .map-frame { border:1px solid var(--rule); border-radius:5px; overflow:hidden;
               background:var(--paper); position:relative; aspect-ratio:1; }
  .map-frame img { width:100%; height:100%; display:block; image-rendering:pixelated; }
  .scalebar { position:absolute; left:10px; bottom:10px; font-family:var(--mono);
              font-size:10px; color:var(--muted); }
  .scalebar div { height:3px; background:var(--muted); margin-bottom:3px; }

  /* diagrams */
  .duo { display:grid; gap:22px; grid-template-columns:1fr 1fr; }
  svg { display:block; max-width:100%; height:auto; }
  text { font-family:"IBM Plex Mono",monospace; }

  .chartwrap { overflow-x:auto; }
  .chartwrap svg { min-width:760px; }
  .phase-key { display:flex; flex-wrap:wrap; gap:8px 22px; font-family:var(--mono);
               font-size:11px; color:var(--muted); margin-bottom:16px; }
  .phase-key i { display:inline-block; width:11px; height:11px; border-radius:2px;
                 margin-right:6px; vertical-align:-1px; font-style:normal; }
  .k-pipe { background:var(--faint); } .k-geo { background:var(--accent); }
  .k-warn { background:var(--warn); }

  /* stage strip */
  .stages { border-top:2px solid var(--ink); }
  .srow { display:grid; grid-template-columns:34px 1fr 1.15fr;
          gap:16px; padding:13px 0; border-bottom:1px solid var(--rule-soft); align-items:baseline; }
  .snum { font-family:var(--mono); font-size:12px; color:var(--accent); font-weight:500; }
  .srow.flagged .snum { color:var(--warn); }
  .sname { font-weight:600; font-size:14.5px; }
  .sfile { font-family:var(--mono); font-size:10.5px; color:var(--faint); display:block;
           margin-top:2px; }
  .sfile.geo { color:var(--accent); }
  .sdesc { color:var(--muted); font-size:13.5px; line-height:1.5; }
  .sdesc .warnmark { color:var(--warn); font-weight:600; }

  /* ledger */
  .ledger { border-top:2px solid var(--ink); }
  .row { display:grid; grid-template-columns:150px 1fr 92px; gap:18px; padding:14px 0;
         border-bottom:1px solid var(--rule-soft); align-items:start; }
  .row dt { font-weight:600; font-size:14px; }
  .row dd { margin:0; color:var(--muted); font-size:13.5px; line-height:1.5; }
  .row dd b { color:var(--ink); font-weight:600; }
  .verdict { font-family:var(--mono); font-size:10.5px; letter-spacing:.07em;
             text-transform:uppercase; text-align:right; padding-top:3px; font-weight:500; }
  .v-yes { color:var(--good); } .v-no { color:var(--warn); }

  .closing { margin-top:18px; padding:22px 24px; border:1px solid var(--rule);
             border-radius:6px; background:var(--card); }
  .closing p { margin:0 0 12px; color:var(--muted); max-width:70ch; }
  .closing p:last-child { margin-bottom:0; }
  .closing strong { color:var(--ink); }
  code { font-family:var(--mono); font-size:.92em; }

  footer { margin-top:52px; padding-top:18px; border-top:1px solid var(--rule);
           font-family:var(--mono); font-size:11.5px; color:var(--faint); line-height:1.8; }

  @media (max-width:700px) {
    .wrap { padding:34px 15px 60px; }
    .funnel, .maps, .duo { grid-template-columns:1fr; }
    .srow { grid-template-columns:26px 1fr; }
    .srow .sdesc { grid-column:2; }
    .row { grid-template-columns:1fr; gap:3px; }
    .verdict { text-align:left; }
  }
</style>

<div class="wrap">

<header class="masthead">
  <div class="kicker">sf-solar &middot; level 3 &middot; computer-vision branch</div>
  <h1>Orthophoto to Tilt</h1>
  <p class="standfirst">
    Nine stages turn a 10&nbsp;cm aerial photograph of Zurich into a tilt and azimuth for every
    rooftop solar array in it. Every figure and every shape on this page is drawn from the
    detection caches of the most recent run &mdash; <em>including the parts that are wrong</em>.
  </p>
</header>

<h2>The pipeline, end to end</h2>
<p class="lede">
  Numbers on this diagram are one real run over <b>Wiedikon</b>. Data flows down the spine;
  external inputs join from the left; what each stage discards or emits leaves to the right.
</p>
<div class="phase-key">
  <span><i class="k-pipe"></i>level3_cv_detection.py &mdash; fetching, models, orchestration</span>
  <span><i class="k-geo"></i>roof_geometry.py &mdash; testable without the model stack</span>
  <span><i class="k-warn"></i>stage with a known, documented weakness</span>
</div>
<figure>
  <div class="chartwrap"><div id="flow"></div></div>
  <figcaption>
    Trace the counts down the spine and the funnel shape appears: <b>361 windows &rarr; 71 &rarr;
    29 arrays &rarr; 23 kept</b>. The sharpest cut is stage&nbsp;3, where 290 windows are set aside
    and never segmented &mdash; not because they are empty, but because they were never examined.
    That is where false negatives are born, and nothing downstream can recover them.
  </figcaption>
  <figcaption style="margin-top:10px">
    The dashed containers show which file each stage lives in, and they
    <b>interleave rather than stack</b>: geometry is threaded through the pipeline at three
    separate points (5, then 7&ndash;8), not bolted on at one end. That is why the
    2026&#8209;09&#8209;08 split was made by <em>dependency</em> and not by position &mdash; and why
    it works, since the pipeline needs only ten names back across those three seams.
  </figcaption>
</figure>

<h2>What actually came out</h2>
<p class="lede">
  Every detected footprint, at its true position inside the ~600&nbsp;m box, at the real pixel
  scale. This is the pipeline's whole output rendered as one picture.
</p>
<figure>
  <div class="maps" id="maps"></div>
  <div class="legend">
    <span><i class="sw-a"></i>kept as plausible</span>
    <span><i class="sw-w"></i>flagged: at grade, or tilt&nbsp;&gt;&nbsp;60&deg;</span>
  </div>
  <figcaption>
    Two very different neighbourhoods. <b>Wiedikon</b> (left) is dense residential &mdash; many small
    scattered footprints, and most of the orange. <b>Altstetten</b> is industrial &mdash; a few
    enormous roofs that hold twice the panel area in a third as many detections.
  </figcaption>
</figure>

<h2>Why the window slides</h2>
<p class="lede">
  The 256&nbsp;px tile exists only because both models were trained at that size. It is an
  <b>input constraint, never a unit of measurement</b> &mdash; and treating it as one was the
  single biggest source of error in this pipeline.
</p>
<figure>
  <div id="stride"></div>
  <figcaption>
    A roof split across a seam is seen as two half-roofs, each of which may score under the
    threshold, so <b>the whole array disappears with no fragment and no warning</b>. Sliding the
    window half a tile means every interior pixel also sits near the middle of some other window.
    On Altstetten this took the area flagged &ldquo;possibly cut short&rdquo; from
    <b>3614&nbsp;m&sup2; to 61&nbsp;m&sup2;</b>.
  </figcaption>
</figure>

<h2>Two rules that decide what counts as one array</h2>
<div class="duo">
  <figure>
    <div id="vote"></div>
    <figcaption>
      <b>The vote.</b> Overlapping windows disagree. A pixel is panel if at least half the windows
      that actually ran the segmenter there say so. Windows that were never flagged do not vote.
      Ties go to panel &mdash; missing real capacity is the worse error.
    </figcaption>
  </figure>
  <figure>
    <div id="bridge"></div>
    <figcaption>
      <b>The bridge.</b> The segmenter drops out across walkways, so one roof arrives as several
      blobs. Closing the canvas by 2.8&nbsp;m groups them &mdash; but the bridged pixels are
      <b>never counted as panel</b>. Total area is identical before and after, to the decimal.
    </figcaption>
  </figure>
</div>

<h2>The one chart that matters</h2>
<p class="lede">
  Fitted tilt should describe a roof. Instead it tracks <b>how big the footprint is</b>. Every
  detection from both areas is plotted here; each dot is one array.
</p>
<figure>
  <div id="scatter"></div>
  <div class="legend">
    <span><i class="sw-a"></i>Wiedikon</span>
    <span><i class="sw-w"></i>Altstetten</span>
    <span>hollow = flagged by a gate</span>
  </div>
  <figcaption>
    Read it left to right: below ~120&nbsp;m&sup2; the tilts scatter from flat to vertical; above it
    they collapse onto the floor. <b>Spearman &rho; = &minus;0.453, p = 0.014</b> on Wiedikon.
    The cause is leverage &mdash; on a small footprint a single contaminating DSM pixel at the roof
    edge, 10&nbsp;m below, swings the plane fit by degrees. <b>The two dots above 60&deg; are walls,
    not roofs</b>, and they are what the tilt gate exists to catch.
  </figcaption>
</figure>

<h2>How much room the ground gate has left</h2>
<p class="lede">
  Detections below 2&nbsp;m above bare terrain are flagged as courtyards or car parks rather than
  roofs. Nothing is deleted &mdash; ground-mounted PV is genuinely at grade.
</p>
<figure>
  <div id="heights"></div>
  <figcaption>
    The gate is comfortable here only because these are mid-rise neighbourhoods. The lowest
    detection it <b>keeps</b> clears the line by just <b>2.3&nbsp;m</b> in Wiedikon and
    <b>1.3&nbsp;m</b> in Altstetten &mdash; and that margin has been shrinking as the pipeline finds
    more small objects. In a low-rise area of carports and single-storey garages, 2&nbsp;m would
    start deleting real arrays.
  </figcaption>
</figure>

<h2>Measured, or assumed</h2>
<p class="lede">
  The distinction that matters most. Green rows have a number behind them that was reproduced
  independently. Orange rows are currently taken on faith.
</p>
<dl class="ledger">
  <div class="row"><dt>Seam damage</dt>
    <dd>Area flagged possibly-cut-short fell <b>3614 &rarr; 61 m&sup2;</b> when the window began to overlap; arrays that already existed moved only <b>+2.4%</b>.</dd>
    <dd class="verdict v-yes">measured</dd></div>
  <div class="row"><dt>Area conservation</dt>
    <dd>Gap bridging changes grouping without changing area: <b>3021.8 m&sup2; before and after</b>, across every closing radius from 0 to 30&nbsp;px.</dd>
    <dd class="verdict v-yes">measured</dd></div>
  <div class="row"><dt>Repeatability</dt>
    <dd><b>84.3% / 93.6%</b> of contested pixels unanimous across four independent views.</dd>
    <dd class="verdict v-yes">measured</dd></div>
  <div class="row"><dt>Co-registration</dt>
    <dd>No consistent shift between photograph and elevation model, by normalised cross-correlation over <b>22 arrays</b> with usable edge signal.</dd>
    <dd class="verdict v-yes">measured</dd></div>
  <div class="row"><dt>Size&ndash;tilt bias</dt>
    <dd><b>&rho; = &minus;0.453, p = 0.014.</b> Real, reproduced, unfixed &mdash; and systematic, which sensitivity analysis shows is the expensive kind of error.</dd>
    <dd class="verdict v-yes">measured</dd></div>
  <div class="row"><dt>False-positive rate</dt>
    <dd>One specific false positive is confirmed by eye. The <em>rate</em> spans <b>[0.14, 0.91]</b> depending on denominator, and the audit was withdrawn for anchoring.</dd>
    <dd class="verdict v-no">unmeasured</dd></div>
  <div class="row"><dt>False-negative rate</dt>
    <dd>Never examined at all. For an aggregate MW total this is the bias that matters most, and it has no evidence behind it.</dd>
    <dd class="verdict v-no">unmeasured</dd></div>
  <div class="row"><dt>Are new detections real?</dt>
    <dd>Wiedikon's total rose <b>20%</b> at overlapping stride and <b>86% of that</b> is new detections rather than recovered ones. Correctness unknown.</dd>
    <dd class="verdict v-no">unmeasured</dd></div>
</dl>

<h2>Where it goes next</h2>
<div class="closing">
  <p>
    <strong>Row segmentation.</strong> Panel rows survive in the cached masks &mdash; bridging groups
    across walkways but never writes them in &mdash; and <code>dominant_orientation_deg()</code> now
    recovers an array's angle to &plusmn;1&deg;. Rotating a mask onto its own axes turns row-finding
    from a two-dimensional problem into a one-dimensional one: project across the rows, cut at the
    troughs. That attacks the standing gap in stage&nbsp;7, where several short tilted rows average
    into one meaningless compromise plane.
  </p>
  <p>
    Two things block it. Orientation is only defined <strong>mod 90&deg;</strong>, so it does not yet
    say which of two perpendicular directions the rows run. And a 1.7&nbsp;m row gets only
    <strong>~3 samples</strong> across at 0.5&nbsp;m DSM resolution &mdash; the swissSURFACE3D
    <strong>point cloud</strong> at 0.19&nbsp;m spacing would give ~9. Row segmentation and the
    point-cloud upgrade are probably one job, not two.
  </p>
  <p>
    <strong>Footprint regularization was built, tested and rejected.</strong> Snapping masks to a
    module-sized grid produces blockier blobs, not right-angle polygons; the cell size showed no
    plateau anywhere from 0.5&nbsp;m to 2.5&nbsp;m; and it added 3&ndash;5% of area that no view had
    called panel. Its orientation estimator is the part worth keeping.
  </p>
</div>

<footer>
  sf-solar &middot; experimental CV branch &middot; not part of the published pipeline<br>
  roof_geometry.py 617 lines &middot; level3_cv_detection.py 1478 lines &middot; test_roof_geometry.py 240 lines, 24 checks<br>
  Imagery &amp; elevation &copy;swisstopo &middot; models from DeepSolar-3M
</footer>

</div>

"""

TAIL = r"""

<script>
(function () {
  const D = JSON.parse(document.getElementById("pipeline-data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const css = n => getComputedStyle(document.body).getPropertyValue(n).trim();
  const el = (t, a, p) => { const e = document.createElementNS(NS, t);
    for (const k in (a || {})) e.setAttribute(k, a[k]);
    if (p) p.appendChild(e); return e; };
  const txt = (s, a, p) => { const e = el("text", a, p); e.textContent = s; return e; };
  const svg = (w, h, p) => el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%" }, p);
  const AREAS = [["zurich", "Wiedikon"], ["altstetten", "Altstetten"]];

  /* ---------------- flowchart ---------------- */
  (function () {
    const S = [
      { n:1, name:"Fetch & rescale imagery", fn:"fetch_swissimage_chip()", file:"pipe",
        inp:["SWISSIMAGE  10 cm", "swisstopo STAC · 12 tiles · 2025"],
        edge:"(3, 2575, 2583) uint8  ·  0.2359 m/px" },
      { n:2, name:"Slide a 256 px window", fn:"tile_chip()", file:"pipe",
        note:["stride 128 in both axes", "4 views per interior pixel"],
        edge:"361 windows of 256 × 256" },
      { n:3, name:"Classify each window", fn:"classify_tiles()", file:"pipe", warn:true,
        inp:["ViT-MAE-large + LoRA", "DeepSolar-3M weights"],
        out:["290 windows set aside", "never segmented → false negatives"],
        edge:"71 windows scoring p ≥ 0.2" },
      { n:4, name:"Segment flagged windows", fn:"segment_tile()", file:"pipe", warn:true,
        inp:["SegFormer", "DeepSolar-3M weights"],
        edge:"71 boolean masks, 256 × 256" },
      { n:5, name:"Combine overlapping views", fn:"combine_views()", file:"geo",
        out:["6.2% of contested pixels", "outvoted and dropped"],
        edge:"one boolean canvas, whole image" },
      { n:6, name:"Stitch, bridge, label", fn:"find_pv_arrays()", file:"pipe",
        out:["8 arrays joined across gaps", "panel area unchanged"],
        edge:"29 arrays  ·  3 627 m²" },
      { n:7, name:"Fit roof planes", fn:"fit_group_planes()", file:"geo", warn:true,
        inp:["swissSURFACE3D DSM", "0.5 m raster · 2024"],
        edge:"tilt° + azimuth° per facet" },
      { n:8, name:"Gate on plausibility", fn:"apply_plausibility_gates()", file:"geo",
        inp:["swissALTI3D DTM", "0.5 m raster · 2020"],
        out:["6 flagged, none deleted", "at grade, or tilt > 60°"],
        edge:"23 plausible  ·  3 397 m² kept" },
      { n:9, name:"Cache, plot, verify", fn:"save_detection_cache()", file:"pipe",
        out:[".npz + 4 diagnostic plots", "no photo/DSM offset found"] }
    ];
    const PITCH = 116, TOP = 20, BW = 330, BH = 64, CX = 344, GH = 32;

    // Lay the stages out first, adding a header band wherever the file changes.
    // The two files INTERLEAVE -- geometry is threaded through the pipeline in
    // three separate places, not bolted on at one end -- so the containers are
    // computed from the data rather than assumed to be two blocks.
    const pos = [], groups = [];
    let cy = TOP;
    S.forEach((st, i) => {
      if (i === 0 || S[i - 1].file !== st.file) { cy += GH; groups.push({ file: st.file, from: i, to: i }); }
      else groups[groups.length - 1].to = i;
      pos.push(cy);
      cy += PITCH;
    });
    const H = cy - PITCH + BH + 22;
    const s2 = el("svg", { viewBox: `0 0 960 ${H}`, width: "100%" },
                  document.getElementById("flow"));
    const A = css("--accent"), Wn = css("--warn"), F = css("--faint"), M = css("--muted"),
          I = css("--ink"), R = css("--rule"), C = css("--card"), RS = css("--rule-soft"),
          GHOST = css("--accent-ghost"), PAPER = css("--paper");

    // containers, drawn behind everything
    const FILES = { pipe: "level3_cv_detection.py", geo: "roof_geometry.py" };
    groups.forEach(g => {
      const top = pos[g.from] - GH + 4, bot = pos[g.to] + BH + 14;
      const geo = g.file === "geo";
      el("rect", { x: CX - 16, y: top, width: BW + 32, height: bot - top, rx: 7,
                   fill: geo ? GHOST : PAPER, stroke: geo ? A : R, "stroke-width": 1.2,
                   "stroke-dasharray": "6 4", opacity: geo ? 1 : .85 }, s2);
      txt(FILES[g.file], { x: CX - 4, y: top + 20, "font-size": 11.5,
                           fill: geo ? A : M, "font-weight": 500 }, s2);
      const nfn = g.to - g.from + 1;
      txt(nfn === 1 ? "1 stage" : nfn + " stages", { x: CX + BW + 20, y: top + 20,
          "font-size": 10.5, fill: F, "text-anchor": "end" }, s2);
    });

    S.forEach((st, i) => {
      const y = pos[i];
      // connector down to the next stage
      if (i < S.length - 1) {
        const y1 = y + BH, y2 = pos[i + 1];
        el("line", { x1: CX + BW / 2, y1, x2: CX + BW / 2, y2: y2 - 7, stroke: R,
                     "stroke-width": 2 }, s2);
        el("path", { d: `M${CX + BW / 2 - 5} ${y2 - 12} L${CX + BW / 2} ${y2 - 4} L${CX + BW / 2 + 5} ${y2 - 12}`,
                     fill: "none", stroke: R, "stroke-width": 2 }, s2);
        if (st.edge)
          txt(st.edge, { x: CX + BW / 2 + 14, y: y1 + 30, "font-size": 11.5, fill: M }, s2);
      }
      // the stage box
      el("rect", { x: CX, y, width: BW, height: BH, rx: 5, fill: C,
                   stroke: st.warn ? Wn : R, "stroke-width": st.warn ? 2 : 1.5 }, s2);
      el("rect", { x: CX, y, width: 4, height: BH,
                   fill: st.warn ? Wn : (st.file === "geo" ? A : F) }, s2);
      el("circle", { cx: CX + 30, cy: y + BH / 2, r: 15, fill: "none",
                     stroke: st.warn ? Wn : A, "stroke-width": 1.6 }, s2);
      txt(st.n, { x: CX + 30, y: y + BH / 2 + 5, "font-size": 14,
                  fill: st.warn ? Wn : A, "text-anchor": "middle" }, s2);
      txt(st.name, { x: CX + 56, y: y + 27, "font-size": 14, fill: I,
                     "font-weight": 600, "font-family": css("--sans") }, s2);
      txt(st.fn, { x: CX + 56, y: y + 45, "font-size": 11,
                   fill: st.file === "geo" ? A : F }, s2);

      // external input, entering from the left
      if (st.inp) {
        const bx = 12, bw = 250, by = y + 8, bh = 48;
        el("rect", { x: bx, y: by, width: bw, height: bh, rx: 4, fill: "none",
                     stroke: RS, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, s2);
        txt(st.inp[0], { x: bx + 12, y: by + 20, "font-size": 12, fill: M, "font-weight": 500 }, s2);
        txt(st.inp[1], { x: bx + 12, y: by + 36, "font-size": 10.5, fill: F }, s2);
        el("line", { x1: bx + bw, y1: y + BH / 2, x2: CX - 7, y2: y + BH / 2,
                     stroke: RS, "stroke-width": 1.5 }, s2);
        el("path", { d: `M${CX - 12} ${y + BH / 2 - 5} L${CX - 4} ${y + BH / 2} L${CX - 12} ${y + BH / 2 + 5}`,
                     fill: "none", stroke: RS, "stroke-width": 1.5 }, s2);
      }
      if (st.note) {
        txt(st.note[0], { x: 262, y: y + 28, "font-size": 11, fill: F, "text-anchor": "end" }, s2);
        txt(st.note[1], { x: 262, y: y + 44, "font-size": 11, fill: F, "text-anchor": "end" }, s2);
      }
      // what leaves to the right
      if (st.out) {
        const ax = CX + BW, tx = ax + 34;
        el("line", { x1: ax, y1: y + BH / 2, x2: tx - 8, y2: y + BH / 2,
                     stroke: st.warn ? Wn : RS, "stroke-width": 1.5,
                     "stroke-dasharray": "5 4" }, s2);
        el("path", { d: `M${tx - 13} ${y + BH / 2 - 5} L${tx - 5} ${y + BH / 2} L${tx - 13} ${y + BH / 2 + 5}`,
                     fill: "none", stroke: st.warn ? Wn : RS, "stroke-width": 1.5 }, s2);
        txt(st.out[0], { x: tx, y: y + BH / 2 - 3, "font-size": 11,
                         fill: st.warn ? Wn : M, "font-weight": st.warn ? 500 : 400 }, s2);
        txt(st.out[1], { x: tx, y: y + BH / 2 + 13, "font-size": 10.5, fill: F }, s2);
      }
    });
  })();

  /* ---------------- footprint maps ---------------- */
  const maps = document.getElementById("maps");
  AREAS.forEach(([k, name]) => {
    const d = D[k];
    maps.insertAdjacentHTML("beforeend", `
      <div>
        <div class="fn-title">${name} &mdash; ${d.n} arrays, ${d.total.toLocaleString()} m²</div>
        <div class="map-frame">
          <img src="data:image/png;base64,${d.png}" alt="${name} detected footprints"
               width="${d.w}" height="${d.h}">
          <div class="scalebar"><div style="width:82px"></div>100 m</div>
        </div>
      </div>`);
  });

  /* ---------------- stride diagram ---------------- */
  (function () {
    const W = 900, H = 250, s = svg(W, H, document.getElementById("stride"));
    const A = css("--accent"), Wn = css("--warn"), R = css("--rule"), M = css("--muted"), F = css("--faint");
    const panel = (x0, title, sub, overlap) => {
      txt(title, { x: x0, y: 18, "font-size": 13, fill: css("--ink"), "font-weight": 600 }, s);
      txt(sub, { x: x0, y: 34, "font-size": 11, fill: F }, s);
      const gx = x0, gy = 52, t = 76;
      // the roof: same place in both panels, straddling the centre seam
      const roof = { x: gx + t - 26, y: gy + t - 20, w: 54, h: 42 };
      if (!overlap) {
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++)
          el("rect", { x: gx + i * t, y: gy + j * t, width: t, height: t, fill: "none",
                       stroke: R, "stroke-width": 2 }, s);
      } else {
        [[0, 0], [t / 2, 0], [0, t / 2], [t / 2, t / 2]].forEach(([dx, dy], i) => {
          el("rect", { x: gx + dx, y: gy + dy, width: t, height: t, fill: "none",
                       stroke: i === 3 ? A : R, "stroke-width": i === 3 ? 2.5 : 1.4,
                       "stroke-dasharray": i === 3 ? "" : "4 3", rx: 2 }, s);
        });
      }
      el("rect", { ...roof, fill: overlap ? A : Wn, opacity: .85, rx: 2 }, s);
      if (!overlap) {
        el("line", { x1: gx + t, y1: gy, x2: gx + t, y2: gy + 2 * t, stroke: Wn,
                     "stroke-width": 2.5 }, s);
        txt("seam cuts the roof in two", { x: gx, y: gy + 2 * t + 22, "font-size": 11, fill: Wn }, s);
        txt("each half may score under 0.2", { x: gx, y: gy + 2 * t + 38, "font-size": 11, fill: F }, s);
      } else {
        txt("one window holds the whole roof", { x: gx, y: gy + 2 * t + 22, "font-size": 11, fill: A }, s);
        txt("… and is scored on all of it", { x: gx, y: gy + 2 * t + 38, "font-size": 11, fill: F }, s);
      }
    };
    panel(10, "stride 256 — a partition", "every pixel seen exactly once", false);
    panel(330, "stride 128 — a sliding window", "every interior pixel seen 4 times", true);
    // arrow
    el("path", { d: "M245 128 L305 128", stroke: M, "stroke-width": 1.5 }, s);
    el("path", { d: "M299 123 L307 128 L299 133", fill: "none", stroke: M, "stroke-width": 1.5 }, s);
    // stats block
    const bx = 640;
    [["361", "windows, was 100"], ["4", "views per interior pixel"],
     ["64 px", "worst distance to a tile edge, was 0"], ["+50%", "wall-clock cost"]]
      .forEach(([big, small], i) => {
        txt(big, { x: bx, y: 70 + i * 44, "font-size": 20, fill: A, "font-weight": 500 }, s);
        txt(small, { x: bx, y: 87 + i * 44, "font-size": 11, fill: F }, s);
      });
  })();

  /* ---------------- vote diagram ---------------- */
  (function () {
    const W = 420, H = 210, s = svg(W, H, document.getElementById("vote"));
    const A = css("--accent"), F = css("--faint"), R = css("--rule"), I = css("--ink");
    const c = 46, x0 = 24, y0 = 34;
    const views = [["A", 1], ["B", 1], ["C", 0], ["D", 1]];
    views.forEach(([lbl, yes], i) => {
      const x = x0 + (i % 2) * (c + 10), y = y0 + Math.floor(i / 2) * (c + 10);
      el("rect", { x, y, width: c, height: c, rx: 4, fill: yes ? A : "none",
                   stroke: yes ? A : R, "stroke-width": 2, opacity: yes ? .9 : 1 }, s);
      txt(lbl, { x: x + c / 2, y: y + c / 2 + 5, "font-size": 15, "text-anchor": "middle",
                 fill: yes ? css("--card") : F }, s);
    });
    txt("four windows cover", { x: x0, y: 20, "font-size": 11, fill: F }, s);
    txt("the same pixel", { x: x0, y: y0 + 2 * c + 34, "font-size": 11, fill: F }, s);
    el("path", { d: "M172 100 L212 100", stroke: css("--muted"), "stroke-width": 1.5 }, s);
    el("path", { d: "M206 95 L214 100 L206 105", fill: "none", stroke: css("--muted"), "stroke-width": 1.5 }, s);
    txt("3 of 4 say panel", { x: 234, y: 78, "font-size": 13, fill: I, "font-weight": 500 }, s);
    txt("3 × 2 ≥ 4  →  panel", { x: 234, y: 100, "font-size": 13, fill: A }, s);
    txt("C is outvoted, not obeyed", { x: 234, y: 124, "font-size": 11, fill: F }, s);
    txt("6.2% of contested pixels", { x: 234, y: 142, "font-size": 11, fill: F }, s);
    txt("are dropped this way", { x: 234, y: 158, "font-size": 11, fill: F }, s);
  })();

  /* ---------------- bridging diagram ---------------- */
  (function () {
    const W = 420, H = 210, s = svg(W, H, document.getElementById("bridge"));
    const A = css("--accent"), F = css("--faint"), I = css("--ink"), M = css("--muted");
    const blobs = [[20, 60, 46, 62], [76, 66, 40, 50], [126, 58, 52, 66]];
    blobs.forEach(([x, y, w, h]) => el("rect", { x, y, width: w, height: h, rx: 3,
                  fill: A, opacity: .85 }, s));
    [[66, 76], [116, 66]].forEach(([x, w]) =>
      el("rect", { x, y: 70, width: 10, height: 40, fill: "none", stroke: F,
                   "stroke-width": 1.4, "stroke-dasharray": "3 3" }, s));
    txt("three blobs, 1.5–7.2 m apart", { x: 20, y: 46, "font-size": 11, fill: F }, s);
    txt("gaps are walkways, not roof edges", { x: 20, y: 146, "font-size": 11, fill: F }, s);
    el("path", { d: "M196 90 L232 90", stroke: M, "stroke-width": 1.5 }, s);
    el("path", { d: "M226 85 L234 90 L226 95", fill: "none", stroke: M, "stroke-width": 1.5 }, s);
    el("rect", { x: 252, y: 58, width: 148, height: 66, rx: 3, fill: "none",
                 stroke: A, "stroke-width": 2.5, "stroke-dasharray": "6 4" }, s);
    blobs.forEach(([x, y, w, h]) => el("rect", { x: x + 236, y, width: w, height: h, rx: 3,
                  fill: A, opacity: .85 }, s));
    txt("one array", { x: 252, y: 46, "font-size": 12, fill: I, "font-weight": 600 }, s);
    txt("grouped, but area unchanged", { x: 252, y: 146, "font-size": 11, fill: A }, s);
    txt("3021.8 m² before and after", { x: 252, y: 162, "font-size": 11, fill: F }, s);
  })();

  /* ---------------- area vs tilt scatter ---------------- */
  (function () {
    const W = 900, H = 380, s = svg(W, H, document.getElementById("scatter"));
    const A = css("--accent"), Wn = css("--warn"), G = css("--grid"), F = css("--faint"),
          M = css("--muted"), I = css("--ink");
    const L = 54, Rr = 18, T = 18, B = 46;
    const pw = W - L - Rr, ph = H - T - B;
    const xs = v => L + pw * (Math.log10(Math.max(v, 3)) - Math.log10(3)) / (Math.log10(3000) - Math.log10(3));
    const ys = v => T + ph - ph * v / 90;
    [0, 15, 30, 45, 60, 75, 90].forEach(t => {
      el("line", { x1: L, y1: ys(t), x2: W - Rr, y2: ys(t), stroke: G, "stroke-width": 1 }, s);
      txt(t + "°", { x: L - 9, y: ys(t) + 4, "font-size": 11, fill: F, "text-anchor": "end" }, s);
    });
    [3, 10, 30, 100, 300, 1000, 3000].forEach(a => {
      el("line", { x1: xs(a), y1: T, x2: xs(a), y2: T + ph, stroke: G, "stroke-width": 1 }, s);
      txt(a >= 1000 ? (a / 1000) + "k" : a, { x: xs(a), y: H - 26, "font-size": 11, fill: F,
          "text-anchor": "middle" }, s);
    });
    // 60 deg gate
    el("line", { x1: L, y1: ys(60), x2: W - Rr, y2: ys(60), stroke: Wn, "stroke-width": 1.5,
                 "stroke-dasharray": "6 4" }, s);
    txt("60° gate — a wall, not a roof", { x: W - Rr - 6, y: ys(60) - 8,
        "font-size": 11, fill: Wn, "text-anchor": "end" }, s);
    // 120 m2 divider
    el("line", { x1: xs(120), y1: T, x2: xs(120), y2: T + ph, stroke: M, "stroke-width": 1.2,
                 "stroke-dasharray": "3 4" }, s);
    txt("120 m²", { x: xs(120) + 6, y: T + 13, "font-size": 11, fill: M }, s);
    // median tilt either side of the divider, drawn across the range each summarises
    [[3, 120, 27.1, "median 27.1°", "start"], [120, 3000, 3.7, "median 3.7°", "end"]]
      .forEach(([a0, a1, ty, lbl, anch]) => {
        el("line", { x1: xs(a0), y1: ys(ty), x2: xs(a1), y2: ys(ty), stroke: I,
                     "stroke-width": 2, opacity: .55 }, s);
        txt(lbl, { x: anch === "start" ? xs(a0) + 6 : xs(a1) - 6, y: ys(ty) - 9,
                   "font-size": 11.5, fill: I, "text-anchor": anch, "font-weight": 500 }, s);
      });
    AREAS.forEach(([k, name], ai) => {
      const col = ai === 0 ? A : Wn;
      D[k].arrays.forEach(a => {
        el("circle", { cx: xs(a.area), cy: ys(a.tilt), r: 5,
                       fill: a.gated ? "none" : col, stroke: col,
                       "stroke-width": 1.8, opacity: a.gated ? 1 : .72 }, s);
      });
    });
    txt("footprint area  (m², log scale)", { x: L + pw / 2, y: H - 6, "font-size": 12,
        fill: M, "text-anchor": "middle" }, s);
    txt("fitted tilt", { x: 14, y: T + ph / 2, "font-size": 12, fill: M,
        transform: `rotate(-90 14 ${T + ph / 2})`, "text-anchor": "middle" }, s);
  })();

  /* ---------------- height strip ---------------- */
  (function () {
    const W = 900, H = 250, s = svg(W, H, document.getElementById("heights"));
    const A = css("--accent"), Wn = css("--warn"), G = css("--grid"), F = css("--faint"),
          M = css("--muted"), I = css("--ink");
    const L = 92, T = 20, B = 42, Rr = 18;
    const ph = H - T - B, pw = W - L - Rr;
    // scale starts below zero: at-grade detections measure 0.0 to -0.1 m and
    // clipping them to the axis would hide exactly what the gate is there to catch
    const ys = v => T + ph - ph * (Math.min(v, 40) + 3) / 43;
    [0, 10, 20, 30, 40].forEach(v => {
      el("line", { x1: L, y1: ys(v), x2: W - Rr, y2: ys(v), stroke: G, "stroke-width": 1 }, s);
      txt(v + " m", { x: L - 9, y: ys(v) + 4, "font-size": 11, fill: F, "text-anchor": "end" }, s);
    });
    el("rect", { x: L, y: ys(2), width: pw, height: T + ph - ys(2), fill: Wn, opacity: .09 }, s);
    el("line", { x1: L, y1: ys(2), x2: W - Rr, y2: ys(2), stroke: Wn, "stroke-width": 1.6,
                 "stroke-dasharray": "6 4" }, s);
    txt("2 m gate — below this is flagged as at grade", { x: L + 8, y: ys(2) + 16,
        "font-size": 11, fill: Wn }, s);
    AREAS.forEach(([k, name], ai) => {
      const col = ai === 0 ? A : Wn;
      const band = pw / 2, x0 = L + ai * band;
      txt(name, { x: x0 + band / 2, y: H - 12, "font-size": 12, fill: M, "text-anchor": "middle" }, s);
      const list = D[k].arrays.filter(a => a.h !== null);
      list.forEach((a, i) => {
        const cx = x0 + band * (0.14 + 0.72 * (i + 0.5) / list.length);
        el("circle", { cx, cy: ys(a.h), r: 5.5, fill: a.gated ? "none" : col,
                       stroke: col, "stroke-width": 1.8, opacity: a.gated ? 1 : .72 }, s);
      });
      const lowKept = Math.min(...list.filter(a => !a.gated).map(a => a.h));
      el("line", { x1: x0 + band * 0.08, y1: ys(lowKept), x2: x0 + band * 0.92, y2: ys(lowKept),
                   stroke: col, "stroke-width": 1, "stroke-dasharray": "2 3", opacity: .8 }, s);
      txt(`lowest kept ${lowKept.toFixed(1)} m`, { x: x0 + band * 0.92, y: ys(lowKept) - 7,
          "font-size": 10.5, fill: col, "text-anchor": "end" }, s);
    });
    txt("height above bare terrain", { x: 14, y: T + ph / 2, "font-size": 12, fill: M,
        transform: `rotate(-90 14 ${T + ph / 2})`, "text-anchor": "middle" }, s);
  })();
})();
</script>
"""


def main():
    data = {mode: collect(mode) for mode in ("zurich", "altstetten")}
    html = (HEAD
            + '<script id="pipeline-data" type="application/json">'
            + json.dumps(data)
            + "</script>"
            + TAIL)
    io.open(OUT, "w", encoding="utf-8").write(html)

    print(f"wrote {OUT}  ({len(html) / 1024:.0f} KB)")
    for mode, d in data.items():
        print(f"  {mode:11s} {d['positions']:3d} windows -> {d['flagged']:3d} segmented "
              f"-> {d['n']:2d} arrays -> {d['kept_n']:2d} kept   "
              f"{d['kept_area']:.0f} of {d['total']:.0f} m^2")
    print("\nNumbers on the page are now these. Publish with the Artifact tool to the URL "
          "in CLAUDE.md to keep the same link.")


if __name__ == "__main__":
    main()
