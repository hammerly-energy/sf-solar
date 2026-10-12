# SF Solar

Goal: estimate how much electricity San Francisco's rooftop solar is producing right now, and show it on a website.

Live site: https://hammerly-energy.github.io/sf-solar/ (right now it shows one example rooftop, not the city).

Method: take every solar system in the city, work out how much sun hits each one, reduce that for clouds, and add it up.

This README records what happened, in order, including the parts that failed. Detailed notes are in [`CLAUDE.md`](CLAUDE.md) and [`docs/`](docs/). Technical terms are explained in the [glossary](#glossary) at the bottom.

---

## The whole project in one picture

Green = worked. Red = roadblock. Grey = not started.

```mermaid
flowchart TD
    A["Aug 4: Level 1<br/>Clear-sky curve for one<br/>example rooftop"]:::ok
    B["Aug 5: Level 2<br/>Website showing that curve"]:::ok
    R1["Roadblock: GitHub Pages only serves<br/>the repo root or /docs, not site/"]:::bad
    B1["Fix: deploy with a GitHub Actions workflow"]:::ok
    C["Aug 16: Level 3<br/>Load the real SF fleet<br/>13,485 systems, 81.1 MW"]:::ok
    D["Side quest: find individual<br/>buildings and measure roof angles<br/>from aerial photos"]:::detour
    R2["Roadblock: no usable SF aerial photos<br/>(too coarse, licensed, or banned to scrape)"]:::bad
    E["Move the experiment to Zurich,<br/>which has open 10 cm photos and laser height data"]:::detour
    F["Pipeline works end to end,<br/>but many wrong-but-plausible numbers"]:::bad
    G["Sept 5: review of everything<br/>plus a sensitivity test"]:::detour
    H["Finding: SF already has angles for<br/>95.9% of its capacity. The detour cannot<br/>meaningfully improve SF's number."]:::bad
    I["Detour kept, with a new purpose:<br/>cities that have no angle data.<br/>SF is the demo."]:::detour
    J["Next: simulate every SF system,<br/>sum to a citywide MW curve,<br/>update the site"]:::todo
    K["Later: real cloud data,<br/>scheduled updates, forecast"]:::todo

    A --> B --> R1 --> B1 --> C
    C --> D --> R2 --> E --> F --> G --> H --> I
    C -.->|"main path, unblocked"| J
    I -.->|"does not block it"| J
    J --> K

    classDef ok fill:#d9f2e0,stroke:#2e7d4f,color:#111
    classDef bad fill:#f8d7d7,stroke:#b03030,color:#111
    classDef detour fill:#e3e8f5,stroke:#4a5fa0,color:#111
    classDef todo fill:#eeeeee,stroke:#888,color:#111
```

---

## Roadblock 1: finding individual buildings in San Francisco

The fleet data only says which ZIP code a system is in. I wanted exact buildings and roof angles, so I tried four ways to get them.

```mermaid
flowchart TD
    Q["How do we find which<br/>buildings have solar?"]
    A1["Geocode the fleet data"]
    X1["Dead end: the data has<br/>no street addresses"]:::bad
    A2["Use SF building permits"]
    X2["Dead end: they cover only<br/>8 to 12% of systems, and<br/>capacity is buried in free text"]:::bad
    A3["Use published DeepSolar results"]
    X3["Dead end: only neighbourhood totals,<br/>never individual locations"]:::bad
    A4["Run DeepSolar's model on aerial photos"]
    X4["Dead end: the model needs ~24 cm photos.<br/>Free SF photos are 60 cm.<br/>Better ones are licensed or banned to scrape."]:::bad
    T["Tried the 60 cm photos anyway<br/>on a known solar farm"]
    X5["Model scored it under 0.11 out of 1.<br/>Unclear whether the cause is blurry photos<br/>or a model that never saw ground-mounted panels."]:::bad
    Z["Decision: test the method in Zurich instead"]:::ok

    Q --> A1 --> X1
    Q --> A2 --> X2
    Q --> A3 --> X3
    Q --> A4 --> X4 --> T --> X5 --> Z

    classDef ok fill:#d9f2e0,stroke:#2e7d4f,color:#111
    classDef bad fill:#f8d7d7,stroke:#b03030,color:#111
```

---

## Roadblock 2: the Zurich pipeline kept giving believable wrong answers

Zurich publishes free aerial photos and free laser-scanned height maps, so the method could be tested there. It worked, but nothing crashed when it was wrong. Each bug below produced a normal-looking number and was found only by drawing a picture.

```mermaid
flowchart TD
    S["Pipeline: spot panels in photo,<br/>outline them, fit a flat plane<br/>through the roof heights underneath,<br/>read off tilt and direction"]

    B1["Bug: used 1 of 8 height tiles"]:::bad
    B2["Bug: fitted a rectangle around the panels,<br/>not the panels themselves.<br/>Found when a 3D plot showed the fit<br/>including a 20 m drop off the roof"]:::bad
    B3["Bug: one plane through separate patches<br/>gave a 73.9 degree roof, i.e. a wall"]:::bad
    B4["Bug: stray points (antennas, vents)<br/>dragged the fit"]:::bad
    B5["Bug: removing outliers without a limit<br/>threw away up to 35% of a roof<br/>to force a tidy answer"]:::bad
    B6["Bug: rows at different angles<br/>were averaged into one meaningless plane"]:::bad

    F1["Fix: combine all tiles"]:::ok
    F2["Fix: fit only the outlined pixels"]:::ok
    F3["Fix: fit each connected patch separately"]:::ok
    F4["Fix: drop outliers and refit"]:::ok
    F5["Fix: cap removal at 20%;<br/>past that, keep the untrimmed fit"]:::ok
    F6["Fix: fit one plane, then fit the<br/>leftover points as a new plane"]:::ok

    S --> B1 --> F1 --> B2 --> F2 --> B3 --> F3 --> B4 --> F4 --> B5 --> F5 --> B6 --> F6

    classDef ok fill:#d9f2e0,stroke:#2e7d4f,color:#111
    classDef bad fill:#f8d7d7,stroke:#b03030,color:#111
```

Two checks that found nothing: I tested whether the photos and height maps were misaligned (no consistent shift) and whether shrinking the photos blurred away the panels (no effect on scores). Both are ruled out.

---

## Roadblock 3: the September 5 review

I went back through the whole detour and tried to break it. Several earlier claims did not survive.

```mermaid
flowchart TD
    R["Review: re-check every number"]

    P1["Height data was from 2018,<br/>photos were from 2025"]:::bad
    P1a["Fixed. One roof's tilt moved<br/>from 8.0 to 12.6 degrees.<br/>Every earlier angle was wrong."]:::ok

    P2["Small outlines gave steeper tilts<br/>than large ones. Correlation -0.705:<br/>the tilt tracks outline size,<br/>not the roof"]:::bad
    P2a["Still present after later fixes.<br/>Impossible roofs of 60.9 and 66.3 degrees remain."]:::bad

    P3["Each 'detection' was a photo tile,<br/>not a building"]:::bad
    P3a["Fixed. Tiles merged, nearby pieces joined,<br/>and tiles now overlap so seams stop<br/>hiding panels. Arrays that were cut short:<br/>3614 m2 down to 61 m2 in one test area"]:::ok

    P4["Second test area was chosen for<br/>'varied roof pitches'"]:::bad
    P4a["After the date fix every roof read<br/>0 to 1.4 degrees. The variety came<br/>from tiles cutting arrays in half."]:::bad

    P5["Claimed false-positive rate"]:::bad
    P5a["Withdrawn. I labelled my own test data<br/>with the model's outlines showing,<br/>and the sample was too small anyway.<br/>Rate is now: unmeasured."]:::bad

    R --> P1 --> P1a
    R --> P2 --> P2a
    R --> P3 --> P3a
    R --> P4 --> P4a
    R --> P5 --> P5a

    classDef ok fill:#d9f2e0,stroke:#2e7d4f,color:#111
    classDef bad fill:#f8d7d7,stroke:#b03030,color:#111
```

One false positive is documented: the model's most confident Wiedikon detection (0.95 out of 1) is four skylights, while two clearly real arrays scored 0.35 and 0.26.

---

## The result that changed the plan

I ran fake errors through the full SF fleet. Question: if the roof angles for some systems were wrong, how wrong would the citywide total be?

```mermaid
flowchart LR
    Q["SF fleet data"] --> A["95.9% of capacity<br/>already has measured<br/>tilt and direction"]
    Q --> B["4.1% is missing<br/>and needs estimating"]
    B --> C["Worst case: all of that 4.1%<br/>is 20 degrees off in tilt AND<br/>facing the opposite way"]
    C --> D["Citywide error:<br/>0.73% at summer noon,<br/>3.35% on a winter morning"]:::bad
    D --> E["Conclusion: the aerial-photo method<br/>cannot meaningfully improve SF's number"]:::bad
    E --> F["It stays as a method for cities<br/>without this data, with SF as the demo"]:::ok

    classDef ok fill:#d9f2e0,stroke:#2e7d4f,color:#111
    classDef bad fill:#f8d7d7,stroke:#b03030,color:#111
```

Two further results from the same test:

- Errors that all lean the same way hurt far more than random errors that cancel out. A 20-degree tilt bias changes winter 9 am output by 48.6%. The same amount as random scatter changes it by 13.2%. The photo method's error is the leaning kind.
- 30 MW of flat systems store "direction = 0" as a placeholder, not as "facing north". Treating it as north would overstate every error.

---

## Where things stand

| Piece | Status |
|---|---|
| One-rooftop curve and website | Done, live |
| SF fleet loaded, cleaned, mapped by ZIP | Done |
| Simulate every SF system and sum to a citywide curve | **Not done. Next.** |
| Update the website to show the citywide curve | Not done |
| Cloud data, scheduled updates (`data/latest.json`), forecast | Not started |
| Zurich photo pipeline | Works on two test areas. Error rates unmeasured. Tilt numbers biased for small arrays. |

Known open problems in the photo pipeline:

- Missed panels (false negatives) have never been counted, and for a citywide total they are the error that matters most.
- The panel outlines cover only strips of some large real arrays.
- The function that produces tilt and direction has no automated test.
- A denser laser point cloud (about 7 times more points) is available and unused.

---

## Rules adopted along the way

These came from specific mistakes:

- **Plot every physics change** and check it against a known range. Most bugs here only showed in a picture.
- **Never set a threshold using a known wrong example.** Doing so gives a number that cannot fail. This happened once and was redone.
- **Keep known false positives out of the work that measures rows.** Mixing the two questions ("where are the rows?" and "is this solar at all?") produced two wrong conclusions in one session.
- **No automatic commits.** They made the history hard to follow.
- **One thing per session turn.**

---

## Glossary

- **Clear-sky output:** electricity a system would make with no clouds.
- **MW / kW:** power. kWh is energy, which is power over time.
- **Nameplate / DC / AC:** nameplate is the rated size of the panels (DC). The inverter converts to AC and caps what reaches the grid, so peak AC is about 70 to 85% of nameplate.
- **Tilt:** roof angle from flat. **Azimuth:** compass direction the panels face (180 = south).
- **LiDAR / height map (DSM):** laser scan giving the height of everything seen from above, including roofs and rooftop equipment.
- **Plane fit:** finding the flat surface that best matches a set of height points.
- **Outlier:** a point that doesn't belong to the surface, such as an antenna.
- **Segmentation:** outlining which pixels of a photo are panels.
- **Classifier:** the model that says whether a photo tile contains solar. Scores run 0 to 1.
- **False positive / false negative:** reported solar that isn't there / solar that is there but was missed.
- **Correlation (ρ):** how closely two measurements move together, from -1 to 1. Here, -0.705 between outline size and tilt means the two are tied together when they should not be.
- **Sensitivity test:** deliberately adding errors to the inputs and watching how much the output moves.

---

## Running it

SF fleet and the error test:

```bash
uv run python scripts/level3_fleet.py
```

```bash
uv run python scripts/level3_geometry_sensitivity.py
```

Geometry tests, which run in seconds:

```bash
uv run --isolated --with numpy --with scipy --with rasterio python scripts/test_roof_geometry.py
```

Zurich photo pipeline (needs about 1.2 GB of model weights in `data/raw/deepsolar_weights/`; use `altstetten` instead of `zurich` for the second area):

```bash
uv run --isolated --with "torch>=2.3" --with transformers==4.46.3 --with rasterio \
  --with pystac-client --with planetary-computer --with matplotlib --with requests --with scipy \
  python scripts/level3_cv_detection.py zurich
```

## Layout

```
sf-solar/
├── CLAUDE.md       working rules and detailed findings
├── docs/           original roadmap and architecture
├── scripts/        fleet loading, simulation checks, Zurich pipeline, tests
├── data/           outputs (data/raw is not committed)
└── site/           the website
```

## Data sources

- California DG Stats, Interconnected Project Sites
- SWISSIMAGE, swissSURFACE3D, swissALTI3D, ©swisstopo
- DeepSolar-3M model
- pvlib-python for solar physics
