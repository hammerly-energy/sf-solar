# SF Solar Nowcast — Roadmap From Zero

Companion to `sf-solar-nowcast-architecture.md`. That doc is the **destination**. This one is the **route**.

---

## The core idea

**Build a working website in ~3 weekends, then upgrade one part at a time.**

Every level below ends with a real, deployed, public URL. You never have a half-finished thing. Each level swaps one component for a better one, and the site keeps working the whole way.

This matters more for you than for an experienced engineer. Momentum is the scarce resource — a live URL you can show someone after three weekends is what keeps a nights-and-weekends project alive. A perfect satellite pipeline that isn't connected to anything is what kills one.

---

## Level 0 — Setup (one weekend, and it's the least fun weekend)

You're on macOS, which makes this easier. In order:

| # | What | Why |
|---|---|---|
| 1 | **Homebrew** (`brew.sh`) | Mac package manager; everything else installs through it |
| 2 | **Python via `uv`** (`astral.sh/uv`) | Modern, fast, avoids the classic "which Python am I using" nightmare. Don't use the system Python. |
| 3 | **VS Code** | Where you'll read code. You'll read more than you write. |
| 4 | **Git + a GitHub account** | Version control. Non-negotiable — it's your undo button and your web host. |
| 5 | **Claude Code** (`npm install -g @anthropic-ai/claude-code`) | The thing that makes this feasible |

**Deliverable:** a repo called `sf-solar` on GitHub with a README in it, created from your terminal. That's it. If you can do that, your environment works.

Don't skip git. Every beginner wants to, and every beginner who does eventually destroys three days of work with one bad edit.

---

## How to actually use Claude Code on this

You've seen engineers use it. A few things that aren't obvious:

**Work in a repo, not in chat.** Open your terminal, `cd` into `sf-solar`, run `claude`. It now sees your files and can edit them directly. This is fundamentally different from copy-pasting code out of a chat window.

**Write a `CLAUDE.md` in your repo.** It's a file Claude Code reads automatically every session. Put in it: what the project is, that you're not a software engineer, that you want explanations alongside changes, and the key project decisions (GOES-18 not 16, pvlib, no backend). This stops you re-explaining context every session.

**Ask it to explain, not just to do.** "Add the clear-sky calculation" gets you working code you don't understand. "Add the clear-sky calculation, and walk me through what each part does" gets you working code and a slightly better you. The second one compounds.

**Commit constantly.** Ask Claude Code to commit after every working change. When something breaks — and it will — you can rewind.

**The failure mode you specifically need to guard against:** Claude Code will happily write you a complete, beautiful, confidently-wrong solar model. Physics bugs don't crash. A sign error in the azimuth convention or a UTC-vs-local mixup produces a smooth, plausible curve that's just *wrong*. You cannot catch this by reading the code — you don't have the background yet, and honestly neither do most engineers.

You catch it by **plotting against real measured data early and often.** This is why Level 1 is what it is. Validation isn't a step at the end for you; it's your only defense.

---

## The ladder

### Level 1 — One system, one day (a weekend)
Pick a single hypothetical SF rooftop: 5 kW, 20° tilt, facing south. Use `pvlib` to compute its clear-sky output for one sunny day. Plot it.

**You'll learn:** running a Python script, the shape of a solar day, what pvlib actually does.
**Deliverable:** a PNG of a smooth bell curve peaking around solar noon.
**Sanity check:** peak should be roughly 70–80% of nameplate; daily total in the ballpark of 25–35 kWh in summer. If you get 500 kWh, something's wrong — and now you know how to tell.

### Level 2 — Your first website (a weekend)
One HTML page. A chart of that curve. Deployed to GitHub Pages at `yourname.github.io/sf-solar`.

**Deliverable:** a URL you can text someone.
This is the psychologically important one. Do not skip ahead past it.

### Level 3 — The real fleet (1–2 weekends)
Download the Interconnected Project Sites dataset, filter to SF, sum the capacity. Site now shows citywide clear-sky MW instead of one fake rooftop.

**Deliverable:** "San Francisco's solar fleet would produce X MW right now under clear skies." Already genuinely interesting, and nobody else has this page.
**You'll learn:** pandas, messy real-world data, geocoding.

### Level 4 — Clouds, the cheap way (2–3 weekends) ← **this is your MVP**
**Do not build the GOES pipeline yet.** Use a free irradiance API — Solcast's hobbyist tier gives 5-minute GHI/DNI nowcasts. Pull it for a handful of points across SF, apply to your fleet.

**Deliverable:** a live, actually-correct estimate of current SF solar generation. This is a complete, honest, useful product. If you stopped here you'd have a real portfolio piece.

**Why not build GOES first:** it's the hardest stage, and building it before you have anything to plug it into means months of work with no visible progress. Do it as an *upgrade* to a working system, where you can A/B it against the API and prove it's better. That comparison is a far stronger portfolio story than "I processed satellite data."

### Level 5 — Your own satellite pipeline (1–2 months)
Now replace the API with GOES-18 + cloud index, per §4 of the architecture doc. You have a working baseline to beat and a site to plug into.

### Level 6 — Validation, forecasting, and the fog model
Sections 6–8 of the architecture doc. This is where the project stops being a tutorial and starts being *yours*. The marine-layer burn-off problem is the interesting, defensible, nobody-else-has-done-this piece — and you'll arrive at it with the skills to actually attack it.

---

## Website architecture (the part that saves you months)

You do **not** need a backend server. Do this instead:

```
GitHub Actions (cron, every 15 min)
        ↓  runs your Python script
   writes data/latest.json
        ↓  commits to the repo
   GitHub Pages serves the static site
        ↓
   Browser fetches latest.json, draws the chart + map
```

**Why this is the right call:** zero cost, zero servers, nothing to maintain or get hacked, no database, no deployment pipeline. Your Python stays plain Python. The website is a static HTML file. This is the single biggest complexity reduction available to you, and plenty of production data sites work exactly this way.

**Frontend:** plain HTML + JavaScript. Chart.js for the time series, MapLibre for the neighborhood map. **Skip React** — you don't need it, and the tooling around it will cost you a weekend you don't have.

**Two known gotchas:**
- GitHub Actions scheduled jobs can be delayed by several minutes under load, and get auto-disabled after 60 days of repo inactivity. Fine for a 15-minute-cadence project; just know it.
- Committing every 15 minutes makes for a noisy repo history. Write data to a separate `data` branch to keep your main history clean.

If you outgrow it later, Cloudflare Workers or Vercel are the natural next step. You probably won't outgrow it.

---

## Scope cuts for v1

From the architecture doc, **defer all of this**:

| Defer | Why |
|---|---|
| **LiDAR horizon profiles** (§3.2) | Hardest engineering in the whole doc. Real accuracy cost, but a wrong-by-15% site beats a nonexistent site. |
| **Archetype clustering** (§3.3) | Premature optimization. SF's fleet is probably small enough to simulate directly. Measure first. |
| Parallax correction | Only matters once you're processing satellite data yourself (Level 5) |
| Advection forecasting | Level 6 |
| Separate DNI/DHI modeling | The API gives it to you at Level 4 |

Write these in your README as "known limitations." Being explicit about what your model *doesn't* do is a mark of someone who understands their model — it reads as rigor, not as apology.

---

## Realistic timeline

At a steady evenings-and-weekends pace:

- **Level 0–2:** 2–3 weekends → you have a live website
- **Level 3–4:** 1–2 months → you have a real, working product
- **Level 5:** 1–2 months → you have the interesting technical piece
- **Level 6:** open-ended → you have something novel

If Level 4 is all you ever finish, that's still a genuinely good project that most people couldn't build.

---

## First three concrete actions

1. Install Homebrew, `uv`, VS Code, git, and Claude Code. Make an empty `sf-solar` repo on GitHub.
2. Write a `CLAUDE.md` in it (see above). Paste both of these docs into a `docs/` folder in the repo so Claude Code has the full context.
3. Open Claude Code in that folder and say: *"Read docs/sf-solar-roadmap.md. Help me do Level 1. I'm not a software engineer — explain as you go."*

That's the whole on-ramp.
