# Pitch Scout — Shape Optimizer

**Live app:** _(add your Streamlit Cloud URL here once deployed)_

Pitch Scout finds the optimal movement profile for a pitcher's arsenal given the
parts of their delivery they can't easily change. Instead of asking a pitcher to
throw harder or completely remold their mechanics, it answers a more useful
question: **given your release point, extension, and velocity, what pitch shape
would play best — and how does it tunnel with the rest of your arsenal?**

---

## What it does

- **Shape optimization.** For any pitch, it searches the space of achievable
  movement (horizontal and vertical break) and finds the shape that maximizes a
  Stuff+-style run-value grade, constrained to what's realistic for that
  pitcher's mechanics.
- **Mechanically-grounded bounds.** "Achievable" isn't hand-waved. Each pitcher's
  optimization is bounded by what *mechanically comparable* pitchers actually
  throw — matched on release point, extension, and spin characteristics, with
  opposite-handed comps mirrored in.
- **Tunneling analysis.** Pitches don't work in isolation. The app evaluates
  which pitches tunnel together — sharing an early flight path before diverging —
  using research-grounded criteria (velocity differential and release-point
  proximity), and flags shapes that improve *both* raw grade and tunnel
  deception.
- **3D flight-path visualization.** An interactive view of every pitch's
  trajectory from release to the plate, with the ~23-ft "decision point" where a
  hitter commits, plus a head-on cross-section showing pitch separation at that
  point.
- **Custom data upload.** Bring your own Trackman or Rapsodo CSV. The app parses
  it, finds MLB mechanical comps, and grades the arsenal. Values can be
  hand-edited in an override panel, and everything — grades, bounds, flight
  paths, tunneling — recomputes consistently.

## How it works

The grading engine is an ensemble of gradient-boosted models (one per
pitch-type × handedness matchup) trained on pitch-level run value from public
Statcast data. Features include velocity, induced movement, release geometry,
spin characteristics, fastball differentials, and — for fastballs and sinkers —
vertical approach angle, which the app derives from the pitch's own trajectory so
it stays physically consistent as the shape changes.

The model is **pluggable**: the app talks to it through a single interface
(`StuffModelInterface`), so an organization can swap in its own in-house run-value
model without touching the rest of the app.

## Tech

Python · Streamlit · XGBoost · NumPy / pandas / SciPy · custom HTML5-canvas
visualizations. The trajectory math (plate-crossing position, approach angle,
Magnus-derived accelerations) is validated against real tracking data to within a
fraction of a degree / a thousandth of a foot.

## Running locally

```bash
pip install -r requirements.txt
streamlit run shape_optimizer.py
```

The app expects the trained model artifacts in `data/`. See `sources.md` for data
provenance and methodology references.

## Project layout

| File | Purpose |
|------|---------|
| `shape_optimizer.py`     | Main Streamlit app: UI, optimizer, visualizations |
| `stuff_model_interface.py` | Pluggable run-value model interface |
| `tunneling.py`           | Trajectory math, VAA, tunnel-pair logic |
| `csv_import.py`          | Trackman / Rapsodo CSV parser |
| `synthetic_comps.py`     | Mechanical comp finding + bounds for uploads |
| `data/`                  | Trained models, comp profiles, norm tables |
| `sources.md`             | Data sources and methodology references |

## Notes & limitations

- Grades are calibrated to MLB run value, so an MLB-trained model evaluating
  sub-MLB-velocity pitches extrapolates outside its training range — the absolute
  Stuff+ number for amateur arms should be read with that caveat. The *shape and
  tunneling guidance* is robust regardless; the calibration is exactly what an
  in-house model swap fixes.
- This is a decision-support tool, not a prescription. It surfaces what the data
  suggests; pitch design always belongs in the hands of coaches and players.

---

_Built as a demonstration of applied baseball analytics — feature engineering,
model serving, physics-based validation, and interactive visualization in one
end-to-end tool._
