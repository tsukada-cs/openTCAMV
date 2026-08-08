# Changelog

## 2.0.0

### Migration to pyVTTrac v2

pyVTTrac v2.0.0 dropped its Julia (`juliacall`) backend for a native
Fortran core with a pure-numpy Python API, removing the `VTTrac.VTT` /
`.setup()` / `.trac()` object API openTCAMV was built on. This release
migrates to that new API and restructures the core logic into an
installable package:

- **New `opentcamv/` package**: `10_conduct_tracking.py`'s tracking logic
  is now importable (`cli`, `data`, `params`, `output`, `rotation`,
  `tracking`, `aggregate`, `screening`, `records`). `scripts/` are now thin
  CLI drivers; `pip install -e .` makes `opentcamv` importable without
  `sys.path` hacks. `11_concat_flows_along_time.py`, `20_finalize_tracking.py`,
  and `30_plot_velocity2d.py` are unchanged and need no re-installation of
  anything to keep working — they only ever consumed the output NetCDF
  schema, which is unchanged (see below).
- **`--revrot` accepts multiple values**, looped over inside a single
  process instead of one process (and one ~12s Julia cold start) per value;
  `-o` needs an `<omega>` placeholder when more than one value is given, same
  naming convention `20_finalize_tracking.py` already expected. Sharing the
  input read and per-`tid0` time-window slicing (see below) across Omega
  values, plus removing the Julia cold start entirely, makes a 6-Omega
  sample run take seconds rather than minutes.
- **New `--workers`** (OpenMP thread count; `None` = OpenMP default),
  **`--subgrid`** (`paraboloid`/`gaussian`/`none`; `--no_subgrid` remains as
  an alias for `--subgrid=none`), and **`--method`** (`xcor`/`ncov`, not
  reachable from openTCAMV under v1).
- **Per-`tid0` time-window slicing**: pyVTTrac v2's `track()` rescans/copies
  its entire input on every call (cheap once, ruinous per `tid0` over
  hundreds of frames), so tracking now slices the minimal
  `[tid0 - back, tid0 + fwd]` window before calling it, and `--revrot`
  rotates only that window instead of the whole file.
- **Output NetCDF schema is unchanged**: variable names, dimension/coordinate
  names, dtypes, and attrs keys match v1 (verified against the v1-generated
  sample file). `_FillValue` for float variables remains
  `np.finfo(float32).max`, matching v1. CLI argument names are unchanged.
- Target platforms: Linux and macOS (pyVTTrac v2 doesn't support Windows).
  Julia and the `pyVTTrac` git submodule are no longer dependencies —
  `pip install pyVTTrac` is sufficient.

See pyVTTrac's own `CHANGELOG.md` (2.1.0) for the API-level changes that
made this migration possible, including a `Grid` sign-handling fix that
affects any workflow using a descending coordinate axis.

### Fixed

Each of these changes the resulting vectors versus v1 for the affected
option; see "Numerical differences from v1" below.

1. **`--Vd` compared a velocity-difference magnitude (m/s) against `Vd**2`**,
   making the effective threshold 400 m/s at the default `Vd=20` — this
   screen barely ever rejected anything. Now compared against `Vd` directly,
   as documented.
2. **Time axis wrapped to 0 every 24h.** `.dt.seconds` is a timedelta's
   *seconds component* (0-86399), not its total; any run spanning more than
   24h had a corrupted time axis and therefore corrupted velocities. Fixed
   to `.dt.total_seconds()`. Runs under 24h (including the shipped sample)
   are unaffected.
3. **`--out_cthmin` alone was silently a no-op** (both the gate for computing
   cth-along-trajectory and the branch selecting which of min/max to output
   checked `out_cthmax` twice, never `out_cthmin`).
4. **Search radius under-sized for `--itstep > 1`**: a tracking step covers
   `itstep` frames' worth of time, not one, so the radius derived from
   `--Vs` now scales `ref_dt` by `itstep`. `itstep=1` (the default) is
   unaffected.
5. **`--dtlimit` gap check missed intermediate frames** when `--traj_int >
   1`: it diffed the `it_rel`-spaced output time grid rather than every
   frame in the actual tracking window, so a large gap between two
   non-`it_rel` frames went undetected. `traj_int=1` (the default) is
   unaffected.
6. **`--record_initpos` relied on `interp()`'s output dim order matching the
   target's by luck**, assigning into `.data` without an explicit transpose.
   Now transposes explicitly to the target's dims before assigning.
7. Removed dead padding code in the `--out_score_ary` path (the padding
   amount was always zero, since the search radius is now fixed once per
   run rather than potentially varying).

### Numerical differences from v1

Given the same input and arguments, v2 output differs from v1 for two
reasons, both expected and documented above/in pyVTTrac's own changelog:

- The `--Vd` fix (#1 above) makes bothward screening substantially
  stricter — expect noticeably fewer valid vectors with `--Vd` set anywhere
  near its old-effective range.
- pyVTTrac's cross-correlation scoring was numerically stabilized (its own
  CHANGELOG's "xcor/ncov variance computation" fix); this mainly affects
  variables with a large mean offset relative to their variance (e.g. `B03`
  reflectance, 0-120%), where v1 could measurably over- or under-estimate
  scores.

At `sample/sample.sh`'s density (`--xint=1 --yint=1 --Vs=10`, matching v1's
own `sample.sh` and the paper), 8.8% of the tracked (x, y, it) grid points
are valid; `--Vd=20` (v1's value, now finally enforced correctly) and
`--Vd=30` give the same 8.8%, confirming `--Vd` isn't the limiting factor at
this density — most rejections are low-contrast/no-clear-peak points, which
a dense 1 km grid over a partly cloud-free scene inevitably has a lot of.
v1 can no longer be run (its Julia backend is gone) to get a same-density
comparison; the only v1 output on hand when this was written happened to
use a coarser, non-representative `--xint=5 --yint=5 --Vs=40` test grid, at
which valid-vector coverage dropped from 76% to 29-35% (depending on
`--Vd`) versus v1, and where a vector passed screening under both versions,
velocities agreed closely (median difference ~0.0001 m/s, 95th percentile
~0.003 m/s) — confirming the underlying tracking numerics are consistent;
the difference is which vectors survive screening. See
`docs/sample/README.md` for the full account, including why the coarse-grid
run wasn't representative and shouldn't be read as "76% -> 8.8%".

Re-run `sample/sample.sh` to regenerate `sample/*.nc` (not tracked in git;
downloaded/generated locally per `docs/sample/README.md`) and
`sample/outputs/AMVs_it24.png` (tracked) under v2.

---

## 1.0.0 and earlier

See git history prior to the v2.0.0 migration.
