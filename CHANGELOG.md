# Changelog

## 2.0.0

### Candidate-selection layer redesign

`20_finalize_tracking.py`'s candidate-selection logic (picking the best
Omega/`ns` candidate per grid point from `10_`'s outputs) moves into an
`opentcamv.finalize` package with swappable selectors, `10_` gains a
single-file default output, and `11_concat_flows_along_time.py`'s logic
moves into `opentcamv.concat`. None of this changes `10_`'s own tracking
math; see "Fixed" below for the parts of `20_` that do change numbers.

- **`opentcamv.finalize` package** (`io`/`window`/`masking`/`candidates`/
  `selectors/`): `20_finalize_tracking.py` is now a thin CLI driver over it,
  matching how `10_conduct_tracking.py` already related to `opentcamv`.
  `selectors/iterative_median.py` is the current (default) selection
  algorithm, ported with no behavior change from the pre-refactor script.
  `selectors/base.py`/`selectors/relaxation.py` add a `Selector` interface
  and shared building blocks (`neighborhood_reduce`/`candidate_distance`/
  `iterate_until_stable`) for a future relaxation-labeling selector
  (`--selector relaxation`, not implemented yet — raises
  `NotImplementedError`; `--selector iterative_median` remains the default
  and only working option).
- **`10_conduct_tracking.py` defaults to a single output file** holding all
  `--revrot` values along a new `omega` dimension, instead of always
  splitting into one file per Omega. `--split_omega` restores the old
  per-file behavior (still needs an `<omega>` placeholder in `-o`). A
  single `--revrot` value is unaffected either way — it never gets an
  `omega` dimension, matching the pre-existing schema exactly. All other
  variable/dimension names and attrs keys are unchanged.
- **`20_finalize_tracking.py` accepts multiple `ns` values** (`--ns 7 9`),
  formally supporting multiple template sizes as candidates at the same
  point (previously crashed — see "Fixed"). It also accepts any of three
  input formats for `ifns_rule`, auto-detected from its `<ns>`/`<omega>`
  placeholders: a single `omega`-dimensioned file (matches `10_`'s new
  default), one such file per `ns` (`<ns>` only), or the legacy one file
  per `(ns, omega)` pair (`<ns>` + `<omega>`, matches `--split_omega`).
  `--omega` is optional for the first two (read from the file) and
  cross-checked against it when given.
- **`11_concat_flows_along_time.py`'s role is now explicitly "memory
  constraints / cluster distribution", not speed**: `10_` already tracks
  every Omega and `ns` within one process, so there's no per-Omega
  parallelism left to reassemble here. For within-process speed, use
  `10_`'s `--workers` (OpenMP thread count). The concatenation logic itself
  (along `time`, unaffected by the `omega`/`ns` dimension changes above) is
  unchanged, just moved into `opentcamv.concat` behind a thin CLI.
- **`10_`'s combined-omega output no longer duplicates `--record_initpos`
  outputs (e.g. `B03`/`B13`/`B14`/`cth`) or `time2` across the `omega`
  dimension.** They're initial-position/frame-timestamp quantities,
  genuinely independent of the rotation applied *during* tracking, so
  they're written once instead of `n_omega` times over — meaningful when
  `--record_initpos` includes full-resolution imagery. `cthmax`/`cthmin`/
  `--record_alongtraj` outputs *do* depend on Omega (trajectory-derived)
  and still get the `omega` dimension.

### Fixed

Each of these changes `20_finalize_tracking.py`'s output for the affected
option/scenario, except the last two, which are `10_conduct_tracking.py`
schema fixes (present since the v2.0.0 migration; the "dtypes match v1"
claim in that release's notes did not hold for these fields).

1. **`--ns` with more than one value crashed** (`ValueError: conflicting
   sizes for dimension 'omega'`) — the old loading concatenated every
   `(ns, omega)` file along a single axis instead of nesting them. Multiple
   `ns` values are now candidates at every point, same as multiple Omega
   values.
2. **`--cthmin`/`--cthmax` screening used a single reference candidate's
   cloud-top height for every candidate**, even though it's a
   trajectory-derived quantity that genuinely depends on which Omega/`ns`
   produced it. Now evaluated per-candidate. On the shipped sample
   (`--cthmax 10`), this changes which candidate is accepted at 0.63% of
   points that resolve either way (close to the pre-fix measurement of
   0.55% from comparing two different single-reference choices), and the
   reported `cthmax` itself (now always the *selected* candidate's own
   value, not always the reference's) changes at ~75% of points —
   direction varies per point: some points previously accepted on a good
   reference's cth are now correctly rejected on their own bad cth, and
   vice versa. `--record_alongtraj` outputs are now loaded per-candidate
   the same way; `--record_initpos` outputs and `--IRdiff` remain
   reference-only (correctly — they're initial-position quantities,
   independent of Omega/`ns`).
3. **The median-filter window's x/y widths were swapped** (`--xw`'s width
   applied to the y axis, `--yw`'s to the x axis). Invisible under the
   default `--xw == --yw` (both fall back to `--Hw`) and `--xint ==
   --yint`, which is every existing invocation on record — no change at
   default settings.
4. **`--rw`/`--aw` (the polar-grid median-filter window) were defined but
   never used** — `--polar` runs got the Cartesian `--xw`/`--yw` (km-based)
   window applied to the `r`/`a` axes regardless, and since `a` is in
   radians, the azimuth window was meaningless. `--polar` now sizes its
   window from `--rw`/`--aw` as documented. No effect outside `--polar`.
5. **`--priority dangv` discarded the physical Omega coordinate**,
   overwriting it with a plain integer index right after using its
   physical value — `--out_final_omega` then reported indices, not rad/s,
   and `--omega_stri` (striation masking, which reads the Omega coordinate
   later in the pipeline) compared a physical rad/s threshold against those
   same indices. `--out_final_omega` now reports physical Omega values
   under `--priority dangv` too, and `--omega_stri` + `--priority dangv`
   together now behave as documented instead of silently comparing
   unrelated quantities.
6. **`--exclude` omitted raised `TypeError`** and **`--polar` crashed
   unconditionally** in `20_finalize_tracking.py` (a hardcoded `"vx"`
   lookup in the validity mask). Every known real invocation always passed
   `--exclude`, and `--polar` had never been run successfully, so neither
   was previously observed.
7. **`10_`'s `--revrot` restoration and `--record_initpos`/
   `--record_alongtraj`/`--out_cthmin`/`--out_cthmax` silently promoted
   `vx`/`vy`/`xloc`/`yloc`/`cth`/`cthmax`/`cthmin`/`B03`/etc. from float32
   to float64** — adding a rotation-derived quantity (itself float64,
   coming from float64 coordinate arrays) to a float32 field, or
   `xarray`/`scipy` interpolation (which always computes and returns
   float64 regardless of the source's dtype), silently widens the whole
   array. Any Omega other than exactly `0.0` triggered the first; every
   `--record_initpos`/`--record_alongtraj`/`--out_cthmin`/`--out_cthmax`
   invocation triggered the second. Both are now cast back to float32
   after computing, doubling on-disk size for these fields versus the
   affected runs (precision-only — values are unchanged to well within the
   1 m/s+ tolerances tests and typical use hold vectors to).
8. **`stf`/`stb` (tracking status codes) were stored as `int16`** despite
   only ever holding `-10` (fill) or `0`-`11` (see the flags list in their
   `long_name` attrs) — now `int8`, which comfortably covers that range.

`--out_final_ns` still does not output anything (`final_ns` is computed
internally but never attached to the final output) — pre-existing,
unrelated to this redesign, left as-is.

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

Requires **pyVTTrac >= 2.2**. See pyVTTrac's own `CHANGELOG.md` (2.1.0 and
2.2.0) for the API-level changes that made this migration possible,
including a `Grid` sign-handling fix that affects any workflow using a
descending coordinate axis, an out-of-bounds template read at the low edge
of the domain that could silently corrupt affected templates, and a fix for
`workers` leaking process-wide OpenMP state between calls. The out-of-bounds
fix does not change this repository's sample output (its seeds sit far from
the domain edge; verified bit-identical), but it can change results for
runs that place templates within a few pixels of the `x`/`y` minimum.

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
8. **`vxhw`/`vyhw` output attributes were wrong under `--hs`**: `--hs` sets
   the pixel search radius directly and `--Vs` is never consulted, but the
   attributes still reported `--Vs`. They now report the search velocity
   the radius actually covers (via pyVTTrac's
   `velocity_from_search_radius()`). Metadata only — no effect on the
   derived vectors, and no change when `--hs` is not used.

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
