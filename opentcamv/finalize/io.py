"""Loading candidate files into the (ns, omega, it, dim1, dim2[, ...]) working dataset."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..output import AxisNames


def resolve_axis_names(polar: bool) -> AxisNames:
    if not polar:
        return AxisNames("y", "x", "yloc", "xloc", "vy", "vx")
    return AxisNames("r", "a", "rloc", "aloc", "vr", "vt")


def _candidate_filename(ifns_rule: str, ns: int, omega: str) -> str:
    return ifns_rule.replace("<ns>", str(ns)).replace("<omega>", omega)


def _reference_filename(ifns_rule: str, ns_list: "list[int]", omega_strs: "list[str] | None") -> str:
    """The file `flows0` (the reference candidate) is read from -- "first
    `ns`, first `omega`" in every input format (see `load_candidates`)."""
    if "<omega>" in ifns_rule:
        return _candidate_filename(ifns_rule, ns_list[0], omega_strs[0])
    return ifns_rule.replace("<ns>", str(ns_list[0])) if "<ns>" in ifns_rule else ifns_rule


def _open_ns_with_omega_dim(fn: str, omega_strs: "list[str] | None") -> xr.Dataset:
    """Opens one `ns`'s file for the two `omega`-as-a-dimension input
    formats (a single file for all `ns`, or one file per `ns`): if the file
    already has an `omega` dimension (multi-Omega `10_conduct_tracking.py`
    default output), validate it against `--omega` when given, else read it
    from the file, and any `--omega` mismatch is a hard error (never
    silently trust the CLI over the data). If the file has no `omega`
    dimension (a single-Omega `10_` run, which never gets one), synthesize
    a length-1 one from its `revrot` attr, similarly cross-checked against
    `--omega` when given.
    """
    ds = xr.open_dataset(fn, chunks={})
    if "omega" in ds.dims:
        if omega_strs is not None:
            have = np.sort(np.asarray(ds["omega"].values, dtype=float))
            want = np.sort(np.array(omega_strs, dtype=float))
            if have.shape != want.shape or not np.allclose(have, want):
                raise ValueError(
                    f"--omega {omega_strs} does not match the omega values found in {fn!r}: {ds['omega'].values.tolist()}"
                )
        return ds
    value = float(ds.attrs["revrot"])
    if omega_strs is not None and (len(omega_strs) != 1 or not np.isclose(float(omega_strs[0]), value)):
        raise ValueError(f"--omega {omega_strs} does not match the single Omega ({value}) {fn!r} was tracked with")
    return ds.expand_dims(omega=[value])


def _classify_keepvars(flows0: xr.Dataset, keepvars: "list[str]", cth_name: str, dim1: str) -> "tuple[list[str], list[str]]":
    """Returns `(candidate_independent, candidate_dependent)` (F5).

    `cthmax`/`cthmin` (`--out_cthmax`/`--out_cthmin`) and `--record_alongtraj`
    outputs are trajectory-derived, hence genuinely dependent on which
    (omega, ns) candidate produced the trajectory -- these must be read
    per-candidate, not from a single reference file. `--record_initpos`
    outputs (`cth`/`B03`/`B13`/`B14` etc.) and anything derived only from
    them (e.g. `--IRdiff`) are initial-position quantities, independent of
    omega/ns, so a single reference candidate is correct for those (and
    still the only sensible source when no candidate-dependent version
    exists, e.g. `--record_initpos cth` without `--out_cthmax`).

    There's no `--record_alongtraj` flag visible here (`20_` only sees the
    resulting NetCDF, not `10_`'s CLI args), so candidate-dependence is
    inferred structurally: a variable shaped like the trajectory fields
    (`used_vars`' `loc1`/`loc2`, dims `(it, it_rel, dim1, dim2)`) is a
    `--record_alongtraj` output. Checking `it_rel in dims` alone isn't
    enough -- `time2` also has an `it_rel` dim (relative-time-index
    timestamps) but no spatial dims at all, and is candidate-independent
    (the underlying frame times don't depend on omega/ns); requiring `dim1`
    too excludes it correctly.
    """
    candidate_dependent_names = {f"{cth_name}max", f"{cth_name}min"}
    candidate_dependent, candidate_independent = [], []
    for var in keepvars:
        dims = flows0[var].dims
        if var in candidate_dependent_names or ("it_rel" in dims and dim1 in dims):
            candidate_dependent.append(var)
        else:
            candidate_independent.append(var)
    return candidate_independent, candidate_dependent


def load_candidates(
    ifns_rule: str, ns_list: "list[int]", omega_strs: "list[str] | None" = None, exclude=None, cth_name: str = "cth",
):
    """Returns `(flows_org, flows0, names, candidate_independent_keepvars,
    candidate_dependent_keepvars, used_vars)`.

    `flows_org`: dask-backed Dataset of `used_vars` and
    `candidate_dependent_keepvars` (plus `candidate_independent_keepvars`
    copied from `flows0`), with `(ns, omega, ...)`-ordered dims and
    `omega`/`ns` coordinates -- for any number of `ns` values, including one
    (fixes F1: the pre-refactor script's flat, single-`concat_dim=["omega"]`
    loading only worked for a single `ns`; with more it raised `ValueError:
    conflicting sizes for dimension 'omega'`, concatenating every (ns,
    omega) file along one axis instead of nesting them).

    `ifns_rule` accepts any of three formats, auto-detected from its
    placeholders:
    1. A single file with an `omega` dimension (default `10_
       conduct_tracking.py` output for multiple `--revrot` values) --
       `ifns_rule` is a literal path, no placeholders.
    2. One `omega`-dimensioned file per `ns` -- `ifns_rule` has a `<ns>`
       placeholder only.
    3. One file per (`ns`, `omega`) pair (`--split_omega` `10_` output, or
       the pre-refactor script's only supported format) -- `ifns_rule` has
       both `<ns>` and `<omega>` placeholders. `--omega` is required in this
       case (there is no dimension to read it from).

    For formats 1/2, `--omega` (`omega_strs`) is optional -- read from each
    file's `omega` dimension (or synthesized from its `revrot` attr for a
    single-Omega file) when omitted, and cross-checked against it (a hard
    error on mismatch) when given.

    Built via two explicit `xr.concat` passes (per-`ns` over `omega`, then
    over `ns`) for format 3, and one pass (over `ns`; `omega` is already a
    dimension) for formats 1/2 -- not `combine="nested"` over a 2-D nested
    file list, whose axis *labeling* (not just order) turns out to depend on
    which nesting level happens to have length 1, silently mislabeling
    `ns`/`omega` in that case. `chunks={}` on each `open_dataset` keeps
    every array dask-backed, same as the code downstream expects.

    `flows0`: the reference candidate ("first `ns`, first `omega`" in every
    format), opened eagerly -- source of `candidate_independent_keepvars`
    and the IRdiff validity check (F5: both are genuinely independent of
    which candidate produced them, so a single reference candidate is
    correct there, unlike cthmax/cthmin/record_alongtraj).
    """
    legacy = "<omega>" in ifns_rule
    if legacy and not omega_strs:
        raise ValueError("--omega is required when `ifns_rule` contains a `<omega>` placeholder")

    flows0_full = xr.open_dataset(_reference_filename(ifns_rule, ns_list, omega_strs))
    flows0 = flows0_full.isel(omega=0) if "omega" in flows0_full.dims else flows0_full
    names = resolve_axis_names(bool(flows0.attrs["polar"]))
    used_vars = [names.v1, names.v2, names.loc1, names.loc2, "score"]

    # F11 (crash-only fix, no prior golden behavior to preserve): the
    # original script did `used_vars + args.exclude`, which raised
    # `TypeError` whenever `--exclude` was omitted (its argparse default is
    # `None`, not `[]`). Every known real invocation always passes
    # `--exclude`, so this was never observed in practice.
    keepvars = [key for key in flows0.data_vars.keys()]
    for var in used_vars + list(exclude or []):
        if var in keepvars:
            keepvars.remove(var)
    candidate_independent_keepvars, candidate_dependent_keepvars = _classify_keepvars(
        flows0, keepvars, cth_name, names.dim1
    )
    per_candidate_vars = used_vars + candidate_dependent_keepvars

    per_ns = []
    for ns in ns_list:
        if legacy:
            per_omega = [
                xr.open_dataset(_candidate_filename(ifns_rule, ns, omega), chunks={})[per_candidate_vars]
                for omega in omega_strs
            ]
            ds = xr.concat(per_omega, dim="omega")
            ds = ds.assign_coords(omega=np.array(omega_strs).astype(float))
        else:
            fn = ifns_rule.replace("<ns>", str(ns)) if "<ns>" in ifns_rule else ifns_rule
            ds = _open_ns_with_omega_dim(fn, omega_strs)[per_candidate_vars]
        per_ns.append(ds)
    flows_org = xr.concat(per_ns, dim="ns")
    flows_org["ns"] = np.array(ns_list)

    for var in candidate_independent_keepvars:
        flows_org[var] = flows0[var]

    return flows_org, flows0, names, candidate_independent_keepvars, candidate_dependent_keepvars, used_vars
