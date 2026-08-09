"""cth / IRdiff / score-threshold / striation masking."""

from __future__ import annotations

import numpy as np
import xarray as xr


def squeeze_cth(flows: xr.Dataset, cth_name: str, minmax: str = "max") -> xr.DataArray:
    if minmax == "max":
        if f"{cth_name}max" in flows:
            return flows[f"{cth_name}max"]
        elif "it_rel" in flows[cth_name].dims:
            return flows[cth_name].max("it_rel")
        return flows[cth_name]
    elif minmax == "min":
        if f"{cth_name}min" in flows:
            return flows[f"{cth_name}min"]
        elif "it_rel" in flows[cth_name].dims:
            return flows[cth_name].min("it_rel")
        return flows[cth_name]
    raise ValueError(f"minmax must be 'max' or 'min', got {minmax!r}")


def validity_mask(
    flows_org: xr.Dataset, flows0: xr.Dataset, v1: str, *, cthmin, cthmax, cth_name: str, ir_diff: "str | None",
    dIR: float,
) -> xr.DataArray:
    """Validity mask, with the full `(ns, omega, it, dim1, dim2)` candidate
    shape.

    F5 (fixed): cthmin/cthmax are trajectory-derived (the max/min cloud-top
    height *along* each candidate's own trajectory), hence genuinely
    dependent on which (omega, ns) candidate produced it -- evaluated here
    from `flows_org` (which, once `io.load_candidates` has classified
    cthmax/cthmin as candidate-dependent, carries them per-candidate), not
    from a single reference candidate applied uniformly to every candidate
    as before. `squeeze_cth` falls back to `flows_org[cth_name]` when no
    `cthmax`/`cthmin` variable exists at all (only plain `--record_initpos
    cth`, no `--out_cthmax`/`--out_cthmin`) -- in that case there is no
    candidate-dependent version to begin with, so `flows_org[cth_name]`
    not existing (it's `candidate_independent`, only ever in `flows0`) is
    handled by falling back to `flows0` there. IRdiff, by contrast, is an
    initial-position quantity (independent of omega/ns), so it's still
    taken from `flows0` alone; broadcasting `&` combines a candidate-shaped
    and a non-candidate-shaped mask naturally.

    Uses `flows_org[v1]` (not a literal `"vx"`) as the shape template: the
    original script hardcoded `flows0.vx`, which crashes unconditionally in
    `--polar` mode (no `vx` variable there). Since that crash means there
    is no prior golden behavior for `--polar` to preserve, this is fixed
    here rather than deferred.
    """
    cth_source = flows_org if (f"{cth_name}max" in flows_org or f"{cth_name}min" in flows_org) else flows0
    valid_index = xr.ones_like(flows_org[v1], dtype=bool)
    if cthmin:
        valid_index = valid_index & (squeeze_cth(cth_source, cth_name, "min") >= cthmin)
    if cthmax:
        valid_index = valid_index & (squeeze_cth(cth_source, cth_name, "max") <= cthmax)
    if ir_diff:
        IR1, IR2 = ir_diff.split("-")
        valid_index = valid_index & ((flows0[IR1] - flows0[IR2]) <= dIR)
    return valid_index


def apply_score_threshold(flows_org: xr.Dataset, used_vars: "list[str]", score_th) -> xr.Dataset:
    if score_th:
        flows_org[used_vars] = flows_org[used_vars].where(flows_org["score"].max("it_rel_v") <= score_th)
    return flows_org


def striation_mask(
    flows_org: xr.Dataset, used_vars: "list[str]", v1: str, v2: str, cth_name: str,
    *, ref_flows_path: "str | None", tau_stri: int, v_stri: float, cth_stri, omega_stri: float,
) -> xr.Dataset:
    if not (omega_stri and ref_flows_path):
        return flows_org
    used_vars_in_ref = [v1, v2]
    if cth_name in flows_org:
        used_vars_in_ref.append(cth_name)
    ref_flows = xr.open_dataset(ref_flows_path)[used_vars_in_ref]
    ref_flows["v"] = np.hypot(ref_flows[v1], ref_flows[v2])

    rolling_obj = ref_flows["v"].rolling(it=tau_stri, center=True, min_periods=1)
    v_contrast = rolling_obj.max("it") - rolling_obj.min("it")

    striation_grid = (v_contrast >= v_stri)
    if cth_stri:
        cthmax = squeeze_cth(ref_flows, cth_name, "max")
        striation_grid = striation_grid * (cthmax <= cth_stri)
    invalid_index_considering_stri = (np.abs(flows_org.omega) < np.abs(omega_stri)) * striation_grid
    flows_org[used_vars] = flows_org[used_vars].where(~invalid_index_considering_stri)
    return flows_org
