"""Priority (ranking cost) computation, and assembling the selector's
result back into a finished Dataset."""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_priority(flows_org: xr.Dataset, priority: str, v1: str, v2: str) -> "tuple[xr.Dataset, str]":
    """Returns `(flows_org, priority_key)`. Mutates `flows_org` to add the
    ranking field ("mscore" or "dangv").

    F6 (fixed): for `priority="dangv"`, the pre-refactor script reassigned
    `flows_org["omega"]` to a plain index (`np.arange(...)`) right after
    using its physical values to compute `dangv`, discarding them --
    `--out_final_omega` then reported indices, not rad/s, and
    `masking.striation_mask`'s `--omega_stri` comparison (which runs later
    in the pipeline and reads `flows_org.omega`'s values directly) compared
    a physical rad/s threshold against those same meaningless indices.
    Nothing downstream actually needs `omega` reassigned to an index --
    dangv's own values don't depend on it -- so the fix is simply to not do
    that.
    """
    if priority == "score":
        flows_org["mscore"] = -flows_org["score"].mean("it_rel_v")
        return flows_org, "mscore"
    elif priority == "dangv":
        if not flows_org.attrs["polar"]:
            r2d = np.hypot(flows_org.x, flows_org.y)
            theta2d = np.arctan2(flows_org.y, flows_org.x)
            angv = (-flows_org["vx"] * np.sin(theta2d) + flows_org["vy"] * np.cos(theta2d)) / r2d / 1000
        else:
            angv = flows_org["vt"] / flows_org["r"] / 1000
        flows_org["dangv"] = np.abs(angv - flows_org["omega"])
        return flows_org, "dangv"
    raise ValueError(f"Unknown priority {priority!r}")


def prepare_target_fields(flows_org: xr.Dataset, *, use_v: bool, v1: str, v2: str) -> xr.Dataset:
    """Lines 190-198 of the original script: materializes the Cartesian
    `vx`/`vy` (or `v`, under `--useV`) fields the selector ranks candidates
    and measures neighborhood-consistency by, converting from polar
    `vr`/`vt` first if needed. This conversion happens regardless of
    `--polar`/`--useV`, matching the original. Mutates and returns
    `flows_org`."""
    if use_v:
        flows_org["v"] = np.hypot(flows_org[v1], flows_org[v2])
    elif flows_org.attrs["polar"]:
        flows_org["vx"] = -flows_org[v2] * np.sin(flows_org.a) + flows_org[v1] * np.cos(flows_org.a)
        flows_org["vy"] = flows_org[v2] * np.cos(flows_org.a) + flows_org[v1] * np.sin(flows_org.a)
    return flows_org


def assemble_final(
    flows_org: xr.Dataset, candidate_independent_keepvars: "list[str]", per_candidate_vars: "list[str]",
    priority_key: str, final_score: xr.DataArray, valid_omega: xr.DataArray, ns_selection: dict,
    *, out_final_ns: bool, out_final_omega: bool,
) -> xr.Dataset:
    """Lines 254-272 of the original script: picks each point's final
    candidate (matching `flows_ns_min[priority_key]` against `final_score`
    via `idxmin`, since the iterative sort loses direct track of which
    original `omega` won).

    `per_candidate_vars` (`used_vars` plus F5's `candidate_dependent_keepvars`
    -- e.g. `cthmax`/`cthmin`/`--record_alongtraj` outputs, all carrying the
    full candidate axis in `flows_org`) are selected via the same
    `.sel(omega=final_omega)`, so each ends up with the *winning* candidate's
    own value. `candidate_independent_keepvars` (e.g. `--record_initpos`
    outputs) don't vary by candidate, so they're attached directly from
    `flows_ns_min` with no `.sel`.
    """
    valid_ns_mins = ns_selection["valid_ns_mins"]
    ns_min = ns_selection["ns_min"]

    if flows_org.ns.size == 1:
        flows_ns_min = flows_org.squeeze()
    else:
        flows_ns_min = flows_org.sel(ns=valid_ns_mins.fillna(ns_min))
    if out_final_ns:
        flows_ns_min["final_ns"] = valid_ns_mins

    corresponding_omega_is_zero = np.abs(flows_ns_min[priority_key] - final_score)
    final_omega = corresponding_omega_is_zero.fillna(np.inf).idxmin("omega")
    final_flows = flows_ns_min[per_candidate_vars].sel(omega=final_omega).where(valid_omega).compute()

    for keepvar in candidate_independent_keepvars:
        final_flows[keepvar] = flows_ns_min[keepvar]
    if out_final_omega:
        final_flows["final_omega"] = final_omega

    return final_flows
