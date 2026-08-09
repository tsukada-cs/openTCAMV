"""Candidate selection: pick the best (Omega, ns) candidate at every grid
point from `10_conduct_tracking.py`'s outputs. Thin CLI driver:
`scripts/20_finalize_tracking.py`.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from . import candidates as candidates_mod
from . import masking
from . import window as window_mod
from .io import load_candidates
from .selectors import RelaxationSelector, iterative_median

logger = logging.getLogger(__name__)


def run(args) -> xr.Dataset:
    """Runs the full candidate-selection pipeline for the given (already
    argparse-parsed) `args`. Returns the finished `xr.Dataset`; writing it
    out (attrs/dims cleanup, `to_netcdf`) is left to the caller -- see
    `scripts/20_finalize_tracking.py`."""
    logger.info(f"[Input filename rule] {args.ifns_rule}")
    logger.info(f"[List of ns] {args.ns}")
    logger.info(f"[List of Ω_i] {args.omega}")
    logger.info(f"[Scheduled output filename] {args.ofn}")

    flows_org, flows0, names, candidate_independent_keepvars, candidate_dependent_keepvars, used_vars = load_candidates(
        args.ifns_rule, args.ns, args.omega, exclude=args.exclude, cth_name=args.cth
    )
    dim1, dim2, loc1, loc2, v1, v2 = names.dim1, names.dim2, names.loc1, names.loc2, names.v1, names.v2
    per_candidate_vars = used_vars + candidate_dependent_keepvars

    windows_sizes = window_mod.window_sizes(
        args.Tw, args.Hw, args.xw, args.yw, args.rw, args.aw,
        flows0.attrs["dtmean"], flows0.attrs["xint"], flows0.attrs["yint"], flows0.attrs["rint"], flows0.attrs["nath"],
        bool(flows0.attrs["polar"]), dim1, dim2,
    )
    logger.info(f"[Median filter window] {windows_sizes}")

    trim = None
    if args.itmin > 0 or args.itmax is not None or args.itstep > 1:
        itmax = args.itmax if args.itmax is not None else flows_org.it.size - 1
        tmin = flows_org.it.isel(it=args.itmin).item()
        tmax = flows_org.it.isel(it=itmax).item()
        itmin_padded = max(args.itmin - (windows_sizes["it"] // 2 + 1), 0)
        itmax_padded = min(itmax + (windows_sizes["it"] // 2 + 1), flows_org.it.size - 1)
        flows_org = flows_org.isel(it=slice(itmin_padded, itmax_padded + 1, args.itstep))
        trim = (tmin, tmax)

    valid_index = masking.validity_mask(
        flows_org, flows0, v1, cthmin=args.cthmin, cthmax=args.cthmax, cth_name=args.cth,
        ir_diff=args.IRdiff, dIR=args.dIR,
    )
    flows_org[per_candidate_vars] = flows_org[per_candidate_vars].where(valid_index)

    flows_org = masking.apply_score_threshold(flows_org, per_candidate_vars, args.score_th)

    flows_org, priority_key = candidates_mod.compute_priority(flows_org, args.priority, v1, v2)

    flows_org = masking.striation_mask(
        flows_org, per_candidate_vars, v1, v2, args.cth,
        ref_flows_path=args.ref_flows, tau_stri=args.tau_stri, v_stri=args.v_stri,
        cth_stri=args.cth_stri, omega_stri=args.omega_stri,
    )

    flows_org = candidates_mod.prepare_target_fields(flows_org, use_v=args.useV, v1=v1, v2=v2)

    selector = getattr(args, "selector", "iterative_median")
    if selector == "relaxation":
        # Skeleton only: raises NotImplementedError. `flows_org`  doesn't actually match 
        # `Selector.select`'s `cand` shape yet -- there's no real implementation to match it against.
        RelaxationSelector().select(flows_org, window=windows_sizes, max_epoch=int(args.max_epoch))
        raise AssertionError("unreachable: RelaxationSelector.select always raises")
    elif selector != "iterative_median":
        raise ValueError(f"Unknown --selector {selector!r}")

    target_flows, final_score, valid_omega, ns_selection, ep = iterative_median.select(
        flows_org, priority_key, use_v=args.useV, windows_sizes=windows_sizes,
        dth=args.dth, dc=args.dc, max_epoch=int(args.max_epoch), out_final_medians=args.out_final_medians,
    )
    del target_flows

    logger.info("Finalizing...")
    final_flows = candidates_mod.assemble_final(
        flows_org, candidate_independent_keepvars, per_candidate_vars, priority_key, final_score, valid_omega,
        ns_selection, out_final_ns=args.out_final_ns, out_final_omega=args.out_final_omega,
    )
    logger.info(f"[FINALIZED] at epoch={ep + 1}")

    if trim is not None:
        tmin, tmax = trim
        final_flows = final_flows.sel(it=slice(tmin, tmax))

    return final_flows


def format_for_write(final_flows: xr.Dataset) -> xr.Dataset:
    """Transposes each variable's dims into the canonical order the output
    schema promises, and fixes up `_FillValue` encoding. Matches the
    original script's final pre-`to_netcdf` pass exactly."""
    for varname in final_flows:
        if "it_rel" in final_flows[varname].dims:
            final_flows[varname] = final_flows[varname].transpose("it", "it_rel", ...)
        elif varname in ("score", "psr"):
            final_flows[varname] = final_flows[varname].transpose("it", "it_rel_v", ...)
        elif varname == "score_ary":
            final_flows[varname] = final_flows[varname].transpose("it", "it_rel_v", "scy", "scx", ...)
        if "missing_value" in final_flows[varname].encoding:
            final_flows[varname].encoding["_FillValue"] = final_flows[varname].encoding["missing_value"]
    return final_flows
