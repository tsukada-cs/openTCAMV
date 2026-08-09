"""The current selection algorithm: restrict each omega
candidate to its minimum valid `ns`, sort the reduced (omega-space)
candidates by cost, then iteratively reject candidates far from their local
neighborhood's median and re-sort, until convergence.

This is a port of the original script's body (lines 190-272), not yet
expressed in terms of `selectors.base.Selector`/`SelectionResult` -- those
describe the *target* shape for the relaxation-labeling selector to
share a common interface with. Refactoring this implementation onto that
interface is deliberately deferred: the `np.argsort`/`take_along_axis`
mechanism below depends on specific, hardcoded axis *positions*
(`omega_axis = 0`, `priority_axis = 0`) that hold only because of how
`finalize.io.load_candidates` happens to order `flows_org`'s dimensions and
because the `ns` reduction below runs before anything is sorted. Preserving
that exactly (rather than "improving" it to look up axis positions by name,
which risks silently picking a different axis if the reasoning about dim
order is wrong somewhere) is what keeps this bit-identical to the
pre-refactor script for a single `ns`.

The `ns` reduction (picking, per *original omega label* `omega`/`it`/`y`/`x`
point, the smallest `ns` whose candidate is not masked out) happens before
the priority sort/rename, not after (as an earlier version of this module
did, matching a literal reading of the pre-refactor script's single-`ns`-only
lines 190-230). Reducing after the sort is wrong once `ns` has more than one
value: the initial sort (`argsort` over the `omega` axis) is done
independently per `ns`, so "priority-rank k" does not name the same physical
omega across different `ns` values -- reducing across `ns` at a shared
priority-rank then silently mixes candidates from different omegas. There
was no such case in the original script (it never supported `ns.size > 1`,
F1), so there is no prior golden behavior this could deviate from; reducing
first sidesteps the ambiguity entirely and is what makes the multi-`ns`
"rescue" case (a point invalid at the smallest `ns` recovered by a larger
one) return a coherent, single-omega answer instead of a broken extra axis.
For `ns.size == 1` the reduction is a no-op irrespective of when it runs (it
never has more than one candidate to choose between), so this reordering
does not change single-`ns` output.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def select(
    flows_org: xr.Dataset,
    priority_key: str,
    *,
    use_v: bool,
    windows_sizes: dict,
    dth: "float | None",
    dc: "float | None",
    max_epoch: int,
    out_final_medians: bool,
) -> "tuple[xr.Dataset, xr.DataArray, xr.DataArray, dict, int]":
    """Returns `(target_flows, final_score, valid_omega, ns_selection, ep)`.

    `ns_selection` is `{"valid_ns_mins": ..., "ns_min": ...}`, in
    *omega-space* (dims `(omega, it, y, x)` -- these are the original omega
    labels, not sorted-priority ranks; see the module docstring). Needed by
    the caller to re-derive `flows_ns_min` for final assembly (mirrors the
    original script keeping `valid_ns_mins`/`ns_min` alive past this block).
    """
    ns_broadcast = flows_org["ns"].broadcast_like(flows_org[priority_key])
    ns_broadcast = ns_broadcast.where(flows_org[priority_key].notnull(), np.nan)
    valid_ns_mins = ns_broadcast.min("ns").compute()
    del ns_broadcast
    ns_min = flows_org.ns.min().item()

    if use_v:
        target_flows = flows_org[["v", priority_key]].sel(ns=valid_ns_mins.fillna(ns_min)).copy(deep=True).compute()
    else:
        target_flows = (
            flows_org[["vx", "vy", priority_key]].sel(ns=valid_ns_mins.fillna(ns_min)).copy(deep=True).compute()
        )

    omega_axis = 0
    index_sortby_priority = np.argsort(target_flows[priority_key], axis=omega_axis)
    for varname in target_flows:
        target_flows[varname].data = np.take_along_axis(
            target_flows[varname].data, index_sortby_priority.values, axis=omega_axis
        )
    target_flows = target_flows.rename({"omega": "priority"})
    priority_axis = 0

    shape = target_flows[priority_key].shape
    final_indices = np.arange(flows_org.omega.size)[:, None, None, None].repeat(shape[1], 1).repeat(shape[2], 2).repeat(shape[3], 3)
    ep = 0
    for ep in range(max_epoch):
        logger.info(f"===> Updating {ep + 1} / {max_epoch}")
        if use_v:
            v_med = target_flows["v"].isel(priority=0).rolling(windows_sizes, min_periods=1, center=True).median()
            d2 = np.abs(target_flows["v"] - v_med)
        else:
            vx_med = target_flows["vx"].isel(priority=0).rolling(windows_sizes, min_periods=1, center=True).median()
            vy_med = target_flows["vy"].isel(priority=0).rolling(windows_sizes, min_periods=1, center=True).median()
            d2 = np.hypot(target_flows["vx"] - vx_med, target_flows["vy"] - vy_med).compute()

        valid_index = xr.ones_like(d2, bool)
        if dth:
            valid_index = valid_index & (d2 <= dth)
        if dc:
            if use_v:
                valid_index = valid_index & (d2 <= v_med * dc)
            else:
                valid_index = valid_index & (d2 <= np.hypot(vx_med, vy_med) * dc)
        target_flows[priority_key] = target_flows[priority_key].where(valid_index)

        index_sortby_priority = np.argsort(target_flows[priority_key], axis=priority_axis)
        if (index_sortby_priority == final_indices).all().item():
            break
        for varname in target_flows:
            target_flows[varname].data = np.take_along_axis(
                target_flows[varname].data, index_sortby_priority.values, axis=priority_axis
            )

    if out_final_medians:
        if use_v:
            target_flows["v_med"] = v_med
        else:
            target_flows["vx_med"] = vx_med
            target_flows["vy_med"] = vy_med

    final_score = target_flows[priority_key].isel(priority=0).compute().drop_vars("priority")
    valid_omega = target_flows[priority_key].notnull().any("priority").compute()

    ns_selection = {"valid_ns_mins": valid_ns_mins, "ns_min": ns_min}
    return target_flows, final_score, valid_omega, ns_selection, ep
