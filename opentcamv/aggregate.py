"""Per-tid0 aggregation: --vagg (org/mean/startend), forward+backward trajectory
concatenation, and Cartesian -> polar (vr/vt, rloc/aloc) conversion.

Ported from v1's `10_conduct_tracking.py`. `fwd`/`bwd` are `TrackResult`s
(or `None` if that direction wasn't tracked). The v1 xarray -> v2 numpy
correspondence this leans on: `.isel(it_rel=idx)` becomes `arr[idx]` (axis 0
is the step axis in both), and `xr.concat([a, b], dim=...)` becomes
`np.concatenate([a, b], axis=0)`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .params import TrackingSetup


@dataclass
class AggregatedStep:
    vx: np.ndarray
    vy: np.ndarray
    xtraj: np.ndarray
    ytraj: np.ndarray
    score: np.ndarray
    stf: "Optional[np.ndarray]"
    stb: "Optional[np.ndarray]"
    vxfm: "Optional[np.ndarray]" = None
    vyfm: "Optional[np.ndarray]" = None
    vxbm: "Optional[np.ndarray]" = None
    vybm: "Optional[np.ndarray]" = None
    zss: "Optional[np.ndarray]" = None
    score_grids: "Optional[np.ndarray]" = None


def _nanmean(arr: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected here
        return np.nanmean(arr, axis=0)


def aggregate_step(
    args, setup: TrackingSetup, fwd, bwd, tid0: int, t: np.ndarray, grid,
    pickup_it_rel: np.ndarray, pickup_it_rel_v: np.ndarray,
) -> AggregatedStep:
    bothward = args.ward == "bothward"
    forward, backward = setup.forward, setup.backward

    vxfm = vyfm = vxbm = vybm = None
    if args.vagg == "org":
        if bothward:
            vx = np.concatenate([bwd.vx[np.flip(pickup_it_rel_v)], fwd.vx[pickup_it_rel_v]], axis=0)
            vy = np.concatenate([bwd.vy[np.flip(pickup_it_rel_v)], fwd.vy[pickup_it_rel_v]], axis=0)
        elif forward:
            vx, vy = fwd.vx[pickup_it_rel_v], fwd.vy[pickup_it_rel_v]
        else:
            vx, vy = bwd.vx[pickup_it_rel_v], bwd.vy[pickup_it_rel_v]
    else:
        if args.vagg == "mean":
            if forward:
                vxfm, vyfm = _nanmean(fwd.vx), _nanmean(fwd.vy)
            if backward:
                vxbm, vybm = _nanmean(bwd.vx), _nanmean(bwd.vy)
        elif args.vagg == "startend":
            if forward:
                start_end_dtf = abs(t[tid0 + setup.nsteps * setup.itstep] - t[tid0])
                vxfm = (fwd.x[-1] - fwd.x[0]) * grid.unit_factor / start_end_dtf
                vyfm = (fwd.y[-1] - fwd.y[0]) * grid.unit_factor / start_end_dtf
            if backward:
                start_end_dtb = abs(t[tid0 - setup.nsteps * setup.itstep] - t[tid0])
                vxbm = (bwd.x[0] - bwd.x[-1]) * grid.unit_factor / start_end_dtb
                vybm = (bwd.y[0] - bwd.y[-1]) * grid.unit_factor / start_end_dtb
        if bothward:
            vx = (vxfm + vxbm) / 2
            vy = (vyfm + vybm) / 2
        elif forward:
            vx, vy = vxfm, vyfm
        else:
            vx, vy = vxbm, vybm

    if bothward:
        xtraj = np.concatenate([bwd.x[np.flip(pickup_it_rel[1:])], fwd.x[pickup_it_rel]], axis=0)
        ytraj = np.concatenate([bwd.y[np.flip(pickup_it_rel[1:])], fwd.y[pickup_it_rel]], axis=0)
        score = np.concatenate([bwd.score[np.flip(pickup_it_rel_v)], fwd.score[pickup_it_rel_v]], axis=0)
        stf, stb = fwd.status, bwd.status
    elif forward:
        xtraj, ytraj = fwd.x[pickup_it_rel], fwd.y[pickup_it_rel]
        score = fwd.score[pickup_it_rel_v]
        stf, stb = fwd.status, None
    else:
        xtraj, ytraj = bwd.x[pickup_it_rel], bwd.y[pickup_it_rel]
        score = bwd.score[pickup_it_rel_v]
        stf, stb = None, bwd.status

    zss = None
    if fwd is not None and getattr(fwd, "templates", None) is not None or bwd is not None and getattr(bwd, "templates", None) is not None:
        if bothward:
            zss = np.concatenate([bwd.templates[np.flip(pickup_it_rel[1:])], fwd.templates[pickup_it_rel]], axis=0)
        elif forward:
            zss = fwd.templates[pickup_it_rel]
        else:
            zss = bwd.templates[pickup_it_rel]

    score_grids = None
    if fwd is not None and getattr(fwd, "score_grids", None) is not None or bwd is not None and getattr(bwd, "score_grids", None) is not None:
        if bothward:
            score_grids = np.concatenate([bwd.score_grids[np.flip(pickup_it_rel_v)], fwd.score_grids[pickup_it_rel_v]], axis=0)
        elif forward:
            score_grids = fwd.score_grids[pickup_it_rel_v]
        else:
            score_grids = bwd.score_grids[pickup_it_rel_v]

    return AggregatedStep(
        vx=vx, vy=vy, xtraj=xtraj, ytraj=ytraj, score=score, stf=stf, stb=stb,
        vxfm=vxfm, vyfm=vyfm, vxbm=vxbm, vybm=vybm, zss=zss, score_grids=score_grids,
    )


def to_polar(vx: np.ndarray, vy: np.ndarray, costh: np.ndarray, sinth: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """`(vr, vt)` from Cartesian `(vx, vy)`, broadcasting `costh`/`sinth`
    (shape `(na,)`) over the trailing azimuth axis of `vx`/`vy` (shape
    `(nr, na)` or `(*leading, nr, na)`)."""
    vr = vx * costh[None, :] + vy * sinth[None, :]
    vt = -vx * sinth[None, :] + vy * costh[None, :]
    return vr, vt


def traj_to_polar(xtraj: np.ndarray, ytraj: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """`(rtraj, atraj)` from Cartesian trajectory positions."""
    rtraj = np.hypot(xtraj, ytraj)
    atraj = np.arctan2(ytraj, xtraj)
    return rtraj, atraj
