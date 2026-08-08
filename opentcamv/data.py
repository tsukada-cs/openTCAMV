"""Input NetCDF reading: time axis, tracking-start indices, physical grid, mask.

Several functions here mutate `args` in place (setting `itran`/`itfst`/`itlst`/
`start`/`end`/`ref_dt`), mirroring v1's script-level mutation of `argparse`'s
`Namespace`. This is intentional: the final `vars(args)` dict is dumped into
the output NetCDF's attrs (see `opentcamv.output.build_attrs`), so the derived
values need to land back on `args` for the output schema to match v1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

import pyvttrac

logger = logging.getLogger(__name__)


def open_frames(args) -> "tuple[xr.Dataset, str, str, str]":
    """Open the input file, apply --sector/time-range selection, and return
    `(frames, tname, yname, xname)`. Mutates `args` (see module docstring)."""
    frames = xr.open_dataset(args.ifn)
    tname, yname, xname = frames[args.varname].dims
    time_ax_org = frames[tname].values

    if args.sector:
        pickup_inds = np.zeros(frames["sector"].shape, bool)
        for sector in args.sector:
            pickup_inds += (frames["sector"] == sector)
        frames = frames.isel({tname: pickup_inds})

    args.itran = resolve_time_range(args, time_ax_org)

    forward = args.ward in ("forward", "bothward")
    backward = args.ward in ("backward", "bothward")

    args.itfst, args.itlst = args.itran.start, args.itran.stop
    if backward:
        args.itfst = max([args.itran.start - args.ntrac * args.itstep, 0])
    if forward:
        args.itlst = min([args.itran.stop + args.ntrac * args.itstep, time_ax_org.size - 1])
    frames = frames.isel({tname: slice(args.itfst, args.itlst + 1)})
    return frames, tname, yname, xname


def resolve_time_range(args, time_ax_org: np.ndarray) -> slice:
    """Resolve --itran, or derive it from --start/--end. Mutates `args.start`/
    `args.end` to their resolved `pd.Timestamp` form (as v1 did)."""
    if args.itran is not None:
        return args.itran

    if args.start is None:
        args.start = time_ax_org[0]
    args.start = pd.to_datetime(args.start)
    if args.end is None:
        args.end = time_ax_org[-1]
    args.end = pd.to_datetime(args.end)

    if (args.end - args.start) >= pd.Timedelta("7day"):
        logger.warning("The period lasts more than a week, which may be too long. Please check the period.")

    try:
        return slice(np.min(np.where(time_ax_org >= args.start)), np.max(np.where(time_ax_org <= args.end)))
    except Exception:
        raise ValueError("Specified time period is out of range")


def compute_tg(args, nt: int) -> np.ndarray:
    """Time indices at which a tracking trajectory starts."""
    forward = args.ward in ("forward", "bothward")
    backward = args.ward in ("backward", "bothward")
    bothward = args.ward == "bothward"
    if bothward:
        return np.arange(args.ntrac * args.itstep, nt - args.ntrac * args.itstep, args.tidstep)
    elif forward:
        return np.arange(0, nt - args.ntrac * args.itstep, args.tidstep)
    elif backward:
        return np.arange(args.ntrac * args.itstep, nt, args.tidstep)
    raise AssertionError(f"unreachable: args.ward={args.ward!r}")


def build_time_seconds(frames: xr.Dataset, tname: str) -> np.ndarray:
    """Seconds elapsed since the first (post-slice) frame, for every frame."""
    return (frames[tname] - frames[tname][0]).dt.total_seconds().values.astype(np.float64)


def build_grid(frames: xr.Dataset, varname: str, xname: str, yname: str) -> "pyvttrac.Grid":
    """Physical <-> index Grid for the tracked variable's (x, y) coordinates.

    `dx`/`dy` are signed (kept from `coord[-1] - coord[0]`) and `x0`/`y0` are
    the coordinate's first value, rather than v1's `abs(dx)`/`coord.min()` --
    this is what makes a descending coordinate axis (e.g. `y` in many
    satellite products) round-trip correctly through `pyvttrac.Grid`. For an
    ascending coordinate, this is numerically identical to v1.
    """
    xcoord = frames[varname].coords[xname]
    dx = float(((xcoord[-1] - xcoord[0]) / (xcoord.size - 1)).item())
    x0 = float(xcoord[0].item())
    ycoord = frames[varname].coords[yname]
    dy = float(((ycoord[-1] - ycoord[0]) / (ycoord.size - 1)).item())
    y0 = float(ycoord[0].item())

    unit_factor = 1e3 if xcoord.attrs.get("units") == "km" else 1.0
    return pyvttrac.Grid(x0=x0, y0=y0, dx=dx, dy=dy, unit_factor=unit_factor)


def build_mask(args, frames: xr.Dataset) -> "np.ndarray | None":
    """`(nt, ny, nx)` boolean mask (True = ignore), or None if --maskvar isn't set."""
    if not args.maskvar:
        return None
    mask = np.zeros(frames[args.maskvar].shape, bool)
    if args.mask_lower_lim:
        mask = mask + (frames[args.maskvar] <= args.mask_lower_lim).values
    if args.mask_upper_lim:
        mask = mask + (frames[args.maskvar] >= args.mask_upper_lim).values
    return mask


def compute_ref_dt(args, t: np.ndarray) -> float:
    """Reference dt for deriving the search radius from --Vs: the largest
    frame-to-frame interval that's still <= --dtlimit. Mutates `args.ref_dt`."""
    if args.ref_dt is None:
        tdiff = np.diff(t)
        args.ref_dt = float(np.max(tdiff[tdiff <= args.dtlimit]))
    return args.ref_dt


def compute_search_radius(args, grid: "pyvttrac.Grid", ref_dt: float) -> "tuple[int, int]":
    """`(iyhw, ixhw)` pixel search radius, from --hs directly or derived from
    --Vs and `ref_dt`.

    A tracking step covers `itstep` frames' worth of time, not one frame, so
    the radius is derived from `ref_dt * itstep`: with the default itstep=1
    this is unchanged, but for itstep > 1 using bare `ref_dt` under-sized
    the search window relative to how far a step can actually move.
    """
    if args.hs:
        return (args.hs, args.hs)
    return pyvttrac.search_radius_from_velocity((args.Vs, args.Vs), dt=ref_dt * args.itstep, grid=grid)
