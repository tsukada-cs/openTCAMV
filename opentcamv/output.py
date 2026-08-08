"""Output coordinates, the empty result Dataset, attrs, and the final write.

This is a close port of v1's dataset-skeleton construction. The output
NetCDF schema (variable names, coordinate names, attrs keys) is preserved
exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

import pyvttrac

FMISS = np.finfo(np.float32).max


@dataclass(frozen=True)
class AxisNames:
    """Names that switch between Cartesian (y, x) and polar (r, a) output."""

    dim1: str  # "y" or "r"
    dim2: str  # "x" or "a"
    loc1: str  # "yloc" or "rloc"
    loc2: str  # "xloc" or "aloc"
    v1: str  # "vy" or "vr"
    v2: str  # "vx" or "vt"


def axis_names(args) -> AxisNames:
    if not args.polar:
        return AxisNames("y", "x", "yloc", "xloc", "vy", "vx")
    return AxisNames("r", "a", "rloc", "aloc", "vr", "vt")


def build_seed_positions(args) -> "tuple[np.ndarray, np.ndarray]":
    """`(xxg, yyg)`: seed template positions, in the (x, y) plane regardless
    of --polar (polar seeds are pre-converted to Cartesian, matching what
    `pyvttrac.track()` needs)."""
    if not args.polar:
        xg = np.arange(args.xgran.start, args.xgran.stop + args.xint, args.xint)
        yg = np.arange(args.ygran.start, args.ygran.stop + args.yint, args.yint)
        xxg, yyg = np.meshgrid(xg, yg)
        return xxg, yyg
    rg = np.arange(args.rgran.start, args.rgran.stop + args.rint, args.rint)
    ag = np.linspace(0, 2 * np.pi, args.nath + 1)[:-1]
    costh, sinth = np.cos(ag), np.sin(ag)
    xxg = rg[:, None] * costh[None, :]
    yyg = rg[:, None] * sinth[None, :]
    return xxg, yyg


def polar_trig(args) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """`(costh, sinth)` over the azimuth grid, or None if not --polar."""
    if not args.polar:
        return None
    ag = np.linspace(0, 2 * np.pi, args.nath + 1)[:-1]
    return np.cos(ag), np.sin(ag)


def _build_it_rel(args) -> "tuple[np.ndarray, np.ndarray]":
    """`(it_rel, it_rel_v)`. Mutates `args.traj_int` if it was None (as v1 did)."""
    forward = args.ward in ("forward", "bothward")
    bothward = args.ward == "bothward"

    if args.traj_int is None:
        args.traj_int = 1 if args.vagg == "org" else args.ntrac

    if bothward:
        ntraj = int(2 * (args.ntrac / args.traj_int) + 1)  # always odd (>=1)
        ntraj_half = ntraj // 2
        it_rel = np.arange(-ntraj_half, ntraj_half + 1) * args.itstep * args.traj_int
    else:
        ntraj = int(args.ntrac / args.traj_int + 1)
        sgn = 1 if forward else -1
        it_rel = np.arange(0, ntraj) * args.itstep * args.traj_int * sgn

    if args.vagg == "org" and bothward and args.traj_int != 1:
        raise ValueError("When -v=org w/o --forward or --backward, set --traj_int 1")

    if bothward:
        ntrajv = ntraj - 1
        it_rel_v = np.arange(-ntraj_half, ntraj_half) * args.itstep * args.traj_int + args.itstep / 2
    else:
        ntrajv = int((args.ntrac - 1) / args.traj_int + 1)
        sgn = 1 if forward else -1
        it_rel_v = (np.arange(0, ntrajv) * args.itstep * args.traj_int + args.itstep / 2) * sgn

    return it_rel, it_rel_v


def build_coords(args, frames: xr.Dataset, tname: str, tg: np.ndarray) -> dict:
    """it/time, x&y or r&a, it_rel, it_rel_v coordinates."""
    base_time = frames[tname].isel({tname: 0})
    time_ax_values = pd.to_timedelta(frames[tname].isel({tname: tg}) - base_time).total_seconds().values
    time_ax = xr.DataArray(
        time_ax_values, coords={"it": (["it"], tg)},
        attrs=dict(long_name="time", units=f'seconds since {base_time.dt.strftime("%F %H:%M:%S").item()}'),
    ).rename("time")
    it_ax = xr.DataArray(tg, coords={"it": (["it"], tg)}, attrs=dict(long_name="time index", units="")).rename("it")
    coords = {"it": it_ax, "time": time_ax}

    if not args.polar:
        xg = np.arange(args.xgran.start, args.xgran.stop + args.xint, args.xint)
        xax = xr.DataArray(xg, coords={"x": (["x"], xg)}, attrs=dict(long_name="x", units="km")).rename("x")
        yg = np.arange(args.ygran.start, args.ygran.stop + args.yint, args.yint)
        yax = xr.DataArray(yg, coords={"y": (["y"], yg)}, attrs=dict(long_name="y", units="km")).rename("y")
        coords.update({"y": yax, "x": xax})
    else:
        rg = np.arange(args.rgran.start, args.rgran.stop + args.rint, args.rint)
        rax = xr.DataArray(rg, coords={"r": (["r"], rg)}, attrs=dict(long_name="radius", units="km")).rename("r")
        ag = np.linspace(0, 2 * np.pi, args.nath + 1)[:-1]
        aax = xr.DataArray(ag, coords={"a": (["a"], ag)}, attrs=dict(long_name="azimuth", units="radian")).rename("a")
        coords.update({"r": rax, "a": aax})

    it_rel, it_rel_v = _build_it_rel(args)
    it_rel_ax = xr.DataArray(
        it_rel, coords={"it_rel": (["it_rel"], it_rel)},
        attrs=dict(long_name="relative time index along the original data", units=""),
    ).rename("it_rel")
    it_rel_v_ax = xr.DataArray(
        it_rel_v, coords={"it_rel_v": (["it_rel_v"], it_rel_v)},
        attrs=dict(long_name="relative time index for speed", units=""),
    ).rename("it_rel_v")
    coords.update({"it_rel": it_rel_ax, "it_rel_v": it_rel_v_ax})
    return coords


def pickup_indices(args, coords: dict) -> "tuple[np.ndarray, np.ndarray]":
    """`(pickup_it_rel, pickup_it_rel_v)`: indices into a single trajectory's
    forward/backward `TrackResult` arrays that land on the `it_rel`/`it_rel_v`
    output grid (spaced by `traj_int`)."""
    bothward = args.ward == "bothward"
    if bothward:
        ntraj_half = coords["it_rel"].size // 2
        pickup_it_rel_v = np.arange(ntraj_half) * args.traj_int
        pickup_it_rel = np.arange(ntraj_half + 1) * args.traj_int
    else:
        ntrajv = coords["it_rel_v"].size
        ntraj = coords["it_rel"].size
        pickup_it_rel_v = np.arange(ntrajv) * args.traj_int
        pickup_it_rel = np.arange(ntraj) * args.traj_int
    return pickup_it_rel, pickup_it_rel_v


def build_empty_dataset(args, frames: xr.Dataset, tname: str, coords: dict, tg: np.ndarray, setup) -> "tuple[xr.Dataset, dict]":
    """The empty result Dataset (all data_vars NaN/fill-valued) plus its
    variable `encoding` dict (mirrors v1's separately-tracked `encoding`)."""
    names = axis_names(args)
    forward = args.ward in ("forward", "bothward")
    backward = args.ward in ("backward", "bothward")
    bothward = args.ward == "bothward"

    dim1, dim2 = names.dim1, names.dim2
    ax1, ax2 = coords[dim1], coords[dim2]
    axes = [coords["it"].name, dim1, dim2]
    vshape = [tg.size, ax1.size, ax2.size]
    # NOTE (inherited from v1): `initpos_shape` is intentionally *not* kept
    # in sync with `vshape` below when --vagg=org (which prepends an
    # `it_rel_v` axis to `vshape`/`axes`) -- v1 had this same mismatch, and
    # --record_initpos with --vagg=org was never a supported combination.
    initpos_shape = list(vshape)

    axes_t = [axes[0], "it_rel", dim1, dim2]
    vshape_t = [vshape[0], coords["it_rel"].size, *vshape[1:]]

    if args.vagg == "org":
        axes = [axes[0], "it_rel_v", *axes[1:]]
        vshape = [vshape[0], coords["it_rel_v"].size, *vshape[1:]]

    ofl = xr.Dataset(
        data_vars={
            names.v1: (axes, np.full(vshape, np.nan, dtype=np.float32)),
            names.v2: (axes, np.full(vshape, np.nan, dtype=np.float32)),
            names.loc1: (axes_t, np.full(vshape_t, np.nan, dtype=np.float32)),
            names.loc2: (axes_t, np.full(vshape_t, np.nan, dtype=np.float32)),
        },
        coords=coords,
    )
    ofl[names.v1].attrs.update({"long_name": f"{dim1}-axis velocity", "units": "m/s"})
    ofl[names.v2].attrs.update({"long_name": f"{dim2}-axis velocity", "units": "m/s"})
    ofl[names.loc1].attrs.update({"long_name": f"{dim1} location", "units": ax1.units})
    ofl[names.loc2].attrs.update({"long_name": f"{dim2} location", "units": ax2.units})
    ofl["score"] = (
        ["it", "it_rel_v", dim1, dim2],
        np.full((tg.size, coords["it_rel_v"].size, *vshape[-2:]), FMISS, np.float32),
    )
    ofl["score"].attrs.update({"long_name": "score", "units": ""})

    encoding = {key: {"_FillValue": FMISS} for key in ofl.data_vars.keys()}

    if forward:
        ofl = ofl.assign({"stf": (["it", dim1, dim2], np.full([tg.size, ax1.size, ax2.size], -10, dtype=np.int16))})
        ofl["stf"].attrs.update(_status_attrs("forward tracking status"))
        encoding.update({"stf": {"_FillValue": -10}})
    if backward:
        ofl = ofl.assign({"stb": (["it", dim1, dim2], np.full([tg.size, ax1.size, ax2.size], -10, dtype=np.int16))})
        ofl["stb"].attrs.update(_status_attrs("backward tracking status"))
        encoding.update({"stb": {"_FillValue": -10}})

    if bothward:
        for key in ("vxfm", "vyfm", "vxbm", "vybm"):
            ofl[key] = (axes, np.full(vshape, np.nan, dtype=np.float32))

    it_plus_it_rel = (coords["it"] + coords["it_rel"]).values
    time2 = frames[tname].isel({tname: it_plus_it_rel.ravel()}).values.reshape(it_plus_it_rel.shape)
    ofl["time2"] = (["it", "it_rel"], time2)

    if args.record_initpos:
        for varname in args.record_initpos:
            ofl[varname] = (axes, np.zeros(initpos_shape, dtype=np.float32))
            ofl[varname].attrs.update(frames[varname].attrs)

    if args.record_alongtraj:
        for varname in args.record_alongtraj:
            ofl[varname] = (axes_t, np.zeros(vshape_t, dtype=np.float32))
            ofl[varname].attrs.update(frames[varname].attrs)

    if args.out_subimage:
        nsy, nsx = setup.template
        zss_shape = (tg.size, coords["it_rel"].size, nsy, nsx, *vshape[-2:])
        ofl["zss"] = (["it", "it_rel", "sy", "sx", dim1, dim2], np.full(zss_shape, np.nan, np.float32))
        ofl["zss"].attrs.update({"long_name": "sub image", "units": frames[args.varname].attrs.get("units", ""), "_FillValue": FMISS})
    if args.out_score_ary:
        iyhw, ixhw = setup.search_radius
        score_ary_shape = (tg.size, coords["it_rel_v"].size, 2 * iyhw + 1, 2 * ixhw + 1, *vshape[-2:])
        ofl["score_ary"] = (["it", "it_rel_v", "scy", "scx", dim1, dim2], np.full(score_ary_shape, np.nan, np.float32))
        ofl["score_ary"].attrs.update({"long_name": "score array", "units": "", "_FillValue": FMISS})

    return ofl, encoding


def _status_attrs(long_name: str) -> dict:
    return {
        "long_name": long_name, "units": "",
        "flags": list(range(12)),
        "flag means": [
            "Alive", "Invalid time index", "Invalid sub image", "Low contrast",
            "Side zsub peak", "Invalid time index", "Can't get score",
            "Can't get score peak", "Low score", "Large V change",
            "Exceed V limit or Large back-forward change", "Max dt",
        ],
    }


def build_attrs(args, omega: float, script_path: str) -> dict:
    """Per-Omega `vars(args)`-derived attrs (v1's exec/history/arg dump).
    `args.revrot` is overridden to the *scalar* `omega` for this output file,
    matching v1's one-file-per-Omega schema (each file's `revrot` attr was
    always a single float, never a list)."""
    args_for_attrs = dict(vars(args))
    args_for_attrs["revrot"] = omega

    args_str = ""
    for key, val in args_for_attrs.items():
        args_str += f" --{key}={val}"
    attrs = {"exec": f"python {script_path}" + args_str}
    attrs.update(args_for_attrs)
    attrs.update({"history": f'{os.getenv("USER")} {pd.Timestamp.now().strftime("%F %H:%M:%S UTC")}'})
    return attrs


def scalarize_attrs(ofl: xr.Dataset) -> None:
    """In-place: coerce attrs values into netCDF-serializable scalars (v1's
    final pre-write attrs pass)."""
    for key, value in ofl.attrs.items():
        if value is None:
            ofl.attrs[key] = "None"
        elif isinstance(value, pd.Timestamp):
            ofl.attrs[key] = value.strftime("%F %H:%M:%S")
        elif isinstance(value, slice):
            ofl.attrs[key] = [value.start, value.stop]
        elif isinstance(value, bool):
            ofl.attrs[key] = int(value)


def write(ofl: xr.Dataset, encoding: dict, path: str, complevel: int) -> None:
    scalarize_attrs(ofl)
    # NOTE (inherited from v1): this *replaces* each variable's encoding
    # dict rather than merging into it, which silently drops the
    # `_FillValue` entries set earlier (in `build_empty_dataset`). That
    # matches v1's actual on-disk output (verified: no `_FillValue` in the
    # sample files' variable encoding) even though it looks like an
    # oversight -- kept for output-schema fidelity, not because it's
    # obviously the intended behavior.
    for var in ofl:
        encoding[var] = {"complevel": complevel, "zlib": True}
    ofl.to_netcdf(path, encoding=encoding)
