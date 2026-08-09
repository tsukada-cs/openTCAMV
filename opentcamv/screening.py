"""--revrot restoration to the inertial frame, and vlim/Vd/Td/Vth screening.

Ported from `10_conduct_tracking.py:508-568`, operating on a single Omega's
completed output Dataset (v1 ran one Omega per process; loops Omega
inside one process, but each Omega still gets restored/screened independently
on its own `ofl`).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .output import AxisNames


def restore_revrot(ofl: xr.Dataset, args, omega: float, names: AxisNames) -> None:
    """Undo the rigid rotation applied during tracking, adding the
    rigid-rotation velocity back and de-rotating positions/azimuths so the
    output is in the inertial frame. No-op if `omega == 0.0`. Mutates `ofl`
    in place."""
    if not omega:
        return
    bothward = args.ward == "bothward"

    if args.polar:
        r2d = ofl["r"].values[:, None].repeat(ofl["a"].size, axis=1)
        a2d = None  # computed below only if bothward needs it
    else:
        a2d = np.arctan2(ofl["y"].T, ofl["x"])
        r2d = np.hypot(ofl["y"].T, ofl["x"]).values

    # `.astype(np.float32)` below is precision-only (matches the rest of
    # the output schema's dtype) -- it doesn't touch the add-back formula
    # or sign convention itself, both of which are load-bearing (see
    # CLAUDE.md). Without it, `omega * r2d` (r2d/a2d come from float64
    # coordinate arrays) silently promotes vx/vy/xloc/yloc to float64.
    revrot_mps = (omega * r2d * 1000).astype(np.float32)
    dt_rel = (ofl["time2"].values - ofl["time2"].sel(it_rel=0).values[:, None]).astype("timedelta64[s]").astype(float)
    azimuth_displacement_on_it_rel = (dt_rel * omega).astype(np.float32)

    if args.polar:
        ofl["vt"] = ofl["vt"] + revrot_mps
        ofl["aloc"] = ((ofl["aloc"] + azimuth_displacement_on_it_rel[:, :, None, None]) % (2 * np.pi)).astype(np.float32)
    else:
        u_rot = (-np.sin(a2d) * revrot_mps).astype(np.float32)
        v_rot = (np.cos(a2d) * revrot_mps).astype(np.float32)
        ofl["vx"] = ofl["vx"] + u_rot
        ofl["vy"] = ofl["vy"] + v_rot
        aloc = np.arctan2(ofl["yloc"], ofl["xloc"]).values
        aloc = aloc + azimuth_displacement_on_it_rel[:, :, None, None]
        rloc = np.hypot(ofl["yloc"], ofl["xloc"]).values
        ofl["xloc"].data = (np.cos(aloc) * rloc).astype(np.float32)
        ofl["yloc"].data = (np.sin(aloc) * rloc).astype(np.float32)

    if bothward:
        if args.polar:
            x = ofl["r"] * np.cos(ofl["a"])
            y = ofl["r"] * np.sin(ofl["a"])
            a2d = np.arctan2(y, x)
        u_rot = (-np.sin(a2d) * revrot_mps).astype(np.float32)
        v_rot = (np.cos(a2d) * revrot_mps).astype(np.float32)
        ofl["vxfm"] = ofl["vxfm"] + u_rot
        ofl["vyfm"] = ofl["vyfm"] + v_rot
        ofl["vxbm"] = ofl["vxbm"] + u_rot
        ofl["vybm"] = ofl["vybm"] + v_rot


def screen(ofl: xr.Dataset, args, names: AxisNames) -> xr.Dataset:
    """Apply --vlim/--Vd/--Td/--Vth, masking rejected points to NaN and
    setting `stf`/`stb` to 10 where a previously-OK (status 0) point was
    rejected. Returns the (possibly new, since `drop_vars` isn't in-place)
    Dataset -- always use the return value."""
    v1name, v2name = names.v1, names.v2
    bothward = args.ward == "bothward"
    forward = args.ward in ("forward", "bothward")
    backward = args.ward in ("backward", "bothward")

    valid = xr.DataArray(
        np.ones(ofl[v1name].shape, dtype=bool), dims=ofl[v1name].dims, coords=ofl[v1name].coords
    )
    if args.vlim > 0:
        valid = valid & (np.hypot(ofl[v1name], ofl[v2name]) <= args.vlim)

    if bothward:
        if args.Vd > 0:
            # Fixed vs v1: v1 compared this velocity-difference
            # *magnitude* (already in m/s) against `Vd**2`, making the
            # effective threshold 400 m/s at the default Vd=20 -- this
            # screen barely ever rejected anything. Now compared directly
            # against Vd, as the CLI help text ("threshold to limit the
            # maximum velocity difference") describes.
            valid = valid & (np.hypot(ofl["vxfm"] - ofl["vxbm"], ofl["vyfm"] - ofl["vybm"]) <= args.Vd)

        if args.Td is not None:
            dot_product = ofl["vxfm"] * ofl["vxbm"] + ofl["vyfm"] * ofl["vybm"]
            vabsf = np.hypot(ofl["vxfm"], ofl["vyfm"])
            vabsb = np.hypot(ofl["vxbm"], ofl["vybm"])
            angle_diff = np.arccos(dot_product / vabsf / vabsb)
            if args.Vth > 0:
                valid = valid & ~((angle_diff > np.deg2rad(args.Td)) & ((vabsf >= args.Vth) | (vabsb >= args.Vth)))
            else:
                valid = valid & ~(angle_diff > np.deg2rad(args.Td))
        ofl = ofl.drop_vars(["vxfm", "vyfm", "vxbm", "vybm"])

    ofl[[v1name, v2name]] = ofl[[v1name, v2name]].where(valid)
    if forward:
        ofl["stf"] = xr.where(~valid & (ofl["stf"] == 0), 10, ofl["stf"])
    if backward:
        ofl["stb"] = xr.where(~valid & (ofl["stb"] == 0), 10, ofl["stb"])

    if args.vagg in ("mean", "startend"):
        valid_t = valid.expand_dims({"it_rel": ofl.it_rel}, axis=0)
        ofl[[names.loc1, names.loc2]] = ofl[[names.loc1, names.loc2]].where(valid_t)

    return ofl
