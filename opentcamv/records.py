"""--record_initpos / --record_alongtraj / --out_cthmin / --out_cthmax / --out_psr.

Ported from `10_conduct_tracking.py:573-622`.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import xarray as xr


def record_initpos(ofl: xr.Dataset, args, frames: xr.Dataset, tname: str, yname: str, xname: str, tg: np.ndarray) -> xr.Dataset:
    if not args.record_initpos:
        return ofl
    if args.polar:
        ofl = ofl.assign(x=ofl.r * np.cos(ofl.a), y=ofl.r * np.sin(ofl.a))
    time_indexer = xr.DataArray(frames[tname][tg].values, dims="it", coords={"it": ofl["it"].values})
    for varname in args.record_initpos:
        interped = frames[varname].interp({tname: time_indexer, yname: ofl["y"], xname: ofl["x"]})
        ofl[varname].data = interped.transpose(*ofl[varname].dims).values
    if args.polar:
        ofl = ofl.drop_vars(["x", "y"])
    return ofl


def _fill_seed_position_at_it_rel_zero(loc: np.ndarray, it_rel: np.ndarray, seed_grid_2d: np.ndarray) -> np.ndarray:
    """`loc` has shape `(nit, nit_rel, *seed_shape)`; overwrite the
    `it_rel == 0` slice with `seed_grid_2d` (shape `seed_shape`), broadcast
    over `nit`, regardless of tracking status -- so trajectory-interpolated
    fields always have a valid starting sample. Returns a new array."""
    out = loc.copy()
    zero_idx = int(np.where(it_rel == 0)[0][0])
    out[:, zero_idx] = seed_grid_2d[None, ...]
    return out


def record_alongtraj_and_cth(
    ofl: xr.Dataset, args, frames: xr.Dataset, tname: str, yname: str, xname: str, xxg: np.ndarray, yyg: np.ndarray,
) -> xr.Dataset:
    if not (args.record_alongtraj or args.out_cthmin or args.out_cthmax):
        return ofl

    if args.polar:
        ofl = ofl.assign(
            xloc=ofl["rloc"] * np.cos(ofl["aloc"]),
            yloc=ofl["rloc"] * np.sin(ofl["aloc"]),
        )

    out_shape = ofl["xloc"].shape  # (it, it_rel, dim1, dim2)
    times_on_traj = np.broadcast_to(ofl["time2"].values[:, :, None, None], out_shape)
    it_rel = ofl["it_rel"].values

    xlocs = _fill_seed_position_at_it_rel_zero(ofl["xloc"].values, it_rel, xxg)
    ylocs = _fill_seed_position_at_it_rel_zero(ofl["yloc"].values, it_rel, yyg)

    times_1d = xr.DataArray(times_on_traj.ravel(), dims="_pts")
    x_1d = xr.DataArray(xlocs.ravel(), dims="_pts")
    y_1d = xr.DataArray(ylocs.ravel(), dims="_pts")

    if args.record_alongtraj:
        for varname in args.record_alongtraj:
            interped = frames[varname].interp({tname: times_1d, yname: y_1d, xname: x_1d})
            ofl[varname].data = interped.data.reshape(out_shape)

    if args.out_cthmin or args.out_cthmax:
        interped = frames[args.cth].interp({tname: times_1d, yname: y_1d, xname: x_1d})
        cth_alongtraj = interped.data.reshape(out_shape)
        cth_da = xr.DataArray(cth_alongtraj, dims=ofl["xloc"].dims, coords=ofl["xloc"].coords)
        if args.out_cthmin:
            ofl[f"{args.cth}min"] = cth_da.min("it_rel")
        if args.out_cthmax:
            ofl[f"{args.cth}max"] = cth_da.max("it_rel")

    if args.polar:
        ofl = ofl.drop_vars(["xloc", "yloc"])
    return ofl


def peak_to_sidelobe_ratio(ofl: xr.Dataset, around_ratio: float = 0.15) -> xr.DataArray:
    """Peak-to-Sidelobe Ratio of `ofl["score_ary"]`."""
    max_scores = ofl["score_ary"].max(dim=["scx", "scy"])
    max_is_true = ofl["score_ary"] == max_scores

    around_wh = int(np.round(np.sqrt(around_ratio * (ofl["scx"].size * ofl["scy"].size))))
    dilated = ndi.binary_dilation(
        max_is_true, structure=np.ones([around_wh, around_wh], bool).reshape(1, 1, around_wh, around_wh, 1, 1)
    )
    sidelobes = xr.where(~dilated, ofl["score_ary"], np.nan)

    psr = (ofl["score"] - sidelobes.mean()) / sidelobes.std()
    return psr.rename("psr")
