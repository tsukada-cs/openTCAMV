"""TrackingSetup: derived tracking parameters, and the pyVTTrac.Tracker built from them.

Encodes the v1 -> v2 parameter correspondence; axis order is the biggest
source of mistakes here: `template=(nsy, nsx)`, `search_radius=(iyhw, ixhw)`,
`max_velocity_change=(dvy, dvx)` -- all (y, x) -- except `min_score=(Sth0,
Sth1)`, which is (first-step, subsequent), not (y, x).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

import pyvttrac

from . import data


@dataclass(frozen=True)
class TrackingSetup:
    """Everything needed to build a `pyvttrac.Tracker` and drive tracking,
    derived once from `args` + the input data."""

    grid: "pyvttrac.Grid"
    template: "tuple[int, int]"  # (nsy, nsx)
    search_radius: "tuple[int, int]"  # (iyhw, ixhw)
    nsteps: int  # = args.ntrac
    itstep: int  # positive; direction is expressed via track()'s `step` sign
    min_score: "tuple[float, float]"
    max_velocity_change: "tuple[float, float]"
    min_contrast: "Optional[float]"
    min_peak_prominence: "Optional[float]"
    subgrid: "Optional[str]"
    method: str
    fixed_template: bool
    min_samples: int
    workers: "Optional[int]"
    forward: bool
    backward: bool
    ref_dt: float
    dtmean: float
    nt: int
    ny: int
    nx: int
    # (vyhw, vxhw) effective search velocity, for attrs only. `--Vs` as given
    # when the radius was derived from it; when `--hs` set the radius
    # directly, the equivalent velocity that radius covers (rather than the
    # ignored `--Vs`, which v1 and earlier v2 both misreported here).
    search_velocity: "tuple[float, float]"

    @classmethod
    def from_args(cls, args, frames: xr.Dataset, varname: str) -> "TrackingSetup":
        tname, yname, xname = frames[varname].dims
        t = data.build_time_seconds(frames, tname)
        grid = data.build_grid(frames, varname, xname, yname)
        ref_dt = data.compute_ref_dt(args, t)
        search_radius = data.compute_search_radius(args, grid, ref_dt)

        forward = args.ward in ("forward", "bothward")
        backward = args.ward in ("backward", "bothward")

        nt = frames[tname].size
        ny = frames[yname].size
        nx = frames[xname].size

        subgrid = None if args.subgrid == "none" else args.subgrid

        if args.hs:
            # --hs sets the pixel radius directly and --Vs is never consulted,
            # so report the velocity that radius actually covers.
            search_velocity = pyvttrac.velocity_from_search_radius(
                search_radius, dt=ref_dt * args.itstep, grid=grid
            )
        else:
            search_velocity = (args.Vs, args.Vs)

        return cls(
            grid=grid,
            template=(args.nsy, args.nsx),
            search_radius=search_radius,
            nsteps=args.ntrac,
            itstep=args.itstep,
            min_score=(args.Sth0, args.Sth1),
            max_velocity_change=(args.Vc, args.Vc),
            min_contrast=args.Cth,
            min_peak_prominence=args.peak_inside_th,
            subgrid=subgrid,
            method=args.method,
            fixed_template=args.use_init_temp,
            min_samples=args.min_samples,
            workers=args.workers,
            forward=forward,
            backward=backward,
            ref_dt=ref_dt,
            dtmean=float((t[-1] - t[0]) / (nt - 1)),
            nt=nt,
            ny=ny,
            nx=nx,
            search_velocity=search_velocity,
        )

    def make_tracker(self) -> "pyvttrac.Tracker":
        """One `Tracker`; forward/backward is selected per-call via `step=+-itstep`."""
        return pyvttrac.Tracker(
            template=self.template,
            search_radius=self.search_radius,
            nsteps=self.nsteps,
            method=self.method,
            subgrid=self.subgrid,
            step=self.itstep,
            min_score=self.min_score,
            max_velocity_change=self.max_velocity_change,
            min_contrast=self.min_contrast,
            min_peak_prominence=self.min_peak_prominence,
            fixed_template=self.fixed_template,
            min_samples=self.min_samples,
            workers=self.workers,
        )

    def to_attrs(self, z: np.ndarray, omega: float, mask: "Optional[np.ndarray]") -> dict:
        """v1's `vtt.attrs`, by key name (`20_finalize_tracking.py` reads
        `dtmean`/`xint`/`yint`/`polar` out of this via the output dataset)."""
        nsy, nsx = self.template
        iyhw, ixhw = self.search_radius
        vyhw, vxhw = self.search_velocity
        Sth0, Sth1 = self.min_score
        vych, vxch = self.max_velocity_change
        fmiss = np.finfo(np.float32).max
        chk_zmiss = bool(np.isnan(z).any()) or (omega != 0.0)
        return {
            "nt": self.nt, "ny": self.ny, "nx": self.nx,
            "dtmean": self.dtmean,
            "nsx": nsx, "nsy": nsy,
            "vxhw": vxhw, "vyhw": vyhw, "ixhw": ixhw, "iyhw": iyhw,
            "subgrid": self.subgrid is not None, "subgrid_gaus": self.subgrid == "gaussian",
            "itstep": self.itstep, "ntrac": self.nsteps,
            "score_method": self.method, "Sth0": Sth0, "Sth1": Sth1,
            "vxch": vxch, "vych": vych,
            "peak_inside_th": self.min_peak_prominence, "Cth": self.min_contrast,
            "use_init_temp": self.fixed_template, "min_samples": self.min_samples,
            "zmiss": fmiss if chk_zmiss else None, "fmiss": fmiss, "imiss": -999,
            "chk_zmiss": chk_zmiss, "chk_mask": mask is not None,
            "pyvttrac_version": pyvttrac.__version__,
        }
