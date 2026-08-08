from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import ndimage

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_10 = SCRIPTS_DIR / "10_conduct_tracking.py"


def smooth_random_field(ny: int, nx: int, X: np.ndarray, Y: np.ndarray, seed: int, corr_length_px: float = 1.5) -> np.ndarray:
    """A smooth, texture-rich scalar field, as Gaussian-smoothed white noise
    (correlation length `corr_length_px` pixels). Unlike a small sum of
    low-frequency sinusoids, this has no long-range periodicity, so a small
    template's best match is genuinely unique nearby -- a sum-of-sinusoids
    field turned out to alias, giving a *higher*-correlation but spurious
    match many pixels away from the true displacement, which made template
    tracking fail even though the tracker itself was working correctly.
    `X`/`Y` are unused (kept for signature symmetry with callers); only
    their shape `(ny, nx)` matters.
    """
    rng = np.random.RandomState(seed)
    noise = rng.standard_normal((ny, nx))
    z0 = ndimage.gaussian_filter(noise, sigma=corr_length_px, mode="wrap")
    z0 = (z0 - z0.mean()) / z0.std()
    return (50.0 + 15.0 * z0).astype(np.float32)


def make_rigid_rotation_frames(
    nt: int, ny: int, nx: int, dx: float, dy: float, x0: float, y0: float,
    dt_sec: float, omega: float, varname: str = "Z", seed: int = 0,
) -> xr.Dataset:
    """Synthetic (t, y, x) data rigidly rotating at angular velocity `omega`
    (rad/s, CCW positive, standard math convention): the analytic velocity
    field is `u = -omega*y*1000`, `v = +omega*x*1000` (m/s), for x/y in km.
    """
    yv = y0 + dy * np.arange(ny)
    xv = x0 + dx * np.arange(nx)
    X, Y = np.meshgrid(xv, yv)
    z0 = smooth_random_field(ny, nx, X, Y, seed)

    t = np.arange(nt) * dt_sec
    z = np.empty((nt, ny, nx), dtype=np.float32)
    for i in range(nt):
        theta = omega * t[i]
        # Sample position (X, Y) at time t[i] originated, at t=0, from the
        # position rotated *backward* by theta (so the pattern rotates
        # forward by theta as time advances).
        Xs = X * np.cos(theta) + Y * np.sin(theta)
        Ys = -X * np.sin(theta) + Y * np.cos(theta)
        ix = (Xs - x0) / dx
        iy = (Ys - y0) / dy
        z[i] = ndimage.map_coordinates(z0, [iy, ix], order=3, mode="nearest")

    return _build_dataset(z, t, xv, yv, varname)


def make_uniform_advection_frames(
    nt: int, ny: int, nx: int, dx: float, dy: float, x0: float, y0: float,
    dt_sec: float, u0: float, v0: float, varname: str = "Z", seed: int = 0,
) -> xr.Dataset:
    """Synthetic (t, y, x) data translating at constant velocity `(u0, v0)`
    m/s (analytic solution: `vx = u0`, `vy = v0` everywhere)."""
    yv = y0 + dy * np.arange(ny)
    xv = x0 + dx * np.arange(nx)
    X, Y = np.meshgrid(xv, yv)
    z0 = smooth_random_field(ny, nx, X, Y, seed)

    t = np.arange(nt) * dt_sec
    z = np.empty((nt, ny, nx), dtype=np.float32)
    for i in range(nt):
        dx_km = u0 * t[i] / 1000.0
        dy_km = v0 * t[i] / 1000.0
        Xs = X - dx_km
        Ys = Y - dy_km
        ix = (Xs - x0) / dx
        iy = (Ys - y0) / dy
        z[i] = ndimage.map_coordinates(z0, [iy, ix], order=3, mode="nearest")

    return _build_dataset(z, t, xv, yv, varname)


def _build_dataset(z: np.ndarray, t_sec: np.ndarray, xv: np.ndarray, yv: np.ndarray, varname: str) -> xr.Dataset:
    time = pd.Timestamp("2020-01-01T00:00:00") + pd.to_timedelta(t_sec, unit="s")
    ds = xr.Dataset(
        data_vars={varname: (["t", "y", "x"], z)},
        coords={
            "t": ("t", time),
            "y": ("y", yv, {"units": "km"}),
            "x": ("x", xv, {"units": "km"}),
        },
    )
    return ds


def run_tracking(tmp_path: Path, frames: xr.Dataset, varname: str, extra_args: "list[str]", ofn_name: str = "out.nc") -> Path:
    """Write `frames` to NetCDF, run `10_conduct_tracking.py` against it via
    subprocess (the actual CLI entry point), and return the output path
    (with `<omega>` substituted if present in `extra_args`/default)."""
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)
    ofn = tmp_path / ofn_name

    argv = [sys.executable, str(SCRIPT_10), str(ifn), "--varname", varname, "-o", str(ofn), *extra_args]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"10_conduct_tracking.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return ofn


@pytest.fixture
def tmp_scratch(tmp_path):
    return tmp_path
