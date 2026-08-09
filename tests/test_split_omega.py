"""`10_conduct_tracking.py`'s output format for multiple `--revrot` values 
-- default single combined file with an `omega` dimension, vs. 
`--split_omega`'s legacy one-file-per-Omega. The gate is numeric parity: 
the same run must produce the same numbers in either format, just packaged differently.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from conftest import SCRIPT_10, make_uniform_advection_frames

NT, NY, NX = 6, 41, 41
DX = DY = 0.5
X0 = Y0 = -10.0
DT_SEC = 150.0
VARNAME = "Z"
REVROTS = [0.0, 0.0007]

COMMON_ARGS = [
    "--ns", "7", "--ntrac", "1", "--Sth0", "0.5", "--Sth1", "0.5", "--Cth", "0",
    "--Vs", "20", "--Vc", "20", "--Vd", "0", "--vlim", "0",
    "--xgran=-6:6", "--ygran=-6:6", "--xint", "3", "--yint", "3",
    "--traj_int", "1",
]


def _run_10(tmp_path, ifn, ofn, extra_args):
    argv = [sys.executable, str(SCRIPT_10), str(ifn), "--varname", VARNAME, "-o", str(ofn), *extra_args]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"10_conduct_tracking.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def test_default_multi_revrot_produces_single_omega_dimensioned_file(tmp_path):
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)
    ofn = tmp_path / "combined.nc"
    _run_10(tmp_path, ifn, ofn, ["--revrot", *[str(o) for o in REVROTS], *COMMON_ARGS])

    ds = xr.open_dataset(ofn)
    assert "omega" in ds["vx"].dims
    np.testing.assert_allclose(ds["omega"].values, REVROTS)
    np.testing.assert_allclose(np.asarray(ds.attrs["revrot"], dtype=float), REVROTS)


def test_split_omega_matches_default_combined_file(tmp_path):
    """Same run, two output formats -- must agree exactly per-Omega."""
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)

    combined_ofn = tmp_path / "combined.nc"
    _run_10(tmp_path, ifn, combined_ofn, ["--revrot", *[str(o) for o in REVROTS], *COMMON_ARGS])
    combined = xr.open_dataset(combined_ofn)

    split_ofn = tmp_path / "split_rot<omega>.nc"
    _run_10(tmp_path, ifn, split_ofn, ["--revrot", *[str(o) for o in REVROTS], "--split_omega", *COMMON_ARGS])

    for omega in REVROTS:
        split_path = Path(str(split_ofn).replace("<omega>", f"{omega:.4f}"))
        assert split_path.exists()
        split_ds = xr.open_dataset(split_path)
        combined_slice = combined.sel(omega=omega)
        for var in ("vx", "vy", "xloc", "yloc", "score"):
            np.testing.assert_array_equal(split_ds[var].values, combined_slice[var].values)
        assert split_ds.attrs["revrot"] == omega


def test_single_revrot_value_has_no_omega_dim_regardless_of_split_omega(tmp_path):
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)

    ofn = tmp_path / "single.nc"
    _run_10(tmp_path, ifn, ofn, ["--revrot", "0.0", "--split_omega", *COMMON_ARGS])
    ds = xr.open_dataset(ofn)
    assert "omega" not in ds.dims
    assert ds.attrs["revrot"] == 0.0


def test_record_initpos_and_time2_have_no_omega_dim_in_combined_output(tmp_path):
    """--record_initpos outputs and time2 are initial-position/frame-time
    quantities, independent of the rotation applied during tracking -- the
    combined file must not duplicate them across the omega dimension."""
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)
    ofn = tmp_path / "combined.nc"
    _run_10(tmp_path, ifn, ofn, ["--revrot", *[str(o) for o in REVROTS], "--record_initpos", VARNAME, *COMMON_ARGS])

    ds = xr.open_dataset(ofn)
    assert "omega" not in ds[VARNAME].dims
    assert "omega" not in ds["time2"].dims
    assert "omega" in ds["vx"].dims


def test_record_initpos_dedup_matches_every_omega(tmp_path):
    """The single copy kept in the combined file must equal what every
    Omega's own (identical, since it's omega-independent) value would be --
    checked against --split_omega's per-file copies."""
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)

    combined_ofn = tmp_path / "combined.nc"
    _run_10(
        tmp_path, ifn, combined_ofn,
        ["--revrot", *[str(o) for o in REVROTS], "--record_initpos", VARNAME, *COMMON_ARGS],
    )
    combined = xr.open_dataset(combined_ofn)

    split_ofn = tmp_path / "split_rot<omega>.nc"
    _run_10(
        tmp_path, ifn, split_ofn,
        ["--revrot", *[str(o) for o in REVROTS], "--split_omega", "--record_initpos", VARNAME, *COMMON_ARGS],
    )
    for omega in REVROTS:
        split_ds = xr.open_dataset(Path(str(split_ofn).replace("<omega>", f"{omega:.4f}")))
        np.testing.assert_array_equal(combined[VARNAME].values, split_ds[VARNAME].values)


def test_stf_stb_and_float_fields_are_correctly_sized(tmp_path):
    """vx/vy/xloc/yloc/--record_initpos outputs must stay float32
    (--revrot restoration and interp() both silently promote to float64
    if not cast back), and stf/stb (status codes, range -10..11) fit
    comfortably in int8."""
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)
    ofn = tmp_path / "out.nc"
    _run_10(tmp_path, ifn, ofn, ["--revrot", "0.0007", "--record_initpos", VARNAME, *COMMON_ARGS])

    ds = xr.open_dataset(ofn)
    for var in ("vx", "vy", "xloc", "yloc", VARNAME):
        assert ds[var].dtype == np.float32, f"{var} is {ds[var].dtype}, expected float32"
    for var in ("stf", "stb"):
        assert ds[var].dtype == np.int8, f"{var} is {ds[var].dtype}, expected int8"


def test_split_omega_without_placeholder_is_rejected(tmp_path):
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)

    argv = [
        sys.executable, str(SCRIPT_10), str(ifn), "--varname", VARNAME, "-o", str(tmp_path / "out.nc"),
        "--revrot", *[str(o) for o in REVROTS], "--split_omega", *COMMON_ARGS,
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode != 0
    assert "<omega>" in result.stderr
