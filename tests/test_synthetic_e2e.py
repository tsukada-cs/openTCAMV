"""End-to-end correctness against analytic solutions.

v1 can no longer be run (its juliacall dependency is gone), so "matches v1"
can't be the ground truth here. A synthetic field with a known analytic
velocity field is a *stronger* check anyway: it validates absolute
correctness, not just agreement with a possibly-buggy prior implementation.
This is the most important correctness gate in the whole migration --
especially the `--revrot` round-trip, which exercises the rotate-to-track /
add-rigid-rotation-back machinery that a pure schema/regression comparison
can't validate on its own.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from conftest import make_rigid_rotation_frames, make_uniform_advection_frames, run_tracking

NT, NY, NX = 8, 241, 241
DX = DY = 0.25
X0 = Y0 = -30.0
DT_SEC = 150.0
OMEGA0 = 0.001  # rad/s
VARNAME = "Z"

# A fine dx (relative to the domain) is deliberate: subgrid position error is
# roughly constant in *pixels* regardless of physical scale, so a coarse dx
# turns a small, perfectly normal pixel-level uncertainty into a large
# m/s error. --Vs is kept just above the true max speed (rather than a large
# safety margin) to keep the search window small -- a big window over a
# modest-uniqueness synthetic texture (see conftest.smooth_random_field)
# otherwise has room to lock onto an occasional spurious, unrelated peak.
COMMON_ARGS = [
    "--ns", "9", "--ntrac", "1", "--Sth0", "0.5", "--Sth1", "0.5", "--Cth", "0",
    "--Vs", "25", "--Vc", "20", "--Vd", "0", "--vlim", "0",
    "--xgran=-15:15", "--ygran=-15:15", "--xint", "3", "--yint", "3",
    "--traj_int", "1",
]


def _assert_recovers(vx, vy, x2d, y2d, u_true, v_true, err_tol=1.0, min_valid_frac=0.95):
    err = np.hypot(vx - u_true, vy - v_true)
    valid = np.isfinite(err)
    valid_frac = valid.mean()
    assert valid_frac >= min_valid_frac, f"only {valid_frac:.1%} of points tracked"
    ok_frac = (err[valid] < err_tol).mean()
    assert ok_frac >= 0.95, f"only {ok_frac:.1%} of tracked points within {err_tol} m/s (max err {np.nanmax(err):.2f})"


def test_rigid_rotation_no_revrot(tmp_path):
    frames = make_rigid_rotation_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, OMEGA0, VARNAME)
    ofn = run_tracking(tmp_path, frames, VARNAME, ["--revrot", "0.0", *COMMON_ARGS])
    ds = xr.open_dataset(ofn)

    it = NT // 2
    x2d, y2d = np.meshgrid(ds.x.values, ds.y.values)
    u_true = -OMEGA0 * y2d * 1000.0
    v_true = OMEGA0 * x2d * 1000.0
    _assert_recovers(ds["vx"].isel(it=it).values, ds["vy"].isel(it=it).values, x2d, y2d, u_true, v_true)


def test_rigid_rotation_with_matching_revrot(tmp_path):
    """The core --revrot validation: rotating the tracking window at the
    data's own true angular velocity should make the pattern trackable in
    the co-rotating frame, and adding the rigid-rotation velocity back
    should recover *the same* analytic solution as revrot=0."""
    frames = make_rigid_rotation_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, OMEGA0, VARNAME)
    ofn = run_tracking(tmp_path, frames, VARNAME, ["--revrot", f"{OMEGA0}", *COMMON_ARGS])
    ds = xr.open_dataset(ofn)

    it = NT // 2
    x2d, y2d = np.meshgrid(ds.x.values, ds.y.values)
    u_true = -OMEGA0 * y2d * 1000.0
    v_true = OMEGA0 * x2d * 1000.0
    _assert_recovers(ds["vx"].isel(it=it).values, ds["vy"].isel(it=it).values, x2d, y2d, u_true, v_true)


def test_rigid_rotation_polar(tmp_path):
    frames = make_rigid_rotation_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, OMEGA0, VARNAME)
    ofn = run_tracking(
        tmp_path, frames, VARNAME,
        ["--revrot", "0.0", "--polar", "--rgran", "5:15", "--rint", "5", "--nath", "16",
         "--ns", "9", "--ntrac", "1", "--Sth0", "0.5", "--Sth1", "0.5", "--Cth", "0",
         "--Vs", "25", "--Vc", "20", "--Vd", "0", "--vlim", "0", "--traj_int", "1"],
    )
    ds = xr.open_dataset(ofn)
    it = NT // 2
    vt = ds["vt"].isel(it=it).values
    vr = ds["vr"].isel(it=it).values
    r2d = ds["r"].values[:, None] * np.ones((1, ds["a"].size))
    vt_true = OMEGA0 * r2d * 1000.0

    valid = np.isfinite(vt) & np.isfinite(vr)
    assert valid.mean() >= 0.95
    assert (np.abs(vt[valid] - vt_true[valid]) < 1.0).mean() >= 0.95
    assert (np.abs(vr[valid]) < 1.0).mean() >= 0.95


def _advection_case(tmp_path, ward, vagg, traj_int="1"):
    u0, v0 = 15.0, -8.0
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, u0, v0, VARNAME)
    args = [
        "--revrot", "0.0", "--ward", ward, "--vagg", vagg,
        "--ns", "9", "--ntrac", "2", "--Sth0", "0.5", "--Sth1", "0.5", "--Cth", "0",
        "--Vs", "25", "--Vc", "20", "--Vd", "0", "--vlim", "0",
        "--xgran=-12:12", "--ygran=-12:12", "--xint", "3", "--yint", "3",
        "--traj_int", traj_int,
    ]
    ofn = run_tracking(tmp_path, frames, VARNAME, args, ofn_name=f"out_{ward}_{vagg}.nc")
    ds = xr.open_dataset(ofn)
    vx, vy = ds["vx"].values, ds["vy"].values
    valid = np.isfinite(vx) & np.isfinite(vy)
    assert valid.mean() >= 0.9, f"only {valid.mean():.1%} valid"
    err = np.hypot(vx - u0, vy - v0)
    ok_frac = (err[valid] < 1.0).mean()
    assert ok_frac >= 0.95, f"only {ok_frac:.1%} within 1 m/s (max err {np.nanmax(err[valid]):.2f})"


def test_uniform_advection_vagg_org(tmp_path):
    _advection_case(tmp_path, "bothward", "org")


def test_uniform_advection_vagg_mean(tmp_path):
    _advection_case(tmp_path, "bothward", "mean")


def test_uniform_advection_vagg_startend(tmp_path):
    _advection_case(tmp_path, "bothward", "startend")


def test_uniform_advection_forward(tmp_path):
    _advection_case(tmp_path, "forward", "mean")


def test_uniform_advection_backward(tmp_path):
    _advection_case(tmp_path, "backward", "mean")
