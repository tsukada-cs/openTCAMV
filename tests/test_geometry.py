"""Seed generation (Cartesian/polar) and vx,vy <-> vr,vt round trips."""

from __future__ import annotations

import numpy as np

from opentcamv import aggregate, output


class _Args:
    def __init__(self, **kw):
        self.polar = False
        self.xgran = slice(-10, 10)
        self.ygran = slice(-10, 10)
        self.xint = 5.0
        self.yint = 5.0
        self.rgran = slice(4, 20)
        self.rint = 4.0
        self.nath = 8
        self.__dict__.update(kw)


def test_axis_names_cartesian_vs_polar():
    cart = output.axis_names(_Args(polar=False))
    assert (cart.dim1, cart.dim2, cart.loc1, cart.loc2, cart.v1, cart.v2) == ("y", "x", "yloc", "xloc", "vy", "vx")
    polar = output.axis_names(_Args(polar=True))
    assert (polar.dim1, polar.dim2, polar.loc1, polar.loc2, polar.v1, polar.v2) == ("r", "a", "rloc", "aloc", "vr", "vt")


def test_build_seed_positions_cartesian_grid():
    args = _Args(polar=False, xgran=slice(-10, 10), ygran=slice(-10, 10), xint=5.0, yint=5.0)
    xxg, yyg = output.build_seed_positions(args)
    assert xxg.shape == yyg.shape == (5, 5)
    np.testing.assert_allclose(xxg[0], [-10, -5, 0, 5, 10])
    np.testing.assert_allclose(yyg[:, 0], [-10, -5, 0, 5, 10])


def test_build_seed_positions_polar_matches_r_a_grid():
    args = _Args(polar=True, rgran=slice(4, 12), rint=4.0, nath=4)
    xxg, yyg = output.build_seed_positions(args)
    r = np.hypot(xxg, yyg)
    a = np.arctan2(yyg, xxg) % (2 * np.pi)
    expected_r = np.arange(4, 12 + 4, 4)
    np.testing.assert_allclose(r[:, 0], expected_r, atol=1e-10)
    # azimuth 0 is along +x for every radius
    np.testing.assert_allclose(a[:, 0], 0.0, atol=1e-10)


def test_vx_vy_to_vr_vt_roundtrip():
    rng = np.random.RandomState(0)
    ag = np.linspace(0, 2 * np.pi, 9)[:-1]
    costh, sinth = np.cos(ag), np.sin(ag)
    vx = rng.uniform(-10, 10, size=(3, ag.size))
    vy = rng.uniform(-10, 10, size=(3, ag.size))

    vr, vt = aggregate.to_polar(vx, vy, costh, sinth)
    # Inverse: vx = vr*cos - vt*sin, vy = vr*sin + vt*cos
    vx_back = vr * costh[None, :] - vt * sinth[None, :]
    vy_back = vr * sinth[None, :] + vt * costh[None, :]
    np.testing.assert_allclose(vx_back, vx, atol=1e-10)
    np.testing.assert_allclose(vy_back, vy, atol=1e-10)


def test_pure_tangential_velocity_is_all_vt():
    # A rigid-rotation-like field: (vx, vy) = (-y, x) at unit radius is purely tangential.
    ag = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    costh, sinth = np.cos(ag), np.sin(ag)
    x, y = costh, sinth
    vx, vy = -y, x
    vr, vt = aggregate.to_polar(vx[None, :], vy[None, :], costh, sinth)
    np.testing.assert_allclose(vr, 0.0, atol=1e-10)
    np.testing.assert_allclose(vt, 1.0, atol=1e-10)


def test_traj_to_polar():
    xtraj = np.array([[3.0, 0.0], [0.0, -5.0]])
    ytraj = np.array([[4.0, 0.0], [0.0, 0.0]])
    rtraj, atraj = aggregate.traj_to_polar(xtraj, ytraj)
    np.testing.assert_allclose(rtraj, [[5.0, 0.0], [0.0, 5.0]])
    np.testing.assert_allclose(atraj[1, 1], np.pi)
