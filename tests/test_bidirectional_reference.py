"""Cross-check openTCAMV's hand-rolled forward+backward stitching against
pyVTTrac's canonical `concat_bidirectional()`.

`aggregate.aggregate_step()` stitches the two legs itself (it has to: it
also applies `--traj_int` subsampling and the `--vagg` aggregation modes,
neither of which `concat_bidirectional()` covers). pyVTTrac's own changelog
notes that the backward leg's velocities must *not* be sign-flipped when
reversing it -- `vx = (xw - xcur) / dt` is already correctly signed for
`dt < 0` -- and that hand-rolled stitching is exactly where that gets got
wrong. So rather than trust the two implementations agree, pin it: run real
tracking, combine it both ways, and require an exact match.
"""

from __future__ import annotations

import numpy as np
import pyvttrac as vt
import pytest

from conftest import make_uniform_advection_frames
from opentcamv import aggregate
from opentcamv.params import TrackingSetup

NT, NY, NX = 9, 121, 121
DX = DY = 0.5
X0 = Y0 = -30.0
DT_SEC = 150.0
NSTEPS = 2
VARNAME = "Z"


class _Args:
    ward = "bothward"
    vagg = "org"


def _track_both_legs():
    """Real forward/backward tracking from a shared origin, plus the
    `TrackingSetup` describing it."""
    frames = make_uniform_advection_frames(
        NT, NY, NX, DX, DY, X0, Y0, DT_SEC, u0=12.0, v0=-7.0, varname=VARNAME
    )
    z = frames[VARNAME].values.astype(np.float32)
    t = np.arange(NT, dtype=np.float64) * DT_SEC
    grid = vt.Grid(x0=X0, y0=Y0, dx=DX, dy=DY, unit_factor=1000.0)

    xs = np.arange(-12.0, 12.1, 4.0)
    xxg, yyg = np.meshgrid(xs, xs)

    tracker = vt.Tracker(
        template=(9, 9), search_radius=(8, 8), nsteps=NSTEPS, step=1,
        min_score=(0.5, 0.5), subgrid="paraboloid",
    )
    t0 = NT // 2
    fwd = tracker.track(z, xxg, yyg, t0=t0, step=+1, time=t, grid=grid)
    bwd = tracker.track(z, xxg, yyg, t0=t0, step=-1, time=t, grid=grid)

    setup = TrackingSetup(
        grid=grid, template=(9, 9), search_radius=(8, 8), nsteps=NSTEPS, itstep=1,
        min_score=(0.5, 0.5), max_velocity_change=(40.0, 40.0), min_contrast=None,
        min_peak_prominence=None, subgrid="paraboloid", method="xcor",
        fixed_template=False, min_samples=1, workers=None, forward=True, backward=True,
        ref_dt=DT_SEC, dtmean=DT_SEC, nt=NT, ny=NY, nx=NX, search_velocity=(40.0, 40.0),
    )
    return fwd, bwd, setup, t, t0, grid


def test_stitched_trajectory_matches_concat_bidirectional():
    fwd, bwd, setup, t, t0, grid = _track_both_legs()
    combined = vt.concat_bidirectional(fwd, bwd)

    # --traj_int=1: every tracked step lands on the output grid.
    pickup_it_rel = np.arange(NSTEPS + 1)
    pickup_it_rel_v = np.arange(NSTEPS)
    step = aggregate.aggregate_step(
        _Args(), setup, fwd, bwd, tid0=t0, t=t, grid=grid,
        pickup_it_rel=pickup_it_rel, pickup_it_rel_v=pickup_it_rel_v,
    )

    # Positions: backward leg reversed (shared origin dropped) then forward.
    np.testing.assert_array_equal(step.xtraj, combined.x)
    np.testing.assert_array_equal(step.ytraj, combined.y)
    # Velocities: same order, and -- the point of this test -- the same signs.
    np.testing.assert_array_equal(step.vx, combined.vx)
    np.testing.assert_array_equal(step.vy, combined.vy)
    np.testing.assert_array_equal(step.score, combined.score)


def test_backward_leg_velocity_is_not_sign_flipped():
    """The sign convention above, stated independently of `concat_bidirectional`:
    a steady flow must give the same-signed velocity from either leg."""
    fwd, bwd, setup, t, t0, grid = _track_both_legs()
    ok = np.isfinite(fwd.vx) & np.isfinite(bwd.vx)
    assert ok.sum() > 0
    # Uniform advection at u0=+12: both legs must report ~+12, not +-12.
    assert np.nanmedian(fwd.vx[ok]) > 0
    assert np.nanmedian(bwd.vx[ok]) > 0
    np.testing.assert_allclose(
        np.nanmedian(fwd.vx[ok]), np.nanmedian(bwd.vx[ok]), rtol=0.05
    )
