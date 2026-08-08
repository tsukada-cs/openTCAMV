"""--vagg (org/mean/startend), forward+backward trajectory concatenation
order, and np.flip direction -- against `TrackResult`-shaped dummy arrays
(no pyvttrac call needed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from opentcamv import aggregate
from opentcamv.params import TrackingSetup


@dataclass
class _FakeResult:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    score: np.ndarray
    status: np.ndarray
    templates: "Optional[np.ndarray]" = None
    score_grids: "Optional[np.ndarray]" = None


class _Grid:
    unit_factor = 1.0


class _Args:
    def __init__(self, ward="bothward", vagg="mean"):
        self.ward = ward
        self.vagg = vagg


def _setup(nsteps=2, itstep=1, forward=True, backward=True):
    return TrackingSetup(
        grid=_Grid(), template=(5, 5), search_radius=(3, 3), nsteps=nsteps, itstep=itstep,
        min_score=(0.8, 0.7), max_velocity_change=(20.0, 20.0), min_contrast=3.0,
        min_peak_prominence=None, subgrid="paraboloid", method="xcor", fixed_template=False,
        min_samples=1, workers=None, forward=forward, backward=backward, ref_dt=100.0,
        dtmean=100.0, nt=20, ny=1, nx=1, search_velocity=10.0,
    )


def _make_fwd_bwd(nsteps=2, seed_shape=(1,)):
    # x/y positions increase by 1 per step for forward, decrease for backward
    # (a trivial uniform-advection-like trajectory: vx=1 always).
    fwd_x = np.array([[0.0] + [float(i) for i in range(1, nsteps + 1)]]).T
    fwd = _FakeResult(
        x=np.broadcast_to(fwd_x, (nsteps + 1, *seed_shape)).copy(),
        y=np.zeros((nsteps + 1, *seed_shape)),
        vx=np.ones((nsteps, *seed_shape)),
        vy=np.zeros((nsteps, *seed_shape)),
        score=np.full((nsteps, *seed_shape), 0.9),
        status=np.zeros(seed_shape, dtype=np.int64),
    )
    bwd_x = np.array([[0.0] + [-float(i) for i in range(1, nsteps + 1)]]).T
    bwd = _FakeResult(
        x=np.broadcast_to(bwd_x, (nsteps + 1, *seed_shape)).copy(),
        y=np.zeros((nsteps + 1, *seed_shape)),
        vx=np.ones((nsteps, *seed_shape)),  # backward vx is also +1 (same flow direction)
        vy=np.zeros((nsteps, *seed_shape)),
        score=np.full((nsteps, *seed_shape), 0.8),
        status=np.zeros(seed_shape, dtype=np.int64),
    )
    return fwd, bwd


def test_vagg_mean_bothward_averages_and_combines():
    setup = _setup()
    fwd, bwd = _make_fwd_bwd()
    args = _Args("bothward", "mean")
    t = np.arange(20, dtype=np.float64) * 100.0
    pickup_it_rel = np.array([0, 1, 2])
    pickup_it_rel_v = np.array([0, 1])

    step = aggregate.aggregate_step(args, setup, fwd, bwd, tid0=2, t=t, grid=_Grid(), pickup_it_rel=pickup_it_rel, pickup_it_rel_v=pickup_it_rel_v)
    np.testing.assert_allclose(step.vxfm, 1.0)
    np.testing.assert_allclose(step.vxbm, 1.0)
    np.testing.assert_allclose(step.vx, 1.0)  # (1+1)/2
    np.testing.assert_allclose(step.vy, 0.0)


def test_vagg_org_bothward_concat_order_and_flip():
    setup = _setup(nsteps=2)
    fwd, bwd = _make_fwd_bwd(nsteps=2)
    # Make forward/backward vx distinguishable per-step to check ordering.
    fwd.vx = np.array([[10.0], [20.0]])
    bwd.vx = np.array([[-10.0], [-20.0]])
    args = _Args("bothward", "org")
    t = np.arange(20, dtype=np.float64) * 100.0
    pickup_it_rel = np.array([0, 1, 2])
    pickup_it_rel_v = np.array([0, 1])

    step = aggregate.aggregate_step(args, setup, fwd, bwd, tid0=2, t=t, grid=_Grid(), pickup_it_rel=pickup_it_rel, pickup_it_rel_v=pickup_it_rel_v)
    # bwd reversed (flip of [-10,-20] -> [-20,-10]) then fwd as-is ([10,20]):
    # concatenated -> [-20, -10, 10, 20]
    np.testing.assert_allclose(step.vx.ravel(), [-20.0, -10.0, 10.0, 20.0])


def test_vagg_startend_uses_grid_unit_factor_and_abs_dt():
    setup = _setup(nsteps=2, itstep=1)
    fwd, bwd = _make_fwd_bwd(nsteps=2)
    args = _Args("bothward", "startend")
    t = np.arange(20, dtype=np.float64) * 100.0  # dt per step = 100s

    class Grid2km:
        unit_factor = 1000.0  # km -> m

    step = aggregate.aggregate_step(args, setup, fwd, bwd, tid0=2, t=t, grid=Grid2km(), pickup_it_rel=np.array([0, 1, 2]), pickup_it_rel_v=np.array([0, 1]))
    # forward: (x[-1]-x[0]) = 2.0 km over 200s * 1000 (unit_factor) = 10 m/s
    np.testing.assert_allclose(step.vxfm, 10.0)
    # backward: (x[0]-x[-1]) = 0 - (-2.0) = 2.0 km over 200s * 1000 = 10 m/s
    np.testing.assert_allclose(step.vxbm, 10.0)
    np.testing.assert_allclose(step.vx, 10.0)


def test_forward_only_uses_fwd_directly():
    setup = _setup(forward=True, backward=False)
    fwd, bwd = _make_fwd_bwd()
    args = _Args("forward", "mean")
    t = np.arange(20, dtype=np.float64) * 100.0
    step = aggregate.aggregate_step(args, setup, fwd, None, tid0=2, t=t, grid=_Grid(), pickup_it_rel=np.array([0, 1, 2]), pickup_it_rel_v=np.array([0, 1]))
    assert step.stb is None
    np.testing.assert_array_equal(step.stf, fwd.status)
    np.testing.assert_allclose(step.vx, 1.0)


def test_backward_only_uses_bwd_directly():
    setup = _setup(forward=False, backward=True)
    fwd, bwd = _make_fwd_bwd()
    args = _Args("backward", "mean")
    t = np.arange(20, dtype=np.float64) * 100.0
    step = aggregate.aggregate_step(args, setup, None, bwd, tid0=2, t=t, grid=_Grid(), pickup_it_rel=np.array([0, 1, 2]), pickup_it_rel_v=np.array([0, 1]))
    assert step.stf is None
    np.testing.assert_array_equal(step.stb, bwd.status)


def test_trajectory_concat_bothward_excludes_duplicate_seed_point():
    setup = _setup(nsteps=2)
    fwd, bwd = _make_fwd_bwd(nsteps=2)
    args = _Args("bothward", "mean")
    t = np.arange(20, dtype=np.float64) * 100.0
    step = aggregate.aggregate_step(args, setup, fwd, bwd, tid0=2, t=t, grid=_Grid(), pickup_it_rel=np.array([0, 1, 2]), pickup_it_rel_v=np.array([0, 1]))
    # bwd x: [0,-1,-2] (it_rel 0,-1,-2); pickup_it_rel[1:] = [1,2] -> flip -> [2,1]
    # bwd.x at those indices: [-2, -1]; fwd x at [0,1,2]: [0,1,2]
    # concatenated xtraj: [-2, -1, 0, 1, 2]
    np.testing.assert_allclose(step.xtraj.ravel(), [-2.0, -1.0, 0.0, 1.0, 2.0])
