"""The relaxation-labeling selector skeleton. 
Not implemented -- these tests pin that the skeleton is wired up
correctly (CLI choice exists, dispatches, raises NotImplementedError) and
that the shared building blocks in `selectors.base` behave sanely, without
testing any actual relaxation-labeling algorithm (there isn't one yet).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from finalize_conftest import make_finalize_candidates
from opentcamv.finalize.selectors.base import candidate_distance, iterate_until_stable, neighborhood_reduce
from opentcamv.finalize.selectors.relaxation import RelaxationSelector

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "20_finalize_tracking.py"


def test_relaxation_selector_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        RelaxationSelector().select(None, window={}, max_epoch=1)


def test_cli_selector_relaxation_fails_clearly(tmp_path):
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=2, ns=7)
    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "--omega", *omega_strs, "-o", str(out),
        "--exclude", "stf", "stb", "score_ary", "psr", "--selector", "relaxation",
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr
    assert "relaxation" in result.stderr.lower()


def test_cli_selector_defaults_to_iterative_median(tmp_path):
    """--selector iterative_median (the default) must behave exactly as
    before -- no --selector flag at all is the common case."""
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=2, ns=7)
    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "--omega", *omega_strs, "-o", str(out),
        "--exclude", "stf", "stb", "score_ary", "psr",
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert np.isfinite(xr.open_dataset(out)["vx"].values).any()


def test_neighborhood_reduce_median():
    field = xr.DataArray(np.array([1.0, 2.0, 100.0, 4.0, 5.0]), dims="it")
    reduced = neighborhood_reduce(field, {"it": 3}, how="median")
    # center point (index 2): median of [2, 100, 4] = 4
    assert reduced.isel(it=2).item() == 4.0


def test_neighborhood_reduce_invalid_how():
    field = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="it")
    with pytest.raises(ValueError):
        neighborhood_reduce(field, {"it": 3}, how="bogus")


def test_candidate_distance_scalar():
    cand = xr.DataArray(np.array([5.0, 5.0]))
    ref = xr.DataArray(np.array([2.0, 8.0]))
    dist = candidate_distance(cand, ref)
    np.testing.assert_allclose(dist.values, [3.0, 3.0])


def test_candidate_distance_vector_matches_hypot():
    vx, vy = xr.DataArray([3.0]), xr.DataArray([4.0])
    vx_ref, vy_ref = xr.DataArray([0.0]), xr.DataArray([0.0])
    dist = candidate_distance((vx, vy), (vx_ref, vy_ref))
    np.testing.assert_allclose(dist.values, np.hypot(3.0, 4.0))


def test_iterate_until_stable_stops_early():
    calls = []

    def step_fn(state):
        calls.append(state)
        new_state = state + 1
        return new_state, new_state >= 3

    final_state, ep = iterate_until_stable(step_fn, 0, max_epoch=10)
    assert final_state == 3
    assert ep == 2  # 0-indexed: stabilized on the 3rd call
    assert len(calls) == 3


def test_iterate_until_stable_respects_max_epoch():
    def step_fn(state):
        return state + 1, False  # never stable

    final_state, ep = iterate_until_stable(step_fn, 0, max_epoch=5)
    assert final_state == 5
    assert ep == 4
