"""Regression test for `20_finalize_tracking.py` / `opentcamv.finalize`.

Compares the *actual CLI entry point* (`scripts/20_finalize_tracking.py`,
via subprocess) against frozen golden output (`tests/data/finalize_golden/
*.npz`), captured once from the pre-refactor script against the same
deterministic synthetic candidates `finalize_conftest.make_finalize_candidates`
produces. This is deliberately entry-point-level, not a call into
`opentcamv.finalize` internals: it stays meaningful across pure
extraction, candidate-axis generalization, and single-file I/O, 
and only *should* start failing at a phase, where each
bug fix changes specific golden scenarios in specific, individually
-explained ways (see CHANGELOG.md once this phase lands).

--polar is deliberately excluded here: it crashes unconditionally in the
pre-refactor script (`valid_index = xr.ones_like(flows0.vx, ...)` hardcodes
the Cartesian variable name), so there is no golden behavior to preserve.
See test_finalize_masking.py for polar coverage post-fix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from finalize_conftest import make_finalize_candidates

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "20_finalize_tracking.py"
GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "finalize_golden"


def _run_finalize(tmp_path, extra_args, n_omega=4, ns=7) -> "xr.Dataset":
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=n_omega, ns=ns)
    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", str(ns), "--omega", *omega_strs, "-o", str(out),
        *extra_args,
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"20_finalize_tracking.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return xr.open_dataset(out)


def _assert_matches_golden(ds: "xr.Dataset", golden_name: str):
    golden = np.load(GOLDEN_DIR / f"{golden_name}.npz")
    assert set(ds.data_vars) == set(golden.files), (set(ds.data_vars), set(golden.files))
    for var in golden.files:
        actual = ds[var].values
        expected = golden[var]
        if actual.dtype.kind in "fc":
            assert np.array_equal(actual, expected, equal_nan=True), f"{var} differs from golden {golden_name!r}"
        else:
            assert np.array_equal(actual, expected), f"{var} differs from golden {golden_name!r}"


def test_matches_golden_main(tmp_path):
    ds = _run_finalize(
        tmp_path,
        ["--cthmax", "10", "--exclude", "stf", "stb", "score_ary", "psr", "--out_final_omega", "--out_final_ns"],
    )
    _assert_matches_golden(ds, "main")


def test_matches_golden_dangv_priority(tmp_path):
    ds = _run_finalize(
        tmp_path,
        ["--cthmax", "10", "--exclude", "stf", "stb", "score_ary", "psr", "--priority", "dangv", "--out_final_omega"],
    )
    _assert_matches_golden(ds, "dangv")


def test_matches_golden_irdiff(tmp_path):
    ds = _run_finalize(
        tmp_path,
        ["--IRdiff", "B13-B14", "--dIR", "2", "--exclude", "stf", "stb", "score_ary", "psr", "--out_final_omega"],
    )
    _assert_matches_golden(ds, "irdiff")


def test_polar_completes_without_crashing(tmp_path):
    """F10: the pre-refactor script crashed unconditionally in --polar mode
    (`xr.ones_like(flows0.vx, ...)` hardcoded the Cartesian variable name).
    No golden exists for this (it never produced output); this just pins
    that it now runs and produces the polar variable set."""
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=4, ns=7, polar=True)
    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "--omega", *omega_strs,
        "--exclude", "stf", "stb", "-o", str(out), "--out_final_omega",
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ds = xr.open_dataset(out)
    assert {"vr", "vt", "rloc", "aloc"} <= set(ds.data_vars)


def test_missing_exclude_does_not_crash(tmp_path):
    """F11: `--exclude` omitted raised `TypeError` (`used_vars + None`) in
    the pre-refactor script. Every known real invocation always passes
    --exclude, so this was never observed; fixed as part of the port
    (argparse default changed from `None` to `[]`)."""
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=2, ns=7)
    out = tmp_path / "final.nc"
    argv = [sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "--omega", *omega_strs, "-o", str(out)]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selector_recovers_spatially_varying_ground_truth(tmp_path):
    """Independent of the golden files: the synthetic data's true_omega_idx
    pattern (see finalize_conftest) must be exactly recovered, including at
    the "trap" points where the highest-scored candidate is wrong and only
    iterative rejection finds the right answer.

    At it=0 only, one point (y=3, x=8) is a deliberate exception (post-F5):
    its "true" candidate's own cthmax happens to land at 10.14 km at that
    timestep, just over the `--cthmax 10` threshold used here, and
    per-candidate cth screening (F5) correctly rejects it in favor of the
    next-best (lower-scored, but cth-valid) candidate. Before F5, this
    point "passed" only because screening used a *different* candidate's
    (the reference's) cthmax, which happened to be under the threshold --
    coincidentally right for the wrong reason. That's exactly the kind of
    case F5 is meant to fix, so it's excluded here (at it=0 only -- cthmax
    has its own per-it noise, and the other timesteps aren't affected)
    rather than papered over."""
    ds = _run_finalize(
        tmp_path,
        ["--cthmax", "10", "--exclude", "stf", "stb", "score_ary", "psr", "--out_final_omega"],
    )
    n_omega = 4
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    true_omega_idx = (np.arange(ny)[:, None] + np.arange(nx)[None, :]) % n_omega
    omegas = [round(0.0005 * i, 4) for i in range(n_omega)]
    expected = np.array([omegas[i] for i in true_omega_idx.ravel()]).reshape(ny, nx)
    for it in range(ds.sizes["it"]):
        expected_it = expected.copy()
        if it == 0:
            expected_it[3, 8] = 0.0  # see docstring: cthmax-rejected trap, not an algorithm failure
        np.testing.assert_allclose(ds["final_omega"].isel(it=it).values, expected_it)


def test_dangv_priority_reports_physical_omega(tmp_path):
    """F6: `--priority dangv` used to overwrite `flows_org["omega"]` with a
    plain index (`np.arange(...)`) right after using its physical values to
    compute `dangv`, so `--out_final_omega` reported indices (0, 1, 2, ...)
    instead of rad/s. `final_omega` must land in the actual omega set, not
    just any integer -- pinned independently of the opaque golden .npz."""
    ds = _run_finalize(
        tmp_path,
        ["--cthmax", "10", "--exclude", "stf", "stb", "score_ary", "psr", "--priority", "dangv", "--out_final_omega"],
    )
    n_omega = 4
    omegas = np.array([round(0.0005 * i, 4) for i in range(n_omega)])
    final_omega = ds["final_omega"].values
    valid = np.isfinite(final_omega)
    assert valid.any()
    assert np.isin(final_omega[valid], omegas).all(), final_omega[valid]
