"""Multi-`ns` candidate loading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from finalize_conftest import make_finalize_candidates

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "20_finalize_tracking.py"


def _write_two_ns_candidates(tmp_path):
    """A minimal, hand-built (not `finalize_conftest`-generated) scenario:
    two omega values, ns=7 and ns=9. At grid point (0, 0), every ns=7
    candidate has a score below `--score_th` (invalid everywhere for that
    ns), while ns=9's candidates score well there and match the true
    field. Elsewhere, both ns are valid and equally good. This isolates "a
    point invalid at the smaller ns is rescued by a larger one" from any
    other effect.

    (Deliberately uses `--score_th`, not `--cthmax`, for the invalidity:
    `--score_th` masks per-candidate (`flows_org["score"]`, which has the
    full (ns, omega) structure), whereas `--cthmax` is still evaluated from
    a single reference candidate only at this point in the refactor (F5,
    not yet fixed) -- a cthmax-based version of this test would be killed
    by that shared masking regardless of which ns actually supports the
    point, conflating "F1 fixed" with "F5 fixed".

    NOTE: `--score_th` *keeps* points with `max(score) <= score_th` and
    *rejects* the rest (`.where(score.max("it_rel_v") <= score_th)`) --
    inverted from the usual "reject low scores" reading. So candidates are
    scored *low* (0.3) by default, kept under `--score_th 0.5`; the "make
    ns=7 invalid at (0,0)" trick is to give it a *high* score (0.9) there
    instead, pushing it above the threshold and out.)
    """
    ny, nx, nit = 3, 3, 4
    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    it = np.arange(nit)
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(it * 150, unit="s")
    it_rel = np.array([-1, 0, 1])
    it_rel_v = np.array([-0.5, 0.5])

    vx_true = np.full((nit, ny, nx), 6.0)
    vy_true = np.full((nit, ny, nx), -3.0)

    omega_strs = ["0.0000", "0.0005"]
    for ns, rescue_point_valid in [(7, False), (9, True)]:
        for omega in omega_strs:
            cthmax = np.full((nit, ny, nx), 5.0)  # always valid; not the mechanism under test here

            score = np.full((nit, ny, nx), 0.3)  # <= --score_th everywhere: valid by default
            if not rescue_point_valid:
                score[:, 0, 0] = 0.9  # > --score_th: rejected only for ns=7 at the rescue point
            vx = vx_true.copy()
            vy = vy_true.copy()

            xloc = np.broadcast_to(x[None, None, None, :], (nit, len(it_rel), ny, nx)).astype(np.float64).copy()
            yloc = np.broadcast_to(y[None, None, :, None], (nit, len(it_rel), ny, nx)).astype(np.float64).copy()
            score_v = np.broadcast_to(score[:, None], (nit, len(it_rel_v), ny, nx)).astype(np.float32).copy()
            it_plus_it_rel = np.clip(it[:, None] + it_rel[None, :], 0, nit - 1)
            time2 = time.values[it_plus_it_rel]

            ds = xr.Dataset(
                data_vars=dict(
                    vx=(["it", "y", "x"], vx.astype(np.float32)),
                    vy=(["it", "y", "x"], vy.astype(np.float32)),
                    xloc=(["it", "it_rel", "y", "x"], xloc.astype(np.float32)),
                    yloc=(["it", "it_rel", "y", "x"], yloc.astype(np.float32)),
                    score=(["it", "it_rel_v", "y", "x"], score_v),
                    cth=(["it", "y", "x"], (cthmax - 1.0).astype(np.float32)),
                    cthmax=(["it", "y", "x"], cthmax.astype(np.float32)),
                    time2=(["it", "it_rel"], time2),
                ),
                coords=dict(
                    it=("it", it), time=("it", time),
                    y=("y", y, {"units": "km"}), x=("x", x, {"units": "km"}),
                    it_rel=("it_rel", it_rel), it_rel_v=("it_rel_v", it_rel_v),
                ),
                attrs=dict(
                    polar=0, dtmean=150.0, xint=1.0, yint=1.0, rint=1.0, nath=60,
                    itstep=1, ntrac=1, nsx=ns, nsy=ns,
                ),
            )
            ds.to_netcdf(tmp_path / f"cand_ns{ns}_rot{omega}.nc")

    ifns_rule = str(tmp_path / "cand_ns<ns>_rot<omega>.nc")
    return ifns_rule, omega_strs


def test_multi_ns_completes_without_error(tmp_path):
    """F1: `--ns 7 9` used to raise `ValueError: conflicting sizes for
    dimension 'omega'`.

    NOTE: doesn't check `--out_final_ns`'s output -- that flag doesn't
    actually output anything (F13, pre-existing, not introduced by this
    port: `final_ns` is assigned onto an intermediate Dataset but never
    copied into the returned one). This test only exercises loading +
    selection completing successfully."""
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=2, ns=7, fname_prefix="a")
    # A second ns, reusing the same synthetic field/omegas (different seed
    # so it's not a byte-identical duplicate).
    make_finalize_candidates(tmp_path, n_omega=2, ns=9, fname_prefix="a", seed=1)

    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "9", "--omega", *omega_strs,
        "--exclude", "stf", "stb", "score_ary", "psr", "-o", str(out),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ds = xr.open_dataset(out)
    assert np.isfinite(ds["vx"].values).any()


def test_larger_ns_rescues_point_invalid_at_smaller_ns(tmp_path):
    ifns_rule, omega_strs = _write_two_ns_candidates(tmp_path)
    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "9", "--omega", *omega_strs,
        "--cthmax", "10", "--score_th", "0.5", "--exclude", "stf", "stb", "score_ary", "psr",
        "-o", str(out),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ds = xr.open_dataset(out)

    # At (0,0): ns=7 is invalid everywhere (score=0.9 > --score_th 0.5), so
    # only ns=9 can supply a valid answer there. If F1's crash were merely
    # papered over (e.g. silently falling back to a single ns) rather than
    # properly fixed, this point would come out NaN; recovering the true
    # velocity here proves ns=9's candidate was actually used.
    assert np.isfinite(ds["vx"].isel(it=0, y=0, x=0).item())
    np.testing.assert_allclose(ds["vx"].isel(it=0, y=0, x=0).item(), 6.0, atol=0.5)
    np.testing.assert_allclose(ds["vy"].isel(it=0, y=0, x=0).item(), -3.0, atol=0.5)

    # Elsewhere, both ns are equally valid; either is an acceptable choice,
    # but the velocity should still recover the true field.
    np.testing.assert_allclose(ds["vx"].isel(it=0, y=1, x=1).item(), 6.0, atol=0.5)
    np.testing.assert_allclose(ds["vy"].isel(it=0, y=1, x=1).item(), -3.0, atol=0.5)
