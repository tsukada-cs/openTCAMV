"""`opentcamv.finalize.masking.validity_mask`.

F5: cthmin/cthmax screening used to be evaluated once from a single
reference candidate (the first ns x first omega file) and applied
uniformly to every candidate, even though cthmax/cthmin are trajectory
quantities that genuinely depend on which candidate (omega, ns) produced
them. Fixed to evaluate each candidate against its own cthmax/cthmin.

See also test_finalize_regression.py::test_matches_golden_main /
test_matches_golden_dangv_priority / test_matches_golden_irdiff, which pin
the real-data-shaped synthetic fixture's golden output including this fix,
and test_selector_recovers_spatially_varying_ground_truth, which documents
one incidental case (that fixture's random cthmax noise pushing a "true"
candidate just over the threshold at one point/timestep).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "20_finalize_tracking.py"


def _write_two_omega_candidates(tmp_path, *, ns=7):
    """Two omega candidates, ns=7 only. Everywhere except (0, 0), omega=0.0005
    is the better answer (higher score) and has a low cthmax (valid). At
    (0, 0) specifically, omega=0.0005's own cthmax is 15 (> the --cthmax 10
    threshold used in the test), so per-candidate screening must reject it
    there and fall back to omega=0.0000 -- even though omega=0.0005 is still
    the higher-scoring candidate globally, and even though the *reference*
    candidate (omega=0.0000, opened first) has a low cthmax everywhere,
    including at (0, 0). A pre-F5 (reference-only) screening would never
    reject omega=0.0005 at (0, 0), since it only ever looks at the
    reference's cthmax.
    """
    ny, nx, nit = 3, 3, 4
    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    it = np.arange(nit)
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(it * 150, unit="s")
    it_rel = np.array([-1, 0, 1])
    it_rel_v = np.array([-0.5, 0.5])

    vx_wrong, vy_wrong = 2.0, -1.0  # omega=0.0000's answer
    vx_true, vy_true = 8.0, -5.0  # omega=0.0005's answer

    omega_strs = ["0.0000", "0.0005"]
    for omega_idx, omega in enumerate(omega_strs):
        is_winner = omega_idx == 1
        vx = np.full((nit, ny, nx), vx_true if is_winner else vx_wrong)
        vy = np.full((nit, ny, nx), vy_true if is_winner else vy_wrong)
        score = np.full((nit, ny, nx), 0.9 if is_winner else 0.5)

        cthmax = np.full((nit, ny, nx), 5.0)
        if is_winner:
            cthmax[:, 0, 0] = 15.0  # only this candidate, only at this point, exceeds --cthmax 10

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

    ifns_rule = str(tmp_path / f"cand_ns<ns>_rot<omega>.nc")
    return ifns_rule, omega_strs


def test_per_candidate_cth_rejects_winner_only_where_its_own_cth_is_bad(tmp_path):
    ifns_rule, omega_strs = _write_two_omega_candidates(tmp_path)
    out = tmp_path / "final.nc"
    argv = [
        sys.executable, str(SCRIPT), ifns_rule, "--ns", "7", "--omega", *omega_strs,
        "--cthmax", "10", "--exclude", "stf", "stb", "score_ary", "psr", "-o", str(out),
        # Large enough to never reject on neighborhood-consistency grounds --
        # this test isolates cth screening, not the iterative rejection loop.
        "--dth", "1000", "--dc", "1000",
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ds = xr.open_dataset(out)

    # At (0, 0): the higher-scoring candidate's own cthmax (15) exceeds the
    # threshold, so it must be rejected there -- even though the reference
    # candidate's cthmax (5) would have passed it.
    np.testing.assert_allclose(ds["vx"].isel(it=0, y=0, x=0).item(), 2.0)
    np.testing.assert_allclose(ds["vy"].isel(it=0, y=0, x=0).item(), -1.0)

    # Everywhere else, the higher-scoring candidate's own cthmax is fine,
    # so it wins as expected.
    np.testing.assert_allclose(ds["vx"].isel(it=0, y=1, x=1).item(), 8.0)
    np.testing.assert_allclose(ds["vy"].isel(it=0, y=1, x=1).item(), -5.0)

    # The final cthmax reported at each point must be the *selected*
    # candidate's own value, not always the reference's.
    np.testing.assert_allclose(ds["cthmax"].isel(it=0, y=0, x=0).item(), 5.0)
    np.testing.assert_allclose(ds["cthmax"].isel(it=0, y=1, x=1).item(), 5.0)
