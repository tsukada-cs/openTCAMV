"""`opentcamv.concat` (extracted, unchanged, from `11_concat_flows_along_time.py`) 
must keep working when input files carry the new `omega` dimension -- concatenation is along
`time` only, `omega` is never the concat axis and must survive untouched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import opentcamv

SCRIPT_11 = Path(__file__).resolve().parent.parent / "scripts" / "11_concat_flows_along_time.py"


def _make_chunk(tmp_path, name, it_start, nit, omegas):
    it = np.arange(it_start, it_start + nit)
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(it * 150, unit="s")
    vx = it[:, None, None].astype(np.float32) + np.zeros((nit, 3, 3), dtype=np.float32)
    ds = xr.Dataset(
        data_vars={"vx": (["it", "omega", "y", "x"], np.broadcast_to(vx[:, None], (nit, len(omegas), 3, 3)).copy())},
        coords={
            "it": ("it", it), "time": ("it", time),
            "omega": ("omega", np.array(omegas, dtype=float)),
            "y": ("y", np.arange(3, dtype=float), {"units": "km"}),
            "x": ("x", np.arange(3, dtype=float), {"units": "km"}),
        },
        attrs=dict(revrot=omegas),
    )
    p = tmp_path / name
    ds.to_netcdf(p)
    return p


def test_concat_along_time_preserves_omega_dim(tmp_path):
    omegas = [0.0, 0.0005]
    p1 = _make_chunk(tmp_path, "chunk1.nc", it_start=0, nit=3, omegas=omegas)
    p2 = _make_chunk(tmp_path, "chunk2.nc", it_start=3, nit=3, omegas=omegas)

    combined = opentcamv.concat.concat_along_time([str(p1), str(p2)])

    assert "omega" in combined["vx"].dims
    np.testing.assert_allclose(combined["omega"].values, omegas)
    assert combined.sizes["it"] == 6
    np.testing.assert_array_equal(combined["vx"].isel(omega=0).values[:, 0, 0], np.arange(6))


def test_11_cli_matches_direct_function_call(tmp_path):
    omegas = [0.0, 0.0005]
    p1 = _make_chunk(tmp_path, "chunk1.nc", it_start=0, nit=3, omegas=omegas)
    p2 = _make_chunk(tmp_path, "chunk2.nc", it_start=3, nit=3, omegas=omegas)
    expected = opentcamv.concat.concat_along_time([str(p1), str(p2)])

    out = tmp_path / "out.nc"
    argv = [sys.executable, str(SCRIPT_11), "-t", str(p1), str(p2), "-o", str(out)]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    actual = xr.open_dataset(out)
    np.testing.assert_array_equal(actual["vx"].values, expected["vx"].values)
    np.testing.assert_allclose(actual["omega"].values, omegas)
