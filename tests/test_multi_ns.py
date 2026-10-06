"""`10_conduct_tracking.py` with multiple `--ns` values: one process, one
output file per `ns` (`<ns>` placeholder in `-o`), each in the same
per-Omega layout a single-`ns` run would write. The gate is numeric parity
with separate single-`ns` runs, plus `20_finalize_tracking.py`'s loader
reading the files as-is.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import opentcamv
from conftest import SCRIPT_10, make_uniform_advection_frames

NT, NY, NX = 6, 41, 41
DX = DY = 0.5
X0 = Y0 = -10.0
DT_SEC = 150.0
VARNAME = "Z"
NS_LIST = [7, 9]
REVROTS = [0.0, 0.0007]
TRACKED_VARS = ("vx", "vy", "xloc", "yloc", "score")

COMMON_ARGS = [
    "--revrot", *[str(o) for o in REVROTS],
    "--ntrac", "1", "--Sth0", "0.5", "--Sth1", "0.5", "--Cth", "0",
    "--Vs", "20", "--Vc", "20", "--Vd", "0", "--vlim", "0",
    "--xgran=-6:6", "--ygran=-6:6", "--xint", "3", "--yint", "3",
    "--traj_int", "1",
]


def _run_10(ifn, ofn, extra_args):
    argv = [sys.executable, str(SCRIPT_10), str(ifn), "--varname", VARNAME, "-o", str(ofn), *extra_args]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"10_conduct_tracking.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Every `10_` run the e2e tests compare, done once: multi-`ns` (combined
    and --split_omega) and one single-`ns` run per value."""
    tmp_path = tmp_path_factory.mktemp("multi_ns")
    frames = make_uniform_advection_frames(NT, NY, NX, DX, DY, X0, Y0, DT_SEC, 12.0, -6.0, VARNAME)
    ifn = tmp_path / "in.nc"
    frames.to_netcdf(ifn)

    ns_args = ["--ns", *[str(ns) for ns in NS_LIST]]
    multi_rule = str(tmp_path / "multi_ns<ns>.nc")
    _run_10(ifn, multi_rule, [*ns_args, *COMMON_ARGS])
    split_rule = str(tmp_path / "split_ns<ns>_rot<omega>.nc")
    _run_10(ifn, split_rule, [*ns_args, "--split_omega", *COMMON_ARGS])
    single = {}
    for ns in NS_LIST:
        single[ns] = tmp_path / f"single_ns{ns}.nc"
        _run_10(ifn, single[ns], ["--ns", str(ns), *COMMON_ARGS])
    return {"multi_rule": multi_rule, "split_rule": split_rule, "single": single}


def test_multi_ns_matches_single_ns_runs(runs):
    for ns in NS_LIST:
        multi = xr.open_dataset(runs["multi_rule"].replace("<ns>", str(ns)))
        single = xr.open_dataset(runs["single"][ns])
        assert "omega" in multi["vx"].dims
        np.testing.assert_allclose(multi["omega"].values, REVROTS)
        for var in TRACKED_VARS:
            np.testing.assert_array_equal(multi[var].values, single[var].values)
        assert multi.attrs["ns"] == multi.attrs["nsx"] == multi.attrs["nsy"] == ns

    # Guard against every file silently using the same template size.
    score7 = xr.open_dataset(runs["multi_rule"].replace("<ns>", "7"))["score"].values
    score9 = xr.open_dataset(runs["multi_rule"].replace("<ns>", "9"))["score"].values
    assert not np.array_equal(score7, score9, equal_nan=True)


def test_multi_ns_split_omega_writes_one_file_per_pair(runs):
    for ns in NS_LIST:
        multi = xr.open_dataset(runs["multi_rule"].replace("<ns>", str(ns)))
        for omega in REVROTS:
            path = Path(runs["split_rule"].replace("<ns>", str(ns)).replace("<omega>", f"{omega:.4f}"))
            assert path.exists()
            split_ds = xr.open_dataset(path)
            for var in TRACKED_VARS:
                np.testing.assert_array_equal(split_ds[var].values, multi.sel(omega=omega)[var].values)
            assert split_ds.attrs["ns"] == ns
            assert split_ds.attrs["revrot"] == omega


def test_multi_ns_output_loads_in_finalize(runs):
    """`20_`'s format 2 (`<ns>`-only rule, `omega` read from each file)."""
    from opentcamv.finalize.io import load_candidates

    flows_org, *_ = load_candidates(runs["multi_rule"], NS_LIST)
    np.testing.assert_array_equal(flows_org["ns"].values, NS_LIST)
    np.testing.assert_allclose(flows_org["omega"].values, REVROTS)
    for ns in NS_LIST:
        single = xr.open_dataset(runs["single"][ns])
        np.testing.assert_array_equal(flows_org["vx"].sel(ns=ns).transpose(*single["vx"].dims).values, single["vx"].values)


def _parse(argv):
    parser = opentcamv.cli.build_parser()
    return opentcamv.cli.normalize_args(parser.parse_args(["in.nc", *argv]))


def test_ns_defaults_to_single_value_list():
    assert _parse([]).ns == [11]


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--ns", "7", "9", "-o", "out.nc"], "<ns>"),
        (["--ns", "7", "9", "--nsx", "5", "-o", "out_ns<ns>.nc"], "nsx"),
        (["--ns", "7", "7", "-o", "out_ns<ns>.nc"], "unique"),
    ],
)
def test_invalid_multi_ns_args_are_rejected(argv, match):
    with pytest.raises(ValueError, match=match):
        _parse(argv)


def test_args_for_ns_narrows_a_copy():
    args = _parse(["--ns", "7", "9", "--revrot", "0", "0.001", "--split_omega", "-o", "out_ns<ns>_rot<omega>.nc"])
    args_ns = opentcamv.cli.args_for_ns(args, 9)
    assert (args_ns.ns, args_ns.nsx, args_ns.nsy) == (9, 9, 9)
    assert args_ns.ofn == "out_ns9_rot<omega>.nc"

    args_ns.ref_dt = 150.0
    assert args.ns == [7, 9]
    assert args.ofn == "out_ns<ns>_rot<omega>.nc"
    assert args.ref_dt is None


def test_args_for_ns_keeps_nsx_nsy_for_single_ns():
    args = _parse(["--ns", "7", "--nsx", "5"])
    args_ns = opentcamv.cli.args_for_ns(args, 7)
    assert (args_ns.nsx, args_ns.nsy) == (5, 7)
    assert args_ns.ofn == args.ofn
