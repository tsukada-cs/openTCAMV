"""`opentcamv.finalize.io.load_candidates` accepting `omega`-dimensioned input 
(the new `10_conduct_tracking.py` default output), not just the legacy 
one-file-per-(ns, omega) format.

The gate is numeric parity: reformatting the *same* candidate
data into the new shapes must not change `20_finalize_tracking.py`'s output
at all. Each test here builds the legacy per-(ns, omega) files via the usual
`finalize_conftest.make_finalize_candidates` fixture, repacks them (with
`xr.concat`, mirroring what `opentcamv.output.combine_omega` does to
`10_`'s per-Omega Datasets) into one of the new formats, and asserts
byte-identical output against the *same* legacy-format run (not the frozen
golden -- these tests care about format equivalence, not about pinning the
selection algorithm itself, which is already covered by
test_finalize_regression.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from finalize_conftest import make_finalize_candidates

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "20_finalize_tracking.py"


def _run(argv_tail, tmp_path, out_name="out.nc"):
    out = tmp_path / out_name
    argv = [sys.executable, str(SCRIPT), *argv_tail, "-o", str(out)]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"20_finalize_tracking.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return xr.open_dataset(out)


def _assert_datasets_equal(a: xr.Dataset, b: xr.Dataset):
    assert set(a.data_vars) == set(b.data_vars), (set(a.data_vars), set(b.data_vars))
    for var in a.data_vars:
        av, bv = a[var].values, b[var].values
        if av.dtype.kind in "fc":
            assert np.array_equal(av, bv, equal_nan=True), f"{var} differs"
        else:
            assert np.array_equal(av, bv), f"{var} differs"


def _combine_into_single_omega_file(paths, omegas, out_path):
    """Mimics `opentcamv.output.combine_omega`'s effect on `10_`'s per-Omega
    Datasets, but starting from already-written per-omega candidate files
    (as the test fixture produces) rather than in-memory Datasets."""
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim="omega")
    combined = combined.assign_coords(omega=np.array(omegas, dtype=float))
    combined.to_netcdf(out_path)
    return out_path


EXTRA_ARGS = ["--cthmax", "10", "--exclude", "stf", "stb", "score_ary", "psr", "--out_final_omega"]


def test_single_combined_omega_file_matches_legacy(tmp_path):
    """Format 1: a single file, `omega` as a dimension, no placeholders."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(legacy_dir, n_omega=4, ns=7)
    legacy_ds = _run(
        [ifns_rule, "--ns", "7", "--omega", *omega_strs, *EXTRA_ARGS], tmp_path, out_name="legacy.nc"
    )

    combined_fn = _combine_into_single_omega_file(paths, omegas, tmp_path / "combined.nc")
    new_ds = _run([str(combined_fn), "--ns", "7", *EXTRA_ARGS], tmp_path, out_name="new.nc")

    _assert_datasets_equal(legacy_ds, new_ds)


def test_single_combined_omega_file_omega_still_checked_against_cli(tmp_path):
    """--omega, if given for format 1/2, must be cross-checked against the
    file's own `omega` dimension rather than silently trusted or ignored."""
    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=4, ns=7)
    combined_fn = _combine_into_single_omega_file(paths, omegas, tmp_path / "combined.nc")

    argv = [
        sys.executable, str(SCRIPT), str(combined_fn), "--ns", "7", "--omega", "0.9999", *EXTRA_ARGS,
        "-o", str(tmp_path / "out.nc"),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not match" in result.stderr


def test_per_ns_omega_dim_files_match_legacy_multi_ns(tmp_path):
    """Format 2: one `omega`-dimensioned file per `ns`, `--omega` omitted
    (read from each file). Reuses the same ns=7/ns=9 "rescue" scenario as
    test_finalize_candidates.test_larger_ns_rescues_point_invalid_at_smaller_ns."""
    ifns_rule7, omega_strs, omegas, paths7 = make_finalize_candidates(tmp_path, n_omega=3, ns=7, fname_prefix="a")
    ifns_rule9, _, _, paths9 = make_finalize_candidates(tmp_path, n_omega=3, ns=9, fname_prefix="a", seed=1)

    legacy_ds = _run(
        [ifns_rule7, "--ns", "7", "9", "--omega", *omega_strs, *EXTRA_ARGS],
        tmp_path, out_name="legacy.nc",
    )

    combined7 = _combine_into_single_omega_file(paths7, omegas, tmp_path / "combined_ns7.nc")
    combined9 = _combine_into_single_omega_file(paths9, omegas, tmp_path / "combined_ns9.nc")
    new_rule = str(tmp_path / "combined_ns<ns>.nc")
    new_ds = _run([new_rule, "--ns", "7", "9", *EXTRA_ARGS], tmp_path, out_name="new.nc")

    _assert_datasets_equal(legacy_ds, new_ds)


def test_single_omega_file_without_omega_dim_is_synthesized(tmp_path):
    """A single-Omega `10_` run never gets an `omega` dimension at all
    (matching v1's schema exactly) -- `load_candidates` must synthesize a
    length-1 one from the file's `revrot` attr rather than erroring."""
    from opentcamv.finalize.io import load_candidates

    ifns_rule, omega_strs, omegas, paths = make_finalize_candidates(tmp_path, n_omega=1, ns=7)
    ds = xr.open_dataset(paths[0])
    assert "omega" not in ds.dims
    ds.attrs["revrot"] = omegas[0]
    single_fn = tmp_path / "single_no_omega_dim.nc"
    ds.to_netcdf(single_fn)

    flows_org, flows0, names, candidate_independent_keepvars, candidate_dependent_keepvars, used_vars = load_candidates(
        str(single_fn), [7]
    )
    assert flows_org.sizes["omega"] == 1
    assert flows_org["omega"].item() == omegas[0]
