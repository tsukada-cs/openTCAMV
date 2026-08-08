"""--out_cthmin/--out_cthmax gating (v1 checked out_cthmax twice, never out_cthmin)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from opentcamv import records


class _Args:
    def __init__(self, **kw):
        self.record_initpos = None
        self.record_alongtraj = None
        self.out_cthmin = False
        self.out_cthmax = False
        self.cth = "cth"
        self.polar = False
        self.__dict__.update(kw)


def _frames():
    t = pd.date_range("2020-01-01", periods=4, freq="150s")
    y = np.arange(-5.0, 6.0)
    x = np.arange(-5.0, 6.0)
    cth = xr.DataArray(np.full((4, 11, 11), 5.0), dims=["t", "y", "x"], coords={"t": t, "y": y, "x": x})
    return xr.Dataset({"cth": cth})


def _ofl():
    it = [1, 2]
    it_rel = [-1, 0, 1]
    dim1 = dim2 = np.array([0.0, 1.0])
    xloc = xr.DataArray(
        np.zeros((2, 3, 2, 2)), dims=["it", "it_rel", "y", "x"],
        coords={"it": it, "it_rel": it_rel, "y": dim1, "x": dim2},
    )
    yloc = xloc.copy()
    time2 = xr.DataArray(
        np.array([pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=150 * k) for k in [0, 1, 2, 1, 2, 3]]).reshape(2, 3),
        dims=["it", "it_rel"], coords={"it": it, "it_rel": it_rel},
    )
    return xr.Dataset({"xloc": xloc, "yloc": yloc, "time2": time2})


def test_record_initpos_matches_target_dim_order():
    # Regression: interp()'s natural output dim order isn't guaranteed to
    # already match `ofl[varname]`'s -- the transpose must be explicit,
    # not assumed.
    t = pd.date_range("2020-01-01", periods=3, freq="150s")
    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0])
    # A field that's easy to check positionally: value = 100*y_index + x_index (independent of t).
    yv, xv = np.meshgrid(y, x, indexing="ij")
    data = (100 * yv + xv)[None, :, :].repeat(3, axis=0).astype(np.float64)
    src = xr.DataArray(data, dims=["t", "y", "x"], coords={"t": t, "y": y, "x": x})
    frames = xr.Dataset({"V": src})

    tg = np.array([0, 1])
    ofl = xr.Dataset(
        {"V": (["it", "y", "x"], np.zeros((2, 3, 2)))},
        coords={"it": [0, 1], "y": y, "x": x},
    )
    out = records.record_initpos(ofl, _Args(record_initpos=["V"]), frames, "t", "y", "x", tg)
    expected = np.broadcast_to((100 * yv + xv)[None, :, :], (2, 3, 2))
    np.testing.assert_allclose(out["V"].values, expected)


def test_out_cthmin_alone_is_not_a_noop():
    ofl = records.record_alongtraj_and_cth(_ofl(), _Args(out_cthmin=True), _frames(), "t", "y", "x", np.zeros((2, 2)), np.zeros((2, 2)))
    assert "cthmin" in ofl
    assert "cthmax" not in ofl


def test_out_cthmax_alone_still_works():
    ofl = records.record_alongtraj_and_cth(_ofl(), _Args(out_cthmax=True), _frames(), "t", "y", "x", np.zeros((2, 2)), np.zeros((2, 2)))
    assert "cthmax" in ofl
    assert "cthmin" not in ofl


def test_neither_flag_is_a_noop():
    before = _ofl()
    ofl = records.record_alongtraj_and_cth(before.copy(deep=True), _Args(), _frames(), "t", "y", "x", np.zeros((2, 2)), np.zeros((2, 2)))
    assert set(ofl.data_vars) == set(before.data_vars)
