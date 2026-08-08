"""opentcamv.data: time-axis construction and search-radius derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from opentcamv import data
import pyvttrac as vt


def test_build_time_seconds_handles_spans_over_24h():
    # A >24h span is the regression case for the v1 `.dt.seconds` bug:
    # it wraps to 0 every 24h instead of accumulating.
    times = pd.date_range("2020-01-01", periods=4, freq="20h")
    frames = xr.Dataset(coords={"t": times})
    t = data.build_time_seconds(frames, "t")
    np.testing.assert_allclose(t, [0.0, 20 * 3600, 40 * 3600, 60 * 3600])


def test_build_time_seconds_short_span_matches_naive_seconds():
    times = pd.date_range("2020-01-01", periods=5, freq="150s")
    frames = xr.Dataset(coords={"t": times})
    t = data.build_time_seconds(frames, "t")
    np.testing.assert_allclose(t, [0.0, 150.0, 300.0, 450.0, 600.0])


class _Args:
    def __init__(self, **kw):
        self.hs = None
        self.Vs = 40.0
        self.itstep = 1
        self.__dict__.update(kw)


def test_compute_search_radius_scales_with_itstep():
    grid = vt.Grid(x0=0.0, y0=0.0, dx=1.0, dy=1.0, unit_factor=1000.0)
    r1 = data.compute_search_radius(_Args(itstep=1), grid, ref_dt=150.0)
    r2 = data.compute_search_radius(_Args(itstep=2), grid, ref_dt=150.0)
    # A tracking step covers `itstep` frames' worth of time, so the search
    # radius for itstep=2 must be derived from 2x the per-frame ref_dt --
    # roughly double r1's radius, not equal to it.
    assert r2[0] > r1[0]
    assert r2 == data.compute_search_radius(_Args(itstep=1), grid, ref_dt=300.0)


def test_compute_search_radius_hs_overrides_vs():
    grid = vt.Grid(x0=0.0, y0=0.0, dx=1.0, dy=1.0, unit_factor=1000.0)
    r = data.compute_search_radius(_Args(hs=7), grid, ref_dt=150.0)
    assert r == (7, 7)
