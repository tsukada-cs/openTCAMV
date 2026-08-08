"""--vlim / --Vd / --Td / --Vth screening, and the status-10 write-back."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from opentcamv import screening
from opentcamv.output import AxisNames

NAMES = AxisNames("y", "x", "yloc", "xloc", "vy", "vx")


class _Args:
    def __init__(self, **kw):
        self.ward = "bothward"
        self.vlim = 0
        self.Vd = 0
        self.Td = None
        self.Vth = 5.0
        self.vagg = "mean"
        self.__dict__.update(kw)


def _make_ofl(vx, vy, vxfm, vyfm, vxbm, vybm):
    it_rel = xr.DataArray([-1, 0, 1], dims="it_rel", coords={"it_rel": [-1, 0, 1]})
    dims = ["it", "y", "x"]
    ofl = xr.Dataset(
        {
            "vx": (dims, vx), "vy": (dims, vy),
            "vxfm": (dims, vxfm), "vyfm": (dims, vyfm),
            "vxbm": (dims, vxbm), "vybm": (dims, vybm),
            "stf": (dims, np.zeros_like(vx, dtype=np.int16)),
            "stb": (dims, np.zeros_like(vx, dtype=np.int16)),
            "yloc": (["it", "it_rel", "y", "x"], np.zeros((*vx.shape[:1], 3, *vx.shape[1:]))),
            "xloc": (["it", "it_rel", "y", "x"], np.zeros((*vx.shape[:1], 3, *vx.shape[1:]))),
        },
        coords={"it_rel": it_rel},
    )
    return ofl


def test_vlim_rejects_fast_points():
    shape = (1, 1, 3)
    vx = np.array([[[0.0, 30.0, 100.0]]])
    vy = np.zeros(shape)
    ofl = _make_ofl(vx, vy, vx, vy, vx, vy)
    args = _Args(vlim=50.0, Vd=0)
    out = screening.screen(ofl, args, NAMES)
    np.testing.assert_array_equal(np.isnan(out["vx"].values[0, 0]), [False, False, True])


def test_vd_rejects_large_forward_backward_disagreement():
    shape = (1, 1, 2)
    vxfm = np.array([[[0.0, 0.0]]])
    vxbm = np.array([[[0.0, 100.0]]])  # second point disagrees by 100 m/s
    vyfm = np.zeros(shape)
    vybm = np.zeros(shape)
    vx = (vxfm + vxbm) / 2
    vy = np.zeros(shape)
    ofl = _make_ofl(vx, vy, vxfm, vyfm, vxbm, vybm)
    args = _Args(Vd=20.0)
    out = screening.screen(ofl, args, NAMES)
    valid = np.isfinite(out["vx"].values[0, 0])
    np.testing.assert_array_equal(valid, [True, False])


def test_status_10_written_only_where_previously_ok():
    shape = (1, 1, 2)
    vx = np.array([[[0.0, 100.0]]])
    vy = np.zeros(shape)
    ofl = _make_ofl(vx, vy, vx, vy, vx, vy)
    ofl["stf"].values[0, 0, 1] = 3  # already-failed point keeps its own status
    args = _Args(vlim=50.0, Vd=0)
    out = screening.screen(ofl, args, NAMES)
    assert out["stf"].values[0, 0, 0] == 0  # valid point untouched
    assert out["stf"].values[0, 0, 1] == 3  # was already non-zero; screening doesn't overwrite it


def test_vxfm_etc_dropped_after_bothward_screening():
    shape = (1, 1, 2)
    vx = np.zeros(shape)
    ofl = _make_ofl(vx, vx, vx, vx, vx, vx)
    out = screening.screen(ofl, _Args(), NAMES)
    for var in ("vxfm", "vyfm", "vxbm", "vybm"):
        assert var not in out.data_vars


def test_no_screening_is_noop_when_all_thresholds_disabled():
    shape = (1, 1, 3)
    vx = np.array([[[0.0, 30.0, 100.0]]])
    vy = np.zeros(shape)
    ofl = _make_ofl(vx, vy, vx, vy, vx, vy)
    out = screening.screen(ofl, _Args(vlim=0, Vd=0, Td=None), NAMES)
    np.testing.assert_array_equal(out["vx"].values, vx)


def test_restore_revrot_noop_for_zero_omega():
    shape = (1, 1, 2)
    vx = np.array([[[1.0, 2.0]]])
    ofl = _make_ofl(vx, vx, vx, vx, vx, vx)
    ofl = ofl.assign_coords(x=("x", [1.0, 2.0]), y=("y", [3.0]))
    ofl["time2"] = (["it", "it_rel"], np.zeros((1, 3), dtype="datetime64[s]"))
    before = ofl["vx"].values.copy()
    screening.restore_revrot(ofl, _Args(), 0.0, NAMES)
    np.testing.assert_array_equal(ofl["vx"].values, before)
