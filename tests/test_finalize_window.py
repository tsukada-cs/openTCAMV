"""`opentcamv.finalize.window.window_sizes`.

F3: the pre-refactor script swapped the x/y median-filter window widths
(`xw`-derived width landed on `dim1`="y", `yw`-derived on `dim2`="x").
Invisible under the default `xw == yw` (both fall back to `Hw`) and
`xint == yint`, so a regression test needs an asymmetric case to pin it.

F4: `--rw`/`--aw` (the polar-grid window width/azimuth-width) were defined
but never used -- `--polar` runs got the Cartesian `xw`/`yw` (km-based)
window applied to the `r`/`a` axes regardless, and since `a` is in radians,
that made the azimuth window meaningless.
"""

from __future__ import annotations

from opentcamv.finalize.window import window_sizes


def _cartesian(**overrides):
    kwargs = dict(
        Tw=1210, Hw=6, xw=None, yw=None, rw=6, aw=30,
        dtmean=150.0, xint=1.0, yint=1.0, rint=1.0, nath=60, polar=False, dim1="y", dim2="x",
    )
    kwargs.update(overrides)
    return window_sizes(**kwargs)


def _polar(**overrides):
    kwargs = dict(
        Tw=1210, Hw=6, xw=None, yw=None, rw=6, aw=30,
        dtmean=150.0, xint=1.0, yint=1.0, rint=1.0, nath=60, polar=True, dim1="r", dim2="a",
    )
    kwargs.update(overrides)
    return window_sizes(**kwargs)


def test_symmetric_default_unaffected_by_fix():
    """xw == yw and xint == yint (the common case, and every existing
    golden fixture): x/y being swapped or not makes no difference."""
    assert _cartesian() == {"it": 9, "y": 7, "x": 7}


def test_asymmetric_xy_not_swapped():
    """F3: dim1 ("y") must get the y-axis width (yw/yint), dim2 ("x") the
    x-axis width (xw/xint) -- not swapped."""
    sizes = _cartesian(xw=10.0, yw=4.0)
    # y window derived from yw=4.0/yint=1.0 -> iyw=4 -> odd window 5
    # x window derived from xw=10.0/xint=1.0 -> ixw=10 -> odd window 11
    assert sizes["y"] == 5
    assert sizes["x"] == 11


def test_asymmetric_xint_yint_not_swapped():
    """Same asymmetry, but driven by xint/yint instead of xw/yw."""
    sizes = _cartesian(xint=2.0, yint=1.0)
    # Hw=6 for both axes: y window = 6//1.0=6 -> odd 7; x window = 6//2.0=3 -> odd 3
    assert sizes["y"] == 7
    assert sizes["x"] == 3


def test_polar_uses_rw_aw_not_cartesian_xw_yw():
    """F4: --polar must size its window from --rw/--aw (and rint/nath),
    never from --xw/--yw -- giving xw/yw wildly different values from
    rw/aw should have zero effect on a polar window."""
    baseline = _polar()
    perturbed = _polar(xw=999.0, yw=999.0)
    assert baseline == perturbed


def test_polar_r_window_from_rw_rint():
    sizes = _polar(rw=4.0, rint=1.0)
    # irw = 4.0 // 1.0 = 4 -> odd window 5
    assert sizes["r"] == 5


def test_polar_azimuth_window_from_aw_degrees():
    """aw is in degrees; the azimuth axis spacing is 360/nath degrees per
    step (matching output.build_seed_positions's linspace(0, 2*pi, nath))."""
    sizes = _polar(aw=60.0, nath=60)
    # azimuth step = 360/60 = 6 deg/step; iaw = 60//6 = 10 -> odd window 11
    assert sizes["a"] == 11


def test_polar_azimuth_window_scales_with_nath():
    sizes_coarse = _polar(aw=60.0, nath=12)  # step = 30 deg; iaw = 60//30=2 -> odd 3
    sizes_fine = _polar(aw=60.0, nath=120)  # step = 3 deg; iaw = 60//3=20 -> odd 21
    assert sizes_coarse["a"] == 3
    assert sizes_fine["a"] == 21
