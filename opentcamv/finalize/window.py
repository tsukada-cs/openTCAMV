"""Median-filter window sizes, in index units, for the iterative selector."""

from __future__ import annotations


def window_sizes(
    Tw: float, Hw: float, xw: "float | None", yw: "float | None", rw: float, aw: float,
    dtmean: float, xint: float, yint: float, rint: float, nath: int, polar: bool, dim1: str, dim2: str,
) -> dict:
    """`{"it": ..., dim1: ..., dim2: ...}`, each an odd window length
    (`(n // 2) * 2 + 1`).

    F3 (fixed): the pre-refactor script assigned the `xw`-derived width to
    `dim1` (`"y"` in Cartesian mode) and the `yw`-derived width to `dim2`
    (`"x"`) -- swapped. With the default `xw == yw` (both fall back to `Hw`)
    and `xint == yint`, this didn't change the result, which is how it went
    unnoticed. `dim1` ("y") now gets the y-axis width (`yw`/`yint`), `dim2`
    ("x") the x-axis width (`xw`/`xint`).

    F4 (fixed): `--rw`/`--aw` (the polar-grid window width/azimuth-width
    CLI args) were defined but never used -- `--polar` runs got `xw`/`yw`
    (km-based) applied to the `r`/`a` axes regardless, and since `a` is in
    radians, that produced a meaningless azimuth window. `--polar` now uses
    `rw`/`rint` for the `r` axis and `aw` (degrees) converted via the
    azimuth grid's own spacing (`360 / nath` degrees per step, matching how
    `output.build_seed_positions` lays the azimuth grid out) for the `a`
    axis. Cartesian is unaffected.
    """
    iTw = int(Tw // dtmean)
    if not polar:
        xw = xw if xw is not None else Hw
        yw = yw if yw is not None else Hw
        ixw = int(xw // xint)
        iyw = int(yw // yint)
        return {"it": (iTw // 2) * 2 + 1, dim1: (iyw // 2) * 2 + 1, dim2: (ixw // 2) * 2 + 1}

    irw = int(rw // rint)
    azimuth_step_deg = 360.0 / nath
    iaw = int(aw // azimuth_step_deg)
    return {"it": (iTw // 2) * 2 + 1, dim1: (irw // 2) * 2 + 1, dim2: (iaw // 2) * 2 + 1}
